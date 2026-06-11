# Qwen3-MoE Training Architecture

Full trace of the Qwen3-MoE training pipeline on Ascend NPU with MindSpeed-LLM + Megatron-LM.

## Entry Point

**File**: `train_shrunk_qwen3_moe.sh`

Launches distributed training via `torch.distributed.launch` with 8 NPUs (single-node). The Python entry is:

```
/data02/npu_stablity/LLM/MindSpeed-LLM/pretrain_gpt.py
```

Key shrunk-model config (smoke-test scale):

| Param | Value |
|-------|-------|
| layers | 16 |
| hidden_size | 1024 |
| attention_heads | 16 |
| query_groups (GQA) | 4 |
| num_experts | 128 |
| moe_router_topk | 8 |
| moe_ffn_hidden_size | 1024 |
| seq_length | 4096 |
| MBS / GBS | 8 / 64 |
| TP / PP / EP / CP | 1 / 1 / 8 / 1 |
| precision | bf16 |
| train_iters | 10000 |

The spec is loaded at runtime via `--spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec`, which calls `import_module(...)` and retrieves the `layer_spec` variable.

---

## Model Initialization

**File**: `pretrain_gpt.py`, function `model_provider()` (line 42)

```
model_provider()
  -> import_module(args.spec)          # loads qwen3_spec.layer_spec
  -> GPTModel(config, transformer_layer_spec=layer_spec, ...)
```

### GPTModel.__init__

**File**: `Megatron-LM/megatron/core/models/gpt/gpt_model.py` (line 74)

1. **Embedding** (`pre_process=True`): `LanguageModelEmbedding` — vocab embedding, position embedding (when not using RoPE), with sequence-parallel scatter support.

2. **RoPE**: `RotaryEmbedding(kv_channels, rotary_percent, rotary_base=1000000)` — precomputes cos/sin for rotary positional encoding. The result is passed through every `TransformerLayer` for Q/K rotation.

3. **Decoder**: `TransformerBlock(config, spec=layer_spec)` — builds N `TransformerLayer` instances from the spec. With PP=1, all 16 layers live on one rank.

4. **Output layer** (`post_process=True`): `ColumnParallelLinear(hidden_size -> padded_vocab_size=151936)` — projects hidden states to logits. When `untie_embeddings_and_output_weights=True`, this is a separate weight matrix.

### How the Spec Builds Operators

**File**: `MindSpeed-LLM/mindspeed_llm/tasks/models/spec/qwen3_spec.py`

Since `--transformer-impl local` is set, `use_te=False`. The spec defines what each `TransformerLayer` contains:

```
TransformerLayer
  input_layernorm: PTNorm (MindSpeed RMSNorm)
  self_attention: SelfAttention
    linear_qkv: ColumnParallelLinear     # Q/K/V fused projection, TP-sharded
    core_attention: DotProductAttention  # scaled dot-product + causal mask + softmax
    linear_proj: RowParallelLinear       # output projection, TP-sharded
    q_layernorm: PTNorm                  # QK-LayerNorm (qk_layernorm=True)
    k_layernorm: PTNorm                  # QK-LayerNorm
  self_attn_bda: bias-dropout-add fusion # residual + dropout
  pre_mlp_layernorm: PTNorm             # RMSNorm before MoE
  mlp: MoELayer (via get_moe_module_spec)
    router: TopKRouter                   # top-8 token-to-expert routing + aux loss
    token_dispatcher: MoEAlltoAllTokenDispatcher  # all-to-all permute
    experts: TEGroupedMLP / GroupedMLP   # 128 experts, GroupedGEMM
      linear_fc1: ColumnParallelLinear   # gate+up projection (SwiGLU, 2x ffn_hidden)
      linear_fc2: RowParallelLinear      # down projection
    shared_experts: SharedExpertMLP      # optional shared expert
  mlp_bda: bias-dropout-add fusion      # residual + dropout
```

When `moe_layer_freq=-1`, every layer uses MoE (no dense MLP alternation). The experts are sharded across 8 NPUs via EP=8, so each NPU holds 16 local experts. The `alltoall` token dispatcher permutes tokens so each NPU only computes its local experts.

---

## Training Loop

**File**: `MindSpeed-LLM/mindspeed_llm/training/training.py`

### Call chain

```
pretrain_gpt.main()
  -> pretrain(...)                         # line 423
    -> initialize_megatron(...)            # distributed init, args, timers
    -> build_train_args(...)               # line 337: model + optimizer + data setup
    -> train(...)                          # line 574: the main loop
```

### pretrain()

1. `initialize_megatron()` — torch distributed init, parse CLI args, NCCL backend.
2. `build_train_args()` — wraps `setup_model_and_optimizer()` from Megatron:
   - Creates DDP-wrapped model
   - Creates AdamW optimizer (beta1=0.9, beta2=0.95)
   - Creates cosine LR scheduler (lr=1e-3 -> min_lr=1e-5, warmup=1%)
   - Builds `BlendedMegatronDatasetBuilder` data iterators from `.bin/.idx` files
3. `train()` — the main while-loop.

### train() — the conventional training loop

**File**: `mindspeed_llm/training/training.py`, line 574

```python
while iteration < args.train_iters:
    update_num_microbatches(...)
    loss_dict, skipped_iter, ... = train_step(
        forward_step_func, data_iterator,
        model, optimizer, opt_param_scheduler, config
    )
    iteration += 1
    args.consumed_train_samples += batch_size

    # logging every log_interval (1)
    training_log(...)

    # evaluation every eval_interval
    evaluate_and_print_results(...)

    # checkpointing every save_interval
    save_checkpoint_and_time(...)

    # optional: manual GC every manual_gc_interval (50)
    gc.collect()
```

There is no `train_one_epoch`. The paradigm is fixed-iteration training (`TRAIN_ITERS=10000`).

---

## train_step — The Single Step

**File**: `Megatron-LM/megatron/training/training.py`, line 1213

```python
def train_step(forward_step_func, data_iterator, model, optimizer, ...):
```

Each step:

1. **Zero gradients**: `model_chunk.zero_grad_buffer()` + `optimizer.zero_grad()`
2. **Forward+Backward**: `forward_backward_func(...)` — runs the pipeline schedule. For PP=1, this is a simple loop over micro-batches:
   - Calls `forward_step_func` = `pretrain_gpt.forward_step()`
   - Runs `model(tokens, position_ids, attention_mask, labels, loss_mask)`
   - Computes loss, runs backward
3. **Optimizer step**: `optimizer.step()` — AdamW update
4. **LR scheduler**: `opt_param_scheduler.step(increment=...)` — cosine decay
5. **Loss reduction** (last pipeline stage only): averages loss across micro-batches and returns `loss_dict`

### forward_step

**File**: `pretrain_gpt.py`, line 216

```
forward_step(data_iterator, model):
  tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
  output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
  return output_tensor, partial(loss_func, loss_mask)
```

### GPTModel.forward

**File**: `Megatron-LM/megatron/core/models/gpt/gpt_model.py`, line 235

```
input_ids, position_ids
  -> self.embedding(input_ids, position_ids)           # token + position embed
  -> rotary_pos_emb = self.rotary_pos_emb(seq_len)      # RoPE cos/sin
  -> hidden_states = self.decoder(hidden_states,        # TransformerBlock
        attention_mask, rotary_pos_emb, ...)
  -> logits = self.output_layer(hidden_states)          # [s, b, h] -> [s, b, V]
  -> loss = compute_language_model_loss(labels, logits) # cross-entropy
  -> return loss
```

### TransformerLayer.forward

**File**: `Megatron-LM/megatron/core/transformer/transformer_layer.py`, line 382

```
def forward(hidden_states, attention_mask, rotary_pos_emb, ...):
    # Attention block
    residual = hidden_states
    input_normed = input_layernorm(hidden_states)
    attn_out = self_attention(input_normed, attention_mask, rotary_pos_emb, ...)
    hidden_states = self_attn_bda(attn_out, residual, dropout)     # residual add

    # MLP block
    residual = hidden_states
    pre_mlp_normed = pre_mlp_layernorm(hidden_states)
    mlp_out = mlp(pre_mlp_normed)                                   # MoE or dense
    hidden_states = mlp_bda(mlp_out, residual, dropout)             # residual add
    return hidden_states
```

### MoELayer.forward

**File**: `Megatron-LM/megatron/core/transformer/moe/moe_layer.py`

```
def forward(hidden_states):
    # 1. Router: compute token-to-expert assignments
    scores, indices = self.router(hidden_states)    # top-8 experts per token
    # also computes aux_loss for load balancing

    # 2. Token dispatch: all-to-all permute tokens to expert-parallel ranks
    dispatched_input, ... = self.token_dispatcher.dispatch(hidden_states, scores, indices)

    # 3. Expert computation: each rank computes its local experts
    expert_output = self.experts(dispatched_input)

    # 4. Token combine: reverse all-to-all to restore original token order
    output, ... = self.token_dispatcher.combine(expert_output)

    # 5. Shared expert (optional)
    if self.use_shared_expert:
        shared_output = self.shared_experts(hidden_states)
        output = output + shared_output

    return output, router_loss
```

---

## Key Operator Modules

| Module | File | Purpose |
|--------|------|---------|
| `GPTModel` | `Megatron-LM/megatron/core/models/gpt/gpt_model.py` | Top-level model: embedding + decoder + output |
| `TransformerBlock` | `Megatron-LM/megatron/core/transformer/transformer_block.py` | Stacks N TransformerLayer instances |
| `TransformerLayer` | `Megatron-LM/megatron/core/transformer/transformer_layer.py` | One transformer layer: attn + MLP |
| `SelfAttention` | `Megatron-LM/megatron/core/transformer/attention.py` | QKV projection, RoPE application, attention compute |
| `DotProductAttention` | `Megatron-LM/megatron/core/transformer/dot_product_attention.py` | Scaled dot-product + mask + softmax |
| `ColumnParallelLinear` | `Megatron-LM/megatron/core/tensor_parallel/layers.py` | TP-column-parallel linear (shards output dim) |
| `RowParallelLinear` | `Megatron-LM/megatron/core/tensor_parallel/layers.py` | TP-row-parallel linear (shards input dim) |
| `MLP` | `Megatron-LM/megatron/core/transformer/mlp.py` | Dense SwiGLU MLP |
| `MoELayer` | `Megatron-LM/megatron/core/transformer/moe/moe_layer.py` | MoE wrapper: router + dispatcher + experts |
| `TopKRouter` | `Megatron-LM/megatron/core/transformer/moe/router.py` | Token-to-expert top-k routing + aux loss |
| `MoEAlltoAllTokenDispatcher` | `Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py` | All-to-all token permutation |
| `GroupedMLP` / `TEGroupedMLP` | `Megatron-LM/megatron/core/transformer/moe/experts.py` | Batched expert computation with GroupedGEMM |
| `PTNorm` | `MindSpeed-LLM/mindspeed_llm/core/transformer/custom_layers/transformer_engine.py` | MindSpeed RMSNorm implementation |
| `RotaryEmbedding` | `Megatron-LM/megatron/core/models/common/embeddings/rotary_pos_embedding.py` | RoPE cos/sin computation |
| `LanguageModelEmbedding` | `Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py` | Word + position embeddings |

---

## Spec Construction Flow

```
qwen3_spec.py
  -> get_moe_module_spec(num_experts=128, moe_grouped_gemm=True)
    -> MoELayer spec
      -> experts: TEGroupedMLP (GroupedGEMM path) or GroupedMLP (fallback)
      -> shared_experts: SharedExpertMLP
  -> TransformerLayerSubmodules(
       input_layernorm=PTNorm,
       self_attention=SelfAttention(...),
       pre_mlp_layernorm=PTNorm,
       mlp=<MoELayer spec>,
       ...
     )
```

---

## Data Flow Summary

```
.bin/.idx files (Megatron indexed format)
  -> BlendedMegatronDatasetBuilder -> GPTDataset
  -> get_batch() -> (tokens, labels, loss_mask, attention_mask, position_ids)
  -> model.forward()
    -> embedding(tokens, position_ids) -> [s, b, h]
    -> RoPE cos/sin -> rotary_pos_emb
    -> for layer in decoder.layers:
         hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)
       where each layer:
         hidden_states = attn_bda(self_attention(norm(h)), residual)
         hidden_states = mlp_bda(moe_mlp(norm(h)), residual)
    -> output_layer(hidden_states) -> logits [s, b, V]
    -> cross_entropy(logits, labels, loss_mask) -> loss
  -> loss.backward()
  -> optimizer.step()
  -> lr_scheduler.step()
```
