# Qwen3-MoE Training Architecture (v2)

Full trace of the Qwen3-MoE v2 training pipeline on Ascend NPU with MindSpeed-LLM + Megatron-LM. This is the scaled-up variant with larger hidden dimensions, fewer experts, and narrower top-k compared to v1.

## Entry Point

**File**: `train_fault_injection_qwen3_moe_v2_monitor.sh` (and sibling scripts `train_shrunk_qwen3_moe_v2.sh`, `train_shrunk_qwen3_moe_v2_monitor.sh`, `train_fault_injection_qwen3_moe_v2.sh`)

Launches distributed training via `torch.distributed.launch` with 8 NPUs (single-node). The Python entry is:

```
/data02/npu_stablity/LLM/MindSpeed-LLM/pretrain_gpt.py
```

Shrunk-model config (smoke-test scale):

| Param | v2 Value | v1 Value (for reference) |
|-------|----------|--------------------------|
| layers | 16 | 8 |
| hidden_size | 1536 | 1024 |
| attention_heads | 16 | 16 |
| query_groups (GQA) | 4 | 4 |
| ffn_hidden_size | 1536 | 1024 |
| num_experts | 64 | 128 |
| moe_router_topk | 6 | 8 |
| moe_ffn_hidden_size | 2048 | 1024 |
| seq_length | 4096 | 4096 |
| MBS / GBS | 8 / 64 | 8 / 64 |
| TP / PP / EP / CP | 1 / 1 / 8 / 1 | 1 / 1 / 8 / 1 |
| precision | bf16 | bf16 |
| train_iters | 10000 | 10000 |
| shared expert | disabled (default `None`) | disabled |
| untied embed/output | yes | yes |
| qk_layernorm | yes | yes |

The spec is loaded at runtime via `--spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec`, which calls `import_module(...)` and retrieves the `layer_spec` variable.

---

## Model Initialization

**File**: `pretrain_gpt.py`, function `model_provider()` (line 42)

```
model_provider()
  -> import_module(args.spec)          # loads qwen3_spec.layer_spec
  -> GPTModel(config, transformer_layer_spec=layer_spec, ...)
```

The MindSpeed `GPTModel` replaces the standard Megatron `GPTModel` at runtime via the feature_manager patch system:

```
pm.register_patch('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel)
```

### GPTModel.__init__ (MindSpeed override)

**File**: `MindSpeed-LLM/mindspeed_llm/core/models/gpt/gpt_model.py`

1. **Embedding** (`pre_process=True`): `LanguageModelEmbedding` — vocab embedding, position embedding (when not using RoPE), with sequence-parallel scatter support.

2. **RoPE**: `RotaryEmbedding(kv_channels, rotary_percent, rotary_base=1000000)` — precomputes cos/sin for rotary positional encoding. The result is passed through every `TransformerLayer` for Q/K rotation.

3. **Decoder**: `TransformerBlock(config, spec=layer_spec)` — builds N `TransformerLayer` instances from the spec. With PP=1, all 16 layers live on one rank. The final RMSNorm is built inside `TransformerBlock` when `post_process=True` and `post_layer_norm=True`.

4. **Output layer** (`post_process=True`): `ColumnParallelLinear(hidden_size -> padded_vocab_size=151936)` — projects hidden states to logits. When `untie_embeddings_and_output_weights=True`, this is a separate weight matrix.

### How the Spec Builds Operators

**File**: `MindSpeed-LLM/mindspeed_llm/tasks/models/spec/qwen3_spec.py`

Since `--transformer-impl local` is set, `use_te=False`. The spec defines what each `TransformerLayer` contains:

```
TransformerLayer
  input_layernorm: PTNorm (MindSpeed RMSNorm)
  self_attention: SelfAttention
    linear_qkv: ColumnParallelLinear     # Q/K/V fused projection, TP-sharded
                                          # v2: 1536 × (1536 + 384 + 384) = 1536 × 2304
    core_attention: DotProductAttention  # scaled dot-product + causal mask + softmax
    linear_proj: RowParallelLinear       # output projection: 1536 × 1536
    q_layernorm: PTNorm                  # QK-LayerNorm (qk_layernorm=True)
    k_layernorm: PTNorm                  # QK-LayerNorm
  self_attn_bda: bias-dropout-add fusion # residual + dropout
  pre_mlp_layernorm: PTNorm             # RMSNorm before MoE
  mlp: MoELayer (via get_moe_module_spec)
    router: TopKRouter                   # top-6 token-to-expert routing + aux loss
                                          # v2: weight [64, 1536], no bias
    token_dispatcher: MoEAlltoAllTokenDispatcher  # all-to-all permute
    experts: GroupedMLP                  # 64 experts, GroupedGEMM
      linear_fc1: ColumnParallelLinear   # gate+up projection (SwiGLU, 2× moe_ffn_hidden)
                                          # v2: 1536 × 4096 per expert
      linear_fc2: RowParallelLinear      # down projection
                                          # v2: 2048 × 1536 per expert
    shared_experts: None                 # disabled (moe_shared_expert_intermediate_size=None)
  mlp_bda: bias-dropout-add fusion      # residual + dropout
```

When `moe_layer_freq=-1`, every layer uses MoE (no dense MLP alternation). The experts are sharded across 8 NPUs via EP=8, so each NPU holds 8 local experts (64 / 8). The `alltoall` token dispatcher permutes tokens so each NPU only computes its local experts.

### V1 vs V2 Expert Layout

| Aspect | v1 | v2 |
|--------|----|----|
| Local experts per NPU (EP=8) | 16 (128 / 8) | 8 (64 / 8) |
| Expert fc1 shape | 1024 × 2048 | 1536 × 4096 |
| Expert fc2 shape | 1024 × 1024 | 2048 × 1536 |
| Per-expert params | 3,145,728 | 9,437,184 |
| Per-NPU expert memory | ~50.3M | ~75.5M |

---

## Parameter Analysis

### Component Breakdown (per layer)

**Dense (always active):**

| Component | Shape | Parameters |
|-----------|-------|------------|
| Input RMSNorm | 1536 | 1,536 |
| QKV projection | 1536 × 2304 | 3,538,944 |
| Q RMSNorm | 1536 | 1,536 |
| K RMSNorm | 384 | 384 |
| Output projection | 1536 × 1536 | 2,359,296 |
| Pre-MLP RMSNorm | 1536 | 1,536 |
| Router weight | 64 × 1536 | 98,304 |
| **Dense per layer** | | **6,001,536** |

**Experts (GroupedMLP, SwiGLU, no bias):**

| Component | Shape | Per Expert |
|-----------|-------|------------|
| fc1 (gate + up) | 1536 × 4096 | 6,291,456 |
| fc2 (down) | 2048 × 1536 | 3,145,728 |
| **Per expert** | | **9,437,184** |
| All 64 experts (total) | | 603,979,776 |
| 6 activated (top-6) | | 56,623,104 |

**Global:**

| Component | Shape | Parameters |
|-----------|-------|------------|
| Token embedding | 151936 × 1536 | 233,373,696 |
| Final RMSNorm | 1536 | 1,536 |
| Output head (untied) | 1536 × 151936 | 233,373,696 |

### Summary

| Metric | Value |
|--------|-------|
| Dense per layer | 6,001,536 |
| Dense total (16 layers) | 96,024,576 |
| Expert total (16 layers × 64 experts) | 9,663,676,416 |
| Expert activated (16 layers × 6/64 experts) | 905,969,664 |
| Embedding + Output | 466,747,392 |
| **Total parameters** | **~10.23B** |
| **Activated parameters (per token, incl. embed+output)** | **~1.47B** |
| **Activated parameters (core only, excl. embed+output)** | **~1.00B** |
| Activation ratio (total) | 14.4% |
| Activation ratio (core) | 10.4% |

The embedding and output head account for 467M parameters — 32% of activated parameters. This is a consequence of the small hidden size (1536) paired with a large vocabulary (151936). Each expert weighs 9.4M parameters (3× the v1 expert size due to the larger hidden and ffn dimensions).

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

**File**: `mindspeed_llm/core/models/gpt/gpt_model.py`

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
    mlp_out = mlp(pre_mlp_normed)                                   # MoE
    hidden_states = mlp_bda(mlp_out, residual, dropout)             # residual add
    return hidden_states
```

### MoELayer.forward

**File**: `Megatron-LM/megatron/core/transformer/moe/moe_layer.py`

```
def forward(hidden_states):
    # 1. Router: compute token-to-expert assignments
    scores, indices = self.router(hidden_states)    # top-6 experts per token (v2)
    # also computes aux_loss for load balancing

    # 2. Token dispatch: all-to-all permute tokens to expert-parallel ranks
    dispatched_input, ... = self.token_dispatcher.dispatch(hidden_states, scores, indices)

    # 3. Expert computation: each rank computes its 8 local experts
    expert_output = self.experts(dispatched_input)

    # 4. Token combine: reverse all-to-all to restore original token order
    output, ... = self.token_dispatcher.combine(expert_output)

    # 5. Shared expert: SKIPPED (use_shared_expert=False)
    return output, router_loss
```

---

## Key Operator Modules

| Module | File | Purpose |
|--------|------|---------|
| `GPTModel` | `MindSpeed-LLM/mindspeed_llm/core/models/gpt/gpt_model.py` | Top-level model: embedding + decoder + output (MindSpeed override) |
| `TransformerBlock` | `Megatron-LM/megatron/core/transformer/transformer_block.py` | Stacks N TransformerLayer instances, builds final RMSNorm |
| `TransformerLayer` | `Megatron-LM/megatron/core/transformer/transformer_layer.py` | One transformer layer: attn + MLP |
| `SelfAttention` | `Megatron-LM/megatron/core/transformer/attention.py` | QKV projection, RoPE application, attention compute |
| `DotProductAttention` | `Megatron-LM/megatron/core/transformer/dot_product_attention.py` | Scaled dot-product + mask + softmax |
| `ColumnParallelLinear` | `Megatron-LM/megatron/core/tensor_parallel/layers.py` | TP-column-parallel linear (shards output dim) |
| `RowParallelLinear` | `Megatron-LM/megatron/core/tensor_parallel/layers.py` | TP-row-parallel linear (shards input dim) |
| `MLP` | `Megatron-LM/megatron/core/transformer/mlp.py` | Dense SwiGLU MLP |
| `MoELayer` | `Megatron-LM/megatron/core/transformer/moe/moe_layer.py` | MoE wrapper: router + dispatcher + experts |
| `TopKRouter` | `Megatron-LM/megatron/core/transformer/moe/router.py` | Token-to-expert top-k routing + aux loss |
| `MoEAlltoAllTokenDispatcher` | `Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py` | All-to-all token permutation |
| `GroupedMLP` | `Megatron-LM/megatron/core/transformer/moe/experts.py` | Batched expert computation with GroupedGEMM |
| `PTNorm` | `MindSpeed-LLM/mindspeed_llm/core/transformer/custom_layers/transformer_engine.py` | MindSpeed RMSNorm implementation |
| `RotaryEmbedding` | `Megatron-LM/megatron/core/models/common/embeddings/rotary_pos_embedding.py` | RoPE cos/sin computation |
| `LanguageModelEmbedding` | `Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py` | Word + position embeddings |

---

## Spec Construction Flow

```
qwen3_spec.py  (config: hidden=1536, num_experts=64, topk=6, moe_ffn_hidden=2048)
  -> get_moe_module_spec(num_experts=64, moe_grouped_gemm=True)
    -> MoELayer spec
      -> experts: GroupedMLP (GroupedGEMM with 64 experts)
      -> shared_experts: None (moe_shared_expert_intermediate_size default None)
  -> TransformerLayerSubmodules(
       input_layernorm=PTNorm,
       self_attention=SelfAttention(
         linear_qkv=ColumnParallelLinear,    # 1536 → 2304
         linear_proj=RowParallelLinear,      # 1536 → 1536
         q_layernorm=PTNorm,                 # 1536
         k_layernorm=PTNorm,                 # 384
       ),
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
    -> embedding(tokens, position_ids) -> [s, b, 1536]
    -> RoPE cos/sin -> rotary_pos_emb
    -> for layer in decoder.layers (16 layers):
         hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)
       where each layer:
         hidden_states = attn_bda(self_attention(norm(h)), residual)
           self_attention:
             qkv = linear_qkv(normed_h)           # 1536 → 2304
             q, k, v = split(qkv)                 # (1536, 384, 384)
             q, k = qk_layernorm(q), qk_layernorm(k)
             q, k = apply_rotary_pos_emb(q, k, rotary_pos_emb)
             attn_out = dot_product_attention(q, k, v, causal_mask)
             attn_out = linear_proj(attn_out)     # 1536 → 1536
         hidden_states = mlp_bda(moe_mlp(norm(h)), residual)
           moe_mlp (no shared expert):
             scores, indices = router(normed_h)   # [64] × [1536] → top6
             dispatched = alltoall_dispatch(normed_h, indices)
             expert_out = GroupedMLP(dispatched)  # 6/64 experts activated per token
               fc1: 1536 → 4096 (SwiGLU gate+up)
               fc2: 2048 → 1536 (down)
             output = alltoall_combine(expert_out)
    -> final_layernorm(hidden_states)             # TransformerBlock built-in
    -> output_layer(hidden_states) -> logits [s, b, 151936]
    -> cross_entropy(logits, labels, loss_mask) -> loss
  -> loss.backward()
  -> optimizer.step()
  -> lr_scheduler.step()
```
