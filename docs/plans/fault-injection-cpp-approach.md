# Fault Injection — C++ Approach (Reference)

## Summary

Inject faults into O (`attention_in`) and dO (`dy`) inside `npu_fusion_attention_grad`,
right before `npu_fusion_attention_backward` is called.
Forward pass is unaffected; both faults only hit the backward computation.

## File

```
third_party/torch_npu/third_party/op-plugin/op_plugin/ops/opapi/FlashAttentionKernelNpuOpApi.cpp
```

## Call chain

```
npu_fusion_attention (line 370)          ← forward: returns O, softmax_*, seed, offset, numels
npu_fusion_attention_grad (line 283)     ← backward entry: generates dropout mask, calls _backward
  └── npu_fusion_attention_backward (line 175) ← does format_trans, calls ACLNN kernels
```

## Injection point

In `npu_fusion_attention_grad` (line ~356), **after** dropout mask generation and **before**
the `npu_fusion_attention_backward` call:

```cpp
// --- FAULT INJECTION START ---
// Read env vars once at function entry (near line 312)
int bitshift_O = std::getenv("bitshift_fa_backward_O") ? std::stoi(std::getenv("bitshift_fa_backward_O")) : 0;
int bitshift_dO = std::getenv("bitshift_fa_backward_dO") ? std::stoi(std::getenv("bitshift_fa_backward_dO")) : 0;

// Corrupt attention_in (O) — affects O⊙dO in backward
if (bitshift_O > 0 && attention_in.has_value()) {
    // bf16/fp16: brief manipulation via int16 view; fp32: int32 view
    // mask = -1 << bitshift_O; then (raw & mask)
}

// Corrupt dy (dO) — affects gradient entering FA backward
if (bitshift_dO > 0) {
    // same logic on dy
}
// --- FAULT INJECTION END ---

auto result = npu_fusion_attention_backward(query,
    key, value, dy, head_num, input_layout_str, pse, drop_mask, padding_mask, atten_mask,
    softmax_max, softmax_sum, softmax_in, attention_in,
    scale_value, keep_prob, pre_tockens, next_tockens,
    inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode, softmax_layout);
```

After modifying: rebuild torch_npu, reinstall, restart training.

## Prerequisite — correct source tree

The installed binary is **2.1.0.post12+git56b0510**. None of the local source
trees match (all are 2.7/2.8-series). The 2.1-series source tree that built the
current install is not present on this machine. You'd need to:

1. Clone `pytorch/third_party/torch_npu` at the matching commit (`56b0510d`)
2. Or locate/build from the correct 2.1-series branch
3. Or upgrade the installed torch_npu to match a local source tree

## After C++ injection: Python cleanup

Once C++ injection is in place, these Python files become dead code and should be removed:

| File | What |
|---|---|
| `custom_dot_product_attention.py:35-67` | `_FaultInjectionFA` class |
| `custom_dot_product_attention.py:467` | `output = _FaultInjectionFA.apply(output)` |
| `custom_dot_product_attention.py:23` | `from ...fault_injection import bitshift, round2nearest` |
| `fault_injection.py` | entire file |
