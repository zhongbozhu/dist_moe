# Explicit execution

`dist_moe.functional` provides a framework-neutral forward/backward interface.
It uses the same context, planner, and registered kernels as
[`routed_experts`](../README.md#public-api), while leaving autograd edges,
parameter hooks, gradient reduction, and optimizer policy to its caller.
Native `routed_experts` autograd behavior is unchanged.

```python
from dist_moe import functional

y_TD, state = functional.forward(
    x_TD, expert_ids_TK, scores_TK, w13_weight, w2_weight, context,
)
dx_TD, dscores_TK = functional.backward(
    state, dy_TD, grad_w13=dw13_EFD, grad_w2=dw2_EDF,
)
```

Forward accepts contiguous CUDA BF16 tokens `[T, D]`, int32/int64 routes
`[T, K]`, BF16/FP32 scores `[T, K]`, and grouped BF16 weights
`[E, 2F, D]` / `[E, D, F]`, where `E` is the local expert count. It also
accepts paired MXFP8 prepared weights, including independently imported FPROP
and DGRAD operands described in [MXFP8 execution](mxfp8_execution.md).
The usual context shapes, routing constraints, and collective participation
requirements apply: all ranks must issue compatible operations in the same
order. Python callbacks, automatic parameter-gradient accumulation, and
NVFP4 training are unsupported by this explicit interface.

Forward always returns a detached output with no automatic autograd edges,
even when called with trainable inputs outside `torch.no_grad()`. Its default
`save_for_backward=True` explicitly retains a training frame regardless of
ambient grad mode. The context must support training. With
`save_for_backward=False`, forward returns `(y_TD, None)` without changing
activation slots; this does not change the context's inference configuration.

Backward accepts a contiguous BF16 output gradient and two contiguous,
non-differentiable CUDA destinations with the logical grouped weight shapes.
Both destinations must use the same BF16 or FP32 dtype and match
`Config.wgrad_dtype` when it is set. Destinations must not overlap each other
or saved operands. Disjoint views into one larger gradient buffer are allowed.
Both are validated before either is written. WGRAD adds directly into these
buffers, including the first contribution, without full temporary weight
gradients. The caller zeroes them at the appropriate accumulation boundary
and serializes writers. Backward returns detached input and score gradients.

Each opaque state supports one backward. Retain input, weight, scale, and
context storage without mutation or rebinding until it completes. Multiple
live forwards must fit the configured activation-slot depth and unwind in
reverse order within a slot. A rejected argument leaves the state available
for correction; once kernel execution begins, a failed invocation must not be
retried. Higher-order derivatives and tracing this Python state interface are
unsupported; use native `routed_experts` for its supported tracing boundary.

CUDA capture uses the same explicit calls. Replay executes captured device
work and does not recreate Python state or invoke Python completion hooks.
Captured operands, context storage, and gradient destinations must retain
stable addresses until graph destruction. Graph pools and the context may
retain storage, so the Python state handle itself can be released after
backward capture. The caller owns replay sequencing, weight refresh, gradient
notification, synchronization, and graph destruction before context closure.

`state.metadata` is a read-only mapping containing borrowed device tensors:

| Key | Meaning |
| --- | --- |
| `need_recompute` | Scalar bool selecting activation recomputation |
| `padded_receive_rows` | One int32 padded receive-row count per EP rank |

The mapping remains inspectable after backward. A retained captured state
observes its captured buffers after replay. Do not mutate these tensors or
assume metadata from a later eager forward describes an earlier graph.
Reading their values synchronizes CUDA; keep observation outside timing.
