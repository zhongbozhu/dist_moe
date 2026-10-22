# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit distributed-MoE execution without autograd or parameter ownership."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from . import api
from ._context import _get_context
from ._execution import (
    _KernelAutogradContext,
    _validate_external_wgrad_destinations,
    ExecutionOptions,
    PreparedWeight,
)
from ._postprocess import _registered_rmsnorm_args

__all__ = ["ForwardState", "forward", "backward"]


class ForwardState(_KernelAutogradContext):
    """Opaque, single-consumption state returned by :func:`forward`.

    The state retains borrowed input/weight tensors and its execution context.
    Callers must not mutate or rebind these operands before backward, close the
    context, or overwrite a live activation slot. Multiple forwards in one slot
    require its configured depth and reverse-order backwards. The package does
    not retain parameter owners, run hooks, or manage optimizer lifetimes.

    CUDA graphs capture forward/backward once and replay their device work.
    Captured tensor storage, operands, context, and gradient destinations must
    remain alive at stable addresses until the graph is destroyed. Graph pools
    and the context may retain that storage; the Python handle itself can be
    released after backward capture. The caller manages per-replay ownership
    and completion; Python state is not recreated on replay.
    """

    def __init__(self) -> None:
        """Reject construction outside :func:`forward`."""
        raise TypeError("ForwardState is created by functional.forward")

    @property
    def metadata(self) -> Mapping[str, torch.Tensor]:
        """Return borrowed read-only device planner metadata without synchronizing.

        ``need_recompute`` is a scalar bool and ``padded_receive_rows`` contains
        one int32 count per EP rank. References remain inspectable after state
        consumption and refer to captured buffers when retained from capture.
        Callers must not write these tensors; reading values synchronizes CUDA.
        """
        return self._metadata


@torch.no_grad()
def forward(
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    w13_weight: torch.Tensor | PreparedWeight,
    w2_weight: torch.Tensor | PreparedWeight,
    context: api.Context,
    *,
    options: ExecutionOptions | None = None,
    save_for_backward: bool = True,
) -> tuple[torch.Tensor, ForwardState | None]:
    """Run explicit forward and return output plus opaque backward state.

    Args:
        x_TD: Contiguous CUDA BF16 tokens ``[T, D]``.
        topk_expert_ids_TK: Contiguous int32/int64 routes ``[T, K]``.
        topk_scores_TK: Contiguous BF16/FP32 route scores ``[T, K]``.
        w13_weight: BF16 grouped gate/up weights or prepared MXFP8 weights.
        w2_weight: Matching grouped down weights or prepared MXFP8 weights.
        context: Live context with matching shapes. Input/weight layouts and
            routing constraints are those of :func:`dist_moe.routed_experts`.
        options: Registered execution controls; Python callbacks and automatic
            parameter-gradient accumulation are unsupported here.
        save_for_backward: Explicitly retain a training frame regardless of
            ambient grad mode. False returns no state and leaves activation
            slots unchanged; it does not change ``context.config.inference``.

    Returns:
        Fresh detached ``[T, D]`` output and opaque state, or None when retention
        is disabled. No autograd edges are created even for trainable inputs.

    Note:
        Collective over the context group; every rank must call in the same
        order with compatible shapes/options. Context buffers are mutated.
        State/operand lifetimes follow :class:`ForwardState`. CUDA capture is
        supported; tracing this Python state interface and higher-order
        derivatives are unsupported. Use ``routed_experts`` for native autograd.

    Raises:
        ValueError: If retention, options, or weight formats are unsupported.
    """
    if not isinstance(save_for_backward, bool):
        raise TypeError("save_for_backward must be bool")
    options, w13_EFD, w2_EDF = api._prepare_execution_inputs(
        x_TD,
        topk_expert_ids_TK,
        topk_scores_TK,
        w13_weight,
        w2_weight,
        context,
        options,
    )
    if options.requires_eager or options.inplace_wgrad_accum:
        raise ValueError(
            "functional execution does not support callbacks or parameter gradients"
        )
    if not save_for_backward:
        return api.routed_experts(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            w13_weight,
            w2_weight,
            context,
            options=options,
        ), None
    if context.config.inference:
        raise ValueError("functional training requires a training context")
    activation = context.activation_buffer
    assert activation is not None
    rmsnorm_args = _registered_rmsnorm_args(options.experts_output_postprocess)
    common = (x_TD, topk_expert_ids_TK, topk_scores_TK)
    tail = (
        activation.activation_slot_id_1,
        activation._num_moe_layers_in_selected_slot,
    )
    state = object.__new__(ForwardState)
    _KernelAutogradContext.__init__(state)
    state._context = context
    state._consumed = False
    state._output_shape = x_TD.shape
    state._block_scaled = context.config.block_scaled is not None
    if state._block_scaled:
        from . import _blockscaled

        if context.config.block_scaled.format is not api.BlockScaledFormat.MXFP8_E4M3:
            raise ValueError("functional training supports BF16 or MXFP8")
        prepared = [*_prepared_operands(w13_weight), *_prepared_operands(w2_weight)]
        compute = (
            w13_EFD if prepared[0] is None else prepared[0],
            w2_EDF if prepared[6] is None else prepared[6],
        )
        args = (
            *common,
            *compute,
            prepared,
            *rmsnorm_args,
            *tail,
            True,
            options.validate_expert_ids,
            context.context_id,
            False,
        )
        output = _blockscaled._block_scaled_forward_op(*args)
        _blockscaled._block_scaled_setup_context(
            state,
            args,
            output,
            mark_state_non_differentiable=False,
        )
    else:
        args = (
            *common,
            w13_EFD,
            w2_EDF,
            *rmsnorm_args,
            *tail,
            options.validate_expert_ids,
            context.context_id,
        )
        if options.swiglu_clip_stats_out_3 is None:
            output = api._bf16_forward_op(*args)
        else:
            observed = api._bf16_forward_with_clip_stats_op(
                *args[:-1],
                options.swiglu_clip_limit,
                context.context_id,
            )
            options.swiglu_clip_stats_out_3.copy_(observed[2])
            output = observed[:2]
        api._bf16_setup_context(
            state, args, output, mark_state_non_differentiable=False
        )
    state._borrowed = (x_TD, *state.saved_tensors)
    routing_offset = 14 if state._block_scaled else 4
    state._metadata = MappingProxyType(
        {
            "need_recompute": state.saved_tensors[routing_offset + 6],
            "padded_receive_rows": state.saved_tensors[routing_offset + 4],
        }
    )
    return output[0], state


def _prepared_operands(weight: torch.Tensor | PreparedWeight) -> tuple:
    """Flatten optional FPROP/DGRAD operands without reading logical sources."""
    if not isinstance(weight, PreparedWeight):
        return (None,) * 6
    if weight._interleaved_fc13:
        raise ValueError("functional training requires concatenated gate/up weights")
    return (
        weight.fprop_data,
        weight.fprop_scale,
        weight.dgrad_data,
        weight.dgrad_scale,
        weight.global_scale,
        weight.global_scale_inv,
    )


@torch.no_grad()
def backward(
    state: ForwardState,
    grad_output_TD: torch.Tensor,
    *,
    grad_w13: torch.Tensor,
    grad_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume forward state and accumulate directly into external gradients.

    Args:
        state: Unconsumed state from a matching explicit forward.
        grad_output_TD: Contiguous CUDA BF16 output gradient ``[T, D]``.
        grad_w13: Non-differentiable contiguous grouped gradient ``[E, 2F, D]``.
        grad_w2: Matching grouped gradient ``[E, D, F]``. Both destinations use
            one BF16/FP32 dtype, matching ``Config.wgrad_dtype`` when specified,
            and cannot overlap each other or borrowed execution operands.

    Returns:
        Detached input and routing-score gradients with the original shapes
        and dtypes. Weight gradients are added in place; they are not returned
        or copied through full temporary weight-gradient tensors.

    Note:
        Collective participation/order, context mutation and captured storage
        lifetimes match :func:`forward`. The caller owns zeroing, serialization,
        optimizer/DDP notification and synchronization. State is consumed once
        before launching backward; an execution failure must not be retried.
        CUDA graph replay reuses captured device work without Python callbacks.

    Raises:
        RuntimeError: If state was consumed, context closed, or tensor metadata
            and destination ownership requirements are violated before launch.
    """
    if not isinstance(state, ForwardState):
        raise TypeError("state must come from functional.forward")
    if state._consumed:
        raise RuntimeError("functional backward consumes each state once")
    _get_context(state.context_id)
    w13_EFD, w2_EDF, ids_TK, scores_TK, *saved = state.saved_tensors
    if (
        grad_output_TD.shape != state._output_shape
        or grad_output_TD.dtype is not torch.bfloat16
        or grad_output_TD.device != w13_EFD.device
        or not grad_output_TD.is_contiguous()
    ):
        raise RuntimeError("output gradient must match contiguous CUDA BF16 output")
    dtype = state._context.config.wgrad_dtype or grad_w13.dtype
    _validate_external_wgrad_destinations(
        (grad_w13, grad_w2),
        (w13_EFD, w2_EDF),
        dtype,
        (*state._borrowed, grad_output_TD),
    )
    if state._block_scaled:
        from ._blockscaled import _block_scaled_backward_accumulate_op as execute
    else:
        execute = api._bf16_backward_accumulate_op
    state._consumed = True
    return execute(
        grad_output_TD,
        w13_EFD,
        w2_EDF,
        grad_w13,
        grad_w2,
        ids_TK,
        scores_TK,
        saved,
        state.rmsnorm_enabled,
        state.rmsnorm_eps,
        state.rmsnorm_norm_output_dtype,
        state.rmsnorm_output_dtype,
        state.rmsnorm_require_bitwise,
        state.rmsnorm_use_kahan,
        state.rmsnorm_recompute_rstd,
        dtype,
        state.context_id,
    )
