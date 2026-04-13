"""Lazy per-expert dequantization for fused MoE weight tensors.

Instead of dequantizing all 256 expert weights to fp16 on GPU (~4.8 GB),
``LazyDequantExperts`` keeps the packed 4-bit data on GPU (~1.2 GB) and
dequantizes individual expert slices on-demand when indexed.
"""

from __future__ import annotations

import torch


class LazyDequantExperts:
    """Stores packed 4-bit fused expert weights on GPU and dequantizes
    individual expert slices on ``__getitem__``.

    Designed to be set as a plain attribute on the ``Qwen3_5MoeExperts``
    module, replacing the ``nn.Parameter``.  The HF eager forward loop
    accesses experts via ``self.gate_up_proj[expert_idx]`` with a scalar
    integer index, which triggers ``__getitem__`` and returns a 2-D fp16
    tensor ready for ``F.linear``.
    """

    def __init__(
        self,
        packed: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        quant_type: str,
        blocksize: int,
        source_dtype: torch.dtype,
        original_shape: list[int],
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda:0",
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.original_shape = original_shape
        self.num_experts = original_shape[0]
        self._expert_rows = original_shape[1]
        self._expert_cols = original_shape[2]
        self._expert_numel = self._expert_rows * self._expert_cols
        self._expert_packed_bytes = self._expert_numel // 2
        self._expert_blocks = self._expert_numel // blocksize

        self._packed = packed.to(self.device)
        self._absmax = absmax.to(self.device)
        self._code = code.to(self.device)
        self._quant_type = quant_type
        self._blocksize = blocksize
        self._source_dtype = source_dtype

    def __getitem__(self, expert_idx: int) -> torch.Tensor:
        """Dequantize a single expert's 2-D weight matrix on GPU."""
        from bitsandbytes.functional import QuantState, dequantize_4bit

        p_start = expert_idx * self._expert_packed_bytes
        p_end = p_start + self._expert_packed_bytes
        a_start = expert_idx * self._expert_blocks
        a_end = a_start + self._expert_blocks

        partial_qs = QuantState(
            absmax=self._absmax[a_start:a_end],
            shape=torch.Size([self._expert_rows, self._expert_cols]),
            blocksize=self._blocksize,
            code=self._code,
            quant_type=self._quant_type,
            dtype=self._source_dtype,
        )

        return dequantize_4bit(self._packed[p_start:p_end], partial_qs).to(self.dtype)

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self.original_shape)

    def to(self, device: torch.device | str) -> LazyDequantExperts:
        """Move packed data and quant state to *device*."""
        device = torch.device(device)
        if device != self.device:
            self._packed = self._packed.to(device)
            self._absmax = self._absmax.to(device)
            self._code = self._code.to(device)
            self.device = device
        return self

    def numel_packed(self) -> int:
        """Total bytes of packed data on GPU."""
        return int(self._packed.numel())

    @classmethod
    def from_quantized(
        cls,
        packed: torch.Tensor,
        qs_entries: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> LazyDequantExperts:
        """Build from the raw entries produced by ``parse_quantized_state_dict``.

        *qs_entries* is mutated (``original_shape`` is popped).
        """
        from bitsandbytes.functional import QuantState

        original_shape = qs_entries.pop("original_shape").tolist()

        # Reconstruct QuantState to extract code / quant_type / etc.
        qs = QuantState.from_dict(qs_entries, device=device)

        return cls(
            packed=packed,
            absmax=qs.absmax,
            code=qs.code,
            quant_type=qs.quant_type,
            blocksize=qs.blocksize,
            source_dtype=qs.dtype,
            original_shape=original_shape,
            dtype=dtype,
            device=device,
        )
