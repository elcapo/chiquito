import pytest
import torch

from chiquito.lazy_experts import LazyDequantExperts


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def expert_data(gpu):
    """Create a small fused expert tensor, quantize it, return both."""
    import bitsandbytes.functional as F

    num_experts, rows, cols = 4, 8, 64
    original = torch.randn(num_experts, rows, cols, dtype=torch.float16, device=gpu)
    flat = original.reshape(-1, cols)
    packed, qs = F.quantize_4bit(flat, quant_type="nf4", blocksize=64)

    return {
        "original": original,
        "packed": packed,
        "qs": qs,
        "shape": [num_experts, rows, cols],
    }


class TestLazyDequantExperts:
    def test_getitem_matches_full_dequant(self, gpu, expert_data):
        """Each wrapper[i] must match full_dequant[i*rows:(i+1)*rows]."""
        import bitsandbytes.functional as F

        full = F.dequantize_4bit(expert_data["packed"], expert_data["qs"])
        qs = expert_data["qs"]

        wrapper = LazyDequantExperts(
            packed=expert_data["packed"],
            absmax=qs.absmax,
            code=qs.code,
            quant_type=qs.quant_type,
            blocksize=qs.blocksize,
            source_dtype=qs.dtype,
            original_shape=expert_data["shape"],
            dtype=torch.float16,
            device=gpu,
        )

        rows = expert_data["shape"][1]
        for i in range(expert_data["shape"][0]):
            sliced = wrapper[i]
            expected = full[i * rows : (i + 1) * rows]
            assert sliced.shape == expected.shape
            assert torch.equal(sliced, expected), f"Expert {i} mismatch"

    def test_getitem_returns_correct_shape(self, gpu, expert_data):
        qs = expert_data["qs"]
        wrapper = LazyDequantExperts(
            packed=expert_data["packed"],
            absmax=qs.absmax,
            code=qs.code,
            quant_type=qs.quant_type,
            blocksize=qs.blocksize,
            source_dtype=qs.dtype,
            original_shape=expert_data["shape"],
            dtype=torch.float16,
            device=gpu,
        )
        result = wrapper[0]
        assert result.shape == (8, 64)
        assert result.dtype == torch.float16
        assert result.device.type == "cuda"

    def test_shape_property(self, gpu, expert_data):
        qs = expert_data["qs"]
        wrapper = LazyDequantExperts(
            packed=expert_data["packed"],
            absmax=qs.absmax,
            code=qs.code,
            quant_type=qs.quant_type,
            blocksize=qs.blocksize,
            source_dtype=qs.dtype,
            original_shape=expert_data["shape"],
            dtype=torch.float16,
            device=gpu,
        )
        assert wrapper.shape == torch.Size([4, 8, 64])

    def test_from_quantized(self, gpu, expert_data):
        """Test the classmethod factory used by composite_model."""
        import bitsandbytes.functional as F

        qs = expert_data["qs"]
        qs_entries = qs.as_dict(packed=True)
        qs_entries["original_shape"] = torch.tensor(expert_data["shape"])

        wrapper = LazyDequantExperts.from_quantized(
            packed=expert_data["packed"],
            qs_entries=qs_entries,
            dtype=torch.float16,
            device=gpu,
        )

        assert wrapper.shape == torch.Size([4, 8, 64])
        assert wrapper.num_experts == 4

        # Verify correctness
        full = F.dequantize_4bit(expert_data["packed"], qs)
        for i in range(4):
            rows = expert_data["shape"][1]
            assert torch.equal(wrapper[i], full[i * rows : (i + 1) * rows])

    def test_to_device(self, gpu, expert_data):
        qs = expert_data["qs"]
        wrapper = LazyDequantExperts(
            packed=expert_data["packed"],
            absmax=qs.absmax,
            code=qs.code,
            quant_type=qs.quant_type,
            blocksize=qs.blocksize,
            source_dtype=qs.dtype,
            original_shape=expert_data["shape"],
            dtype=torch.float16,
            device=gpu,
        )
        moved = wrapper.to("cpu")
        assert moved is wrapper
        assert wrapper._packed.device.type == "cpu"

        wrapper.to(gpu)
        assert wrapper._packed.device == gpu
