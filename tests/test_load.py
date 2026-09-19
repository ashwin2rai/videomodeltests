import pytest

import h3

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


def test_mock_backend_load_is_noop():
    h3.MockH3Backend().load()


@pytest.mark.skipif(_CUDA_AVAILABLE, reason="requires a machine without CUDA")
def test_h3_backend_load_raises_without_cuda():
    with pytest.raises(RuntimeError, match="CUDA"):
        h3.H3Backend().load()


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a real CUDA GPU")
def test_h3_backend_load_succeeds_with_cuda():
    h3.H3Backend().load()
