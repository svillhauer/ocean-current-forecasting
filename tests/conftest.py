import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
