"""Device planning — fast, no model, no video."""
import pytest
import torch

from voxlens.devices import DevicePlan, available_devices, resolve_device

MPS = torch.backends.mps.is_available()


def test_cpu_plan_runs_everything_on_cpu():
    plan = resolve_device("cpu")
    assert plan.encoder.type == "cpu"
    assert plan.decoder.type == "cpu"


def test_cpu_plan_is_always_available():
    assert "cpu" in available_devices()


def test_unknown_device_is_rejected_by_name():
    with pytest.raises(ValueError, match="wibble"):
        resolve_device("wibble")


def test_plan_is_immutable():
    plan = resolve_device("cpu")
    with pytest.raises(Exception):
        plan.encoder = torch.device("meta")


def test_plan_reports_the_name_it_was_asked_for():
    assert resolve_device("cpu").name == "cpu"


@pytest.mark.skipif(not MPS, reason="requires an Apple Silicon GPU")
def test_hybrid_puts_the_encoder_on_gpu_and_the_search_on_cpu():
    """The measured-best split: dense convolution on GPU, beam search on CPU."""
    plan = resolve_device("hybrid")
    assert plan.encoder.type == "mps"
    assert plan.decoder.type == "cpu"


@pytest.mark.skipif(not MPS, reason="requires an Apple Silicon GPU")
def test_mps_plan_runs_everything_on_gpu():
    plan = resolve_device("mps")
    assert plan.encoder.type == "mps"
    assert plan.decoder.type == "mps"


@pytest.mark.skipif(MPS, reason="only meaningful without a GPU")
def test_hybrid_without_a_gpu_fails_with_an_actionable_message():
    with pytest.raises(RuntimeError, match="--device cpu"):
        resolve_device("hybrid")
