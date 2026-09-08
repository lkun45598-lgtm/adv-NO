import importlib.util
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "3_flow_reconstruction" / "no" / "adv_training"
MODULE_PATH = TRAINING_DIR / "train_sparse_adv_no.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_sparse_adv_no", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_threads_default_and_runtime_configuration(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING_DIR))
    module = load_training_module()
    args = module.build_arg_parser().parse_args(
        ["--data", "input.h5", "--output-dir", "output"]
    )
    original_threads = torch.get_num_threads()
    try:
        assert args.cpu_threads == 1
        module.configure_cpu_threads(args.cpu_threads)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original_threads)


def test_cpu_threads_must_be_positive(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING_DIR))
    module = load_training_module()
    args = module.build_arg_parser().parse_args(
        ["--data", "input.h5", "--output-dir", "output", "--cpu-threads", "0"]
    )
    with pytest.raises(ValueError, match="--cpu-threads must be positive"):
        module.validate_args(args)
