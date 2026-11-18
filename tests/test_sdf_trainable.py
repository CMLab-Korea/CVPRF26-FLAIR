import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from modules import bla, bla_float, speed


ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_PARAMETERS = ("T", "zeta", "sigma")
COORDS = torch.linspace(-0.7, 0.7, 18).reshape(2, 3, 3)


def load_sdf(patch):
    # The release references BACON helpers outside this checkout. The test
    # only exercises the parser and model, before data or training is entered.
    for name in ("dataio", "utils", "training", "loss_functions"):
        patch.setitem(sys.modules, name, ModuleType(name))
    tensorboard = ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = None
    patch.setitem(sys.modules, tensorboard.__name__, tensorboard)
    patch.setattr(sys, "argv", ["train_sdf.py", "--experiment_name", "trainability-test"])
    patch.setattr(sys, "path", list(sys.path))
    patch.setattr(torch, "set_num_threads", lambda count: None)
    spec = importlib.util.spec_from_file_location("flair_train_sdf_test", ROOT / "tasks/sdf/train_sdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sdf():
    with pytest.MonkeyPatch.context() as patch:
        yield load_sdf(patch)


def patch_cpu(patch):
    patch.setattr(torch.nn.Module, "cuda", lambda model: model)
    patch.setattr(speed, "setup_fast_env", lambda: None)
    patch.setattr(speed, "maybe_compile", lambda model, enabled: model)


@pytest.fixture
def cpu(monkeypatch):
    patch_cpu(monkeypatch)


def options(sdf, fast=False, layers=3):
    args = ["--experiment_name", "trainability-test", "--hidden_size", "4", "--hidden_layers", str(layers)]
    if fast:
        args.append("--fast")
    return sdf.p.parse_args(args)


def activations(inr, fast):
    blocks = list(inr.net)[:-1]
    return [block.act for block in blocks] if fast else blocks


def assert_trainability(inr, fast, layers, trainable):
    expected = bla_float if fast else bla
    assert isinstance(inr, expected.INR)
    blocks = list(inr.net)[:-1]
    assert len(blocks) == layers
    for activation in activations(inr, fast):
        for name in ACTIVATION_PARAMETERS:
            assert getattr(activation, name).requires_grad is trainable
        assert not activation.beta.requires_grad
    for block in blocks:
        assert block.linear.weight.requires_grad
    assert inr.net[-1].weight.requires_grad


@pytest.mark.parametrize(
    "extra,expected",
    [([], True), (["--trainable", "True"], True), (["--trainable", "False"], False),
     (["--trainable", "YES"], True), (["--trainable", "0"], False)],
    ids=["default", "true", "false", "yes", "zero"],
)
def test_cli_boolean(sdf, extra, expected):
    assert sdf.p.parse_args(["--experiment_name", "fixture", *extra]).trainable is expected


@pytest.mark.parametrize("value,expected", [("True", True), ("False", False), ("n", False)])
def test_config_boolean(sdf, tmp_path, value, expected):
    config = tmp_path / "config.ini"
    config.write_text(f"[DEFAULT]\nexperiment_name = fixture\ntrainable = {value}\n", encoding="ascii")
    assert sdf.p.parse_args(["--config", str(config)]).trainable is expected


def test_shipped_default_config_is_trainable(sdf):
    config = ROOT / "tasks/sdf/configs/bla_armadillo.ini"
    assert sdf.p.parse_args(["--config", str(config)]).trainable is True


@pytest.mark.parametrize("use_config", [False, True], ids=["cli", "config"])
def test_invalid_boolean_has_clear_error(sdf, tmp_path, capsys, use_config):
    args = ["--experiment_name", "fixture", "--trainable", "maybe"]
    if use_config:
        config = tmp_path / "config.ini"
        config.write_text("experiment_name = fixture\ntrainable = maybe\n", encoding="ascii")
        args = ["--config", str(config)]
    with pytest.raises(SystemExit) as failure:
        sdf.p.parse_args(args)
    assert failure.value.code == 2
    stderr = capsys.readouterr().err
    assert "--trainable" in stderr and "boolean" in stderr and "maybe" in stderr


@pytest.mark.parametrize("fast", [False, True], ids=["standard", "fast"])
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("trainable", [True, False], ids=["train", "freeze"])
def test_inr_forwards_trainability_to_every_block(fast, layers, trainable):
    inr = (bla_float if fast else bla).INR(3, 4, layers, 1, trainable=trainable)
    assert_trainability(inr, fast, layers, trainable)


@pytest.mark.parametrize("fast", [False, True], ids=["standard", "fast"])
@pytest.mark.parametrize("trainable", [True, False], ids=["train", "freeze"])
def test_sdf_adapter_forwards_trainability(sdf, fast, trainable):
    model = sdf.BLAforSDF(4, 3, 2.0, 1.0, 0.05, 0.1, use_fast=fast, trainable=trainable)
    assert_trainability(model.inr, fast, 3, trainable)


@pytest.mark.parametrize("fast", [False, True], ids=["standard", "fast"])
@pytest.mark.parametrize("trainable", [True, False], ids=["train", "freeze"])
def test_sdf_factory_forwards_python_bool(sdf, cpu, fast, trainable):
    opt = options(sdf, fast)
    opt.trainable = trainable  # Separate propagation from CLI parsing.
    assert_trainability(sdf.init_model(opt).inr, fast, 3, trainable)


@pytest.mark.parametrize("fast", [False, True], ids=["standard", "fast"])
@pytest.mark.parametrize("trainable", [True, False], ids=["train", "freeze"])
def test_forward_backward_and_optimizer(sdf, cpu, fast, trainable):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        opt = options(sdf, fast)
        opt.trainable = trainable
        model = sdf.init_model(opt)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    selected = {id(p) for group in optimizer.param_groups for p in group["params"]}
    original = {name: param.detach().clone() for name, param in model.named_parameters()}
    input_dict = {"coords": COORDS.clone()}
    output = model(input_dict)
    assert output["model_in"] is input_dict
    assert output["model_out"].shape == (2, 3, 1)
    assert torch.isfinite(output["model_out"]).all()
    output["model_out"].square().mean().backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        active = trainable or name.split(".")[-1] not in ACTIVATION_PARAMETERS
        assert (parameter.grad is not None) is active
        assert (id(parameter) in selected) is active
        if active:
            assert torch.isfinite(parameter.grad).all()
        else:
            torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
    assert not torch.equal(model.inr.net[-1].weight, original["inr.net.3.weight"])


@pytest.mark.parametrize("fast", [False, True], ids=["standard", "fast"])
def test_checkpoint_restores_state_without_overriding_freeze(sdf, cpu, tmp_path, fast):
    trained = sdf.init_model(options(sdf, fast))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save(trained.state_dict(), checkpoints / "model_step_0002.pth")
    opt = options(sdf, fast)
    opt.root_path = str(tmp_path)
    opt.num_steps = 5
    opt.ckpt_step = 2
    opt.trainable = False

    restored = sdf.init_model(opt)
    assert opt.num_steps == 3
    assert restored.state_dict().keys() == trained.state_dict().keys()
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, trained.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(
            restored({"coords": COORDS})["model_out"], trained({"coords": COORDS})["model_out"], rtol=0, atol=0
        )
    assert_trainability(restored.inr, fast, 3, False)
