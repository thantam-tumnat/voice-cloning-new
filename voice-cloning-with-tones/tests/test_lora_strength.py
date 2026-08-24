"""The Thai LoRA is applied per side, and the DiT side is what flattens emotion.

set_lora_strength has to reach the real LoRALinear layers -- if it silently matched
nothing, synthesis would keep running at the shipped strength and the only symptom
would be flat audio, which is exactly the bug it exists to fix. These fakes stand in
for the VoxCPM module tree so that wiring is checked without an 8 GB model.
"""
import sys
import types

import pytest

from app.services.siangtts_service import set_lora_strength


class FakeScaling:
    """Stands in for the torch buffer LoRALinear keeps its strength in."""

    def __init__(self, value):
        self.value = value

    def fill_(self, v):
        self.value = v


class FakeLoRALinear:
    def __init__(self, value=2.0):
        self.scaling = FakeScaling(value)

    def modules(self):
        return [self]


class FakeBranch:
    def __init__(self, n):
        self.layers = [FakeLoRALinear() for _ in range(n)]

    def modules(self):
        return [self] + self.layers


class FakeModel:
    def __init__(self, n_lm=4, n_residual=2, n_dit=3):
        self.base_lm = FakeBranch(n_lm)
        self.residual_lm = FakeBranch(n_residual)
        self.feat_decoder = types.SimpleNamespace(estimator=FakeBranch(n_dit))

    @property
    def lm_layers(self):
        return self.base_lm.layers + self.residual_lm.layers

    @property
    def dit_layers(self):
        return self.feat_decoder.estimator.layers


@pytest.fixture
def fake_lora_module(monkeypatch):
    """Install a voxcpm.modules.layers.lora whose LoRALinear is the fake above."""
    module = types.ModuleType("voxcpm.modules.layers.lora")
    module.LoRALinear = FakeLoRALinear
    for name in ("voxcpm", "voxcpm.modules", "voxcpm.modules.layers"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "voxcpm.modules.layers.lora", module)
    return module


def test_set_lora_strength_scales_each_side_independently(fake_lora_module):
    model = FakeModel()

    applied = set_lora_strength(model, lm=2.0, dit=0.0)

    assert [l.scaling.value for l in model.lm_layers] == [2.0] * 6
    assert [l.scaling.value for l in model.dit_layers] == [0.0] * 3
    assert applied["lm_layers"] == 6
    assert applied["dit_layers"] == 3


def test_set_lora_strength_can_restore_the_shipped_setting(fake_lora_module):
    model = FakeModel()

    set_lora_strength(model, lm=2.0, dit=0.0)
    set_lora_strength(model, lm=2.0, dit=2.0)

    assert all(l.scaling.value == 2.0 for l in model.lm_layers + model.dit_layers)


def test_set_lora_strength_survives_a_voxcpm_without_lora(monkeypatch):
    """No LoRA layers is a degraded model, not a crashed request."""
    monkeypatch.setitem(sys.modules, "voxcpm.modules.layers.lora", None)

    applied = set_lora_strength(FakeModel(), lm=2.0, dit=0.0)

    assert applied == {"lm": 0, "dit": 0}
