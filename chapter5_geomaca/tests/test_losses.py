"""The credit-module objective: loss forms, weights and the agent-entropy switch."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from chapter5_geomaca.geomaca.losses import GeoMACALossComputer, GeoMACALosses
from chapter5_geomaca.geomaca.model import GeoMACA, GeoMACAConfig

K, Z, A = 9, 9, 3


def _credit(seed: int = 0):
    """A credit tensor of the form alpha_temp x alpha_geo x alpha_agent, normalised."""
    g = torch.Generator().manual_seed(seed)
    t = torch.softmax(torch.randn(K, generator=g), 0)
    z = torch.softmax(torch.randn(Z, generator=g), 0)
    a = torch.softmax(torch.randn(A, generator=g), 0)
    c = torch.einsum("t,z,a->tza", t, z, a).unsqueeze(0)
    return c / c.sum(), t, z, a


TRAJ = {"zone_developed": torch.arange(K).unsqueeze(0)}


def test_default_total_adds_the_agent_term():
    c, *_ = _credit()
    out = GeoMACALossComputer(device="cpu").compute(c, TRAJ)
    expected = (torch.tensor(0.0) + 0.1 * out["entropy"] + 0.5 * 0.1 * out["agent_entropy"]
                + 0.05 * out["consistency"] + 0.02 * out["causality"])
    assert torch.equal(out["total"], expected)


def test_chapter5_total_has_only_the_three_terms():
    c, *_ = _credit(1)
    out = GeoMACALossComputer(device="cpu", agent_entropy_coef=0.0).compute(c, TRAJ)
    expected = (torch.tensor(0.0) + 0.1 * out["entropy"] + 0.05 * out["consistency"]
                + 0.02 * out["causality"])
    assert torch.equal(out["total"], expected)
    assert "agent_entropy" in out  # still reported, not added


def test_loss_forms():
    """L_entropy = (H(c)/log(K Z A) - 0.5)^2; hinge consistency with m = 0.1;
    causality = symmetric KL between alpha_temp and nu_k proportional to 0.9^(K-k)."""
    c, t, z, a = _credit(2)
    cn = c[0].numpy().astype(np.float64)
    h = -(cn * np.log(cn)).sum()
    ent = (h / math.log(K * Z * A) - 0.5) ** 2
    ck = cn.sum(axis=2)  # c_k(z)
    hinge = np.mean([max(0.0, 0.1 + (ck[k].sum() - ck[k, k]) / (Z - 1) - ck[k, k]) for k in range(K)])
    nu = 0.9 ** np.arange(K - 1, -1, -1, dtype=np.float64)
    nu = nu / nu.sum()
    at = t.numpy().astype(np.float64)
    caus = 0.5 * ((nu * np.log(nu / at)).sum() + (at * np.log(at / nu)).sum())
    assert float(GeoMACALosses.entropy_loss(c)) == pytest.approx(ent, abs=1e-6)
    assert float(GeoMACALosses.consistency_loss(c, TRAJ)) == pytest.approx(hinge, abs=1e-6)
    assert float(GeoMACALosses.temporal_causality_loss(c)) == pytest.approx(caus, abs=1e-5)
    assert nu[0] == pytest.approx(0.070, abs=5e-4) and nu[-1] == pytest.approx(0.163, abs=5e-4)


def _agent_head_grads(total: torch.Tensor, model: GeoMACA):
    params = [p for n, p in model.named_parameters() if n.startswith(("agent_attention", "agent_encoder"))]
    return torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)


def test_only_the_entropy_term_reaches_the_agent_head():
    """With the Chapter 5 objective, consistency and causality do not move the agent head."""
    torch.manual_seed(0)
    model = GeoMACA(GeoMACAConfig(state_dim=20, action_dim=5, num_zones=9, num_turns=9,
                                  num_agents=3, attention_normalization="softmax"))
    model.eval()
    g = torch.Generator().manual_seed(4)
    traj = {"zone_states": torch.randn(1, K, Z, 20, generator=g),
            "agent_actions": torch.rand(1, K, A, 5, generator=g),
            "global_state": torch.randn(1, K, 20, generator=g),
            "zone_developed": torch.arange(K).unsqueeze(0)}
    credit, _ = model(traj, torch.tensor([-0.1]))
    parts = GeoMACALossComputer(device="cpu", agent_entropy_coef=0.0).compute(credit, traj)
    for name in ("consistency", "causality"):
        for grad in _agent_head_grads(parts[name], model):
            if grad is not None:
                assert grad.abs().max() < 1e-7, name
    assert any(gr is not None and gr.abs().max() > 1e-6
               for gr in _agent_head_grads(parts["entropy"], model))
    # the default objective adds a second term that reaches the agent head
    default = GeoMACALossComputer(device="cpu").compute(credit, traj)
    diff = [x - y for x, y in zip(_agent_head_grads(default["total"], model),
                                  _agent_head_grads(parts["total"], model)) if x is not None]
    assert max(d.abs().max() for d in diff) > 0
