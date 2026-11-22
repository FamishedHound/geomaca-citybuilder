"""citybuilder/baseline_governments.py, and that the package needs nothing outside itself."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chapter5_geomaca.citybuilder import baseline_governments as bg
from chapter5_geomaca.evaluation.no_government import no_government_constraints

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Top-level packages of the research code base that the release must not import.
OUTSIDE_PACKAGES = {"src", "training", "geomaca", "experiments", "models", "paper_experiments"}


def test_three_agent_option_raises_a_clear_error():
    with pytest.raises(NotImplementedError, match="not part of the chapter5_geomaca package") as exc:
        bg.create_government("three_agent")
    assert "MAPPOGovernment" in str(exc.value)
    with pytest.raises(NotImplementedError):
        bg.create_government()  # 'three_agent' is the default model type


def test_three_agent_option_imports_nothing():
    before = set(sys.modules)
    with pytest.raises(NotImplementedError):
        bg.create_government("three_agent")
    assert not [m for m in set(sys.modules) - before if m.split(".")[0] in OUTSIDE_PACKAGES]


@pytest.mark.parametrize("model_type", bg.AVAILABLE_MODEL_TYPES)
def test_available_government_types(model_type):
    gov = bg.create_government(model_type, "balanced", device="cpu")
    constraints, info = gov.select_action(np.random.default_rng(0).standard_normal(20))
    assert isinstance(info, dict)
    assert 0.0 <= constraints.affordable_housing_pct <= 1.0
    assert 0.0 <= constraints.max_density <= 1.0


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError):
        bg.create_government("mystery")


def test_no_government_constraints_of_table_5_5():
    c = bg.create_government("no_gov").constraints
    assert (c.affordable_housing_pct, c.green_space_pct, c.woodland_min_cluster,
            c.max_density, c.tax_rate) == (0.0, 0.0, 0, 1.0, 0.0)
    g = no_government_constraints()
    assert (g.affordable_housing_pct, g.green_space_pct, g.woodland_preservation,
            g.max_density, g.tax_rate) == (0.0, 0.0, 0.0, 1.0, 0.0)


def test_module_self_test_runs():
    """``python -m chapter5_geomaca.citybuilder.baseline_governments`` (its own check) works."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "-m", "chapter5_geomaca.citybuilder.baseline_governments"],
                         cwd=PACKAGE_ROOT.parent, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "not part of the chapter5_geomaca package" in res.stdout
    assert "Baseline tests complete!" in res.stdout


def _absolute_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def test_no_module_imports_outside_the_package():
    """Every import is relative, of chapter5_geomaca, or of an installed library."""
    found = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for lineno, name in _absolute_imports(path):
            if name.split(".")[0] in OUTSIDE_PACKAGES:
                found.append(f"{path.relative_to(PACKAGE_ROOT)}:{lineno} imports {name}")
    assert not found, found
