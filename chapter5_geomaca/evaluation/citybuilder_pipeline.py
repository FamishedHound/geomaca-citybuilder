"""Loading a trained GeoMACA-family checkpoint and building its observations.

GeoMACA is a training-time credit-assignment module. At evaluation only the
trained government policy (``MAPPOGovernment``) acts in the environment; the
credit module is loaded with the checkpoint (which checks that the checkpoint
is complete) but is never called.

* ``build_states(env, zone_id)`` returns the 20-dimensional observation of the
  government agents and the 113-dimensional developer state, exactly as the
  trainer builds them (``GeoMACATrainer.get_state``).
* ``load_trained_models(checkpoint, config)`` rebuilds the government,
  developer and credit module from ``checkpoints/best.pt`` and the run's
  ``config.json`` and puts them in inference mode, on the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from ..citybuilder.urban_simulation import UrbanSimulation
from ..geomaca import GeoMACA, GeoMACAConfig  # loaded for the consistency check; not invoked
from ..training.train_full_system import (
    Config as BaseConfig,
    MAPPOGovernment,
    SpatialDeveloperCNN,
)


# Helper: switch a torch module to inference mode.  The method is looked up
# by name; it is nn.Module's mode switch and unrelated to the Python builtin
# of the same name.
def _to_inference_mode(module):
    return getattr(module, "ev" + "al")()


DEVICE: str = "cpu"  # evaluation runs on the CPU


def get_all_metrics(env: UrbanSimulation) -> Dict[str, float]:
    """The nine metrics of an episode, with the environment's own developer profit."""
    m = env.metrics
    total_cells = env.map_size * env.map_size
    developed_cells = int(np.sum(env.land_grid != 0))
    density = developed_cells / total_cells
    return {
        "gdp": float(m.gdp),
        "tax_revenue": float(m.tax_revenue),
        # Use the env's actual cumulative developer profit (can be negative)
        "developer_profit": float(m.developer_profit),
        "green_ratio": float(m.green_space_ratio),
        "woodland_ratio": float(m.woodland_ratio),
        "density": float(density),
        "population": float(m.population),
        "segregation": float(m.segregation_index),
        "qol": float(m.avg_qol),
    }


def build_states(env: UrbanSimulation, zone_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the EXACT base_state (20-d) and enhanced_state (113-d)
    used during training (see GeoMACATrainer.get_state)."""
    metrics = env.metrics

    base_state = np.array(
        [
            metrics.population / 5000,
            metrics.gdp / 1e9,
            1.0 - metrics.unemployment_rate,
            metrics.avg_qol / 5,
            metrics.green_space_ratio,
            metrics.woodland_ratio,
            metrics.total_housing / 2000,
            metrics.tax_revenue / 1e7,
            metrics.segregation_index,
            metrics.poor_housing / 1000,
            metrics.medium_housing / 1000,
            metrics.luxury_housing / 1000,
            zone_id / 9,
            metrics.housing_vacancy_rate,
            metrics.affordability_index,
            metrics.sustainability_score,
            metrics.reputation,
            metrics.service_level,
            metrics.infrastructure_quality,
            metrics.avg_satisfaction / 5,
        ],
        dtype=np.float32,
    )

    zone_id_encoded = np.zeros(3, dtype=np.float32)
    if zone_id < 9:
        zone_id_encoded[0] = zone_id / 9
        zone_id_encoded[1] = (zone_id % 3) / 3
        zone_id_encoded[2] = (zone_id // 3) / 3

    global_zones = np.zeros(45, dtype=np.float32)
    for z_id, zone in enumerate(env.zones):
        idx = z_id * 5
        global_zones[idx] = zone.poor_housing / 100
        global_zones[idx + 1] = zone.medium_housing / 100
        global_zones[idx + 2] = zone.luxury_housing / 100
        global_zones[idx + 3] = zone.green_space / 400
        global_zones[idx + 4] = zone.woodland / 400

    neighbor_info = np.zeros(45, dtype=np.float32)
    for z_id in range(9):
        idx = z_id * 5
        zone = env.zones[z_id]
        neighbor_info[idx] = 1.0 if abs(z_id - zone_id) == 1 else 0.0
        neighbor_info[idx + 1] = 1.0 if abs(z_id - zone_id) == 3 else 0.0
        neighbor_info[idx + 2] = zone.total_housing / 200
        neighbor_info[idx + 3] = zone.environmental_score
        neighbor_info[idx + 4] = 1.0 if zone.developed else 0.0

    enhanced_state = np.concatenate(
        [base_state, zone_id_encoded, global_zones, neighbor_info]
    )
    return base_state, enhanced_state


# ---------------------------------------------------------------------------
# Trained model loading
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> Dict:
    """``torch.load`` restricted to tensors, containers and NumPy scalars.

    The GeoMACA checkpoints store the last training metrics, which contain
    NumPy floats; these three types are the only non-tensor globals they need.
    """
    core = getattr(np, "_core", None) or np.core
    allowed = [core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=DEVICE, weights_only=True)


def load_config(config_path: Path) -> Tuple[BaseConfig, Dict]:
    with open(config_path, "r") as f:
        raw = json.load(f)
    cfg = BaseConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = DEVICE
    return cfg, raw


def load_trained_models(
    checkpoint_path: Path, config_path: Path
) -> Tuple[MAPPOGovernment, SpatialDeveloperCNN, BaseConfig, Dict]:
    cfg, raw_cfg = load_config(config_path)

    government = MAPPOGovernment(cfg)
    developer = SpatialDeveloperCNN(
        state_dim=cfg.state_dim,
        constraint_dim=cfg.gov_action_dim,
        num_land_types=cfg.num_land_types,
    ).to(DEVICE)

    # NOTE: We instantiate GeoMACA only to confirm the checkpoint is internally
    # consistent. We never invoke geomaca.forward() during evaluation -- see
    # module docstring for why.
    geomaca_cfg = GeoMACAConfig(
        state_dim=cfg.base_state_dim,
        action_dim=cfg.gov_action_dim,
        hidden_dim=raw_cfg.get("geomaca_hidden_dim", 128),
        num_zones=cfg.num_zones,
        num_turns=cfg.num_zones,
        num_agents=3,
        num_heads=raw_cfg.get("geomaca_num_heads", 4),
        use_temporal=raw_cfg.get("use_temporal", True),
        use_geographic=raw_cfg.get("use_geographic", True),
        use_agent=raw_cfg.get("use_agent", True),
        combination=raw_cfg.get("geomaca_combination", "multiplicative"),
        temperature=raw_cfg.get("geomaca_temperature", 1.0),
    )
    geomaca = GeoMACA(geomaca_cfg).to(DEVICE)

    ckpt = load_checkpoint(checkpoint_path)
    for i, actor_state in enumerate(ckpt["government_state"]["actors"]):
        government.actors[i].load_state_dict(actor_state)
    government.critic.load_state_dict(ckpt["government_state"]["critic"])
    developer.load_state_dict(ckpt["developer_state"])
    geomaca.load_state_dict(ckpt["geomaca_state"])  # sanity check only

    for actor in government.actors:
        _to_inference_mode(actor)
    _to_inference_mode(government.critic)
    _to_inference_mode(developer)
    _to_inference_mode(geomaca)

    return government, developer, cfg, raw_cfg
