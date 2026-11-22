# GeoMACA and CityBuilder

> Code for Chapter 5 of the PhD thesis *Multi-Level Agent-Based Land Use Decision-Making Using Multi-Agent Reinforcement Learning* (Lukasz Pelcner, Lancaster University, 2026).

**GeoMACA** (Geographic Multi-Agent Credit Assignment) divides a shared end-of-episode reward among agents, time steps and locations with three attention heads. **CityBuilder** is an urban-growth simulator in which three government agents set planning constraints, a developer builds the city zone by zone and households move in and out. It reports nine economic, environmental and social metrics.

The repository has the credit-assignment module, the simulator, the training code for GeoMACA and for the QMIX and COMA baselines, a fixed Heuristic government, the evaluation that produces the tables and the figure of the chapter, and the Level-Based Foraging experiments.

If you use this code, please cite the thesis (see [Citation](#citation)).

## What is where

| | |
|---|---|
| `chapter5_geomaca/geomaca` | the credit-assignment module and its losses |
| `chapter5_geomaca/citybuilder` | the simulator and the baseline governments |
| `chapter5_geomaca/training` | GeoMACA training, QMIX and COMA |
| `chapter5_geomaca/evaluation` | the common evaluation, tables and Figure 5.3 |
| `chapter5_geomaca/lbf` | Level-Based Foraging experiments |
| `chapter5_geomaca/records` | the evaluation records behind the printed tables |
| `chapter5_geomaca/checkpoints` | manifest and check of the released checkpoints |
| `chapter5_geomaca/tests` | tests |

## Setup

Python 3.10 or newer (tested with 3.13).

```bash
git clone https://github.com/FamishedHound/geomaca-citybuilder.git
cd geomaca-citybuilder
pip install -r requirements.txt
```

Run everything from the repository root. Training the CityBuilder agents uses a GPU when there is one; the evaluation and the Level-Based Foraging runs use the CPU.

## Reproducing the results of Chapter 5

**From the records in the repository** (no checkpoints, a few seconds). Writes every table and Figure 5.3 to `chapter5_geomaca/results/`:

```bash
python -m chapter5_geomaca.reproduce all --from-records
```

**From the trained checkpoints.** They are attached to the [release](https://github.com/FamishedHound/geomaca-citybuilder/releases) as `chapter5_checkpoints.zip` (about 131 MB): the GeoMACA, ablation, QMIX and COMA checkpoints and the 60 Level-Based Foraging result files.

```bash
python -m zipfile -e chapter5_checkpoints.zip chapter5_checkpoints
python -m chapter5_geomaca.checkpoints.verify --root chapter5_checkpoints
python -m chapter5_geomaca.reproduce all --checkpoints chapter5_checkpoints \
    --lbf-results chapter5_checkpoints/lbf --workers 4
```

One result at a time:

| In the thesis | Command |
|---|---|
| Table 5.2 | `python -m chapter5_geomaca.reproduce table5.2 --checkpoints chapter5_checkpoints --workers 4` |
| Table 5.3 | `python -m chapter5_geomaca.reproduce table5.3 --checkpoints chapter5_checkpoints` |
| Figure 5.3 | `python -m chapter5_geomaca.reproduce figure5.3 --checkpoints chapter5_checkpoints` |
| Table 5.4 | `python -m chapter5_geomaca.reproduce table5.4 --checkpoints chapter5_checkpoints --workers 4` |
| Table 5.5 | `python -m chapter5_geomaca.reproduce table5.5 --workers 4` |
| Tables 5.6 and 5.7 | `python -m chapter5_geomaca.reproduce table5.6 --lbf-results chapter5_checkpoints/lbf` (and `table5.7`) |
| Table 5.8 | `python -m chapter5_geomaca.lbf.attention_analysis --runs runs/lbf_attention --reference chapter5_checkpoints/lbf --out runs/lbf_attention/analysis` |

Tables 5.2 to 5.4 and Figure 5.3 evaluate the checkpoints (2,800 episodes in all, a few seconds each) and share one evaluation record, so the first command does most of the work. The evaluation does not seed the background terrain, so two evaluations of the same checkpoints differ slightly; `--seed-all` makes one repeatable.

**Training from scratch.** On one RTX 5090 a GeoMACA run took 42 to 77 minutes with ten runs at a time.

```bash
python -m chapter5_geomaca.training.launch geomaca   --out runs --max-concurrent 10
python -m chapter5_geomaca.training.launch ablations --out runs --max-concurrent 10
python -m chapter5_geomaca.training.launch baselines --out runs --max-concurrent 12
python -m chapter5_geomaca.reproduce all --checkpoints runs --workers 4

# Level-Based Foraging (Tables 5.6 and 5.7), then with the attention weights recorded (Table 5.8)
python -m chapter5_geomaca.lbf.train --chapter5 --method all \
    --seeds 42 123 456 789 1024 1337 2048 3141 4096 5555 \
    --episodes 50000 --eval-episodes 100 --results-dir runs/lbf
python -m chapter5_geomaca.lbf.train --chapter5 --method geomaca_fixed \
    --seeds 42 123 456 789 1024 1337 2048 3141 4096 5555 \
    --episodes 50000 --eval-episodes 100 --log-attention-every 25 --results-dir runs/lbf_attention
```

A retrained run need not be bit-identical to the released one, because GPU arithmetic differs between GPU models and drivers.

Tests: `python -m pytest chapter5_geomaca/tests`

## Citation

```bibtex
@phdthesis{pelcner2026thesis,
  title  = {Multi-Level Agent-Based Land Use Decision-Making Using Multi-Agent Reinforcement Learning},
  author = {Pelcner, Lukasz},
  school = {Lancaster University},
  year   = {2026},
  note   = {Chapter 5}
}
```

## License

MIT, see [LICENSE](LICENSE).
