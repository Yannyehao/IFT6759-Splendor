# Splendor AI: Reward Shaping and Guided Lookahead

This repository contains an IFT6759 course project on reinforcement learning and planning for the board game **Splendor**. The project studies how different reward designs and search strategies affect agent strength in a large, dynamic discrete action space.

The main result is that shallow planning with a high-quality semantic evaluator is more effective than deeper AlphaZero-style search in this setting. A MaskablePPO policy trained with event-based reward shaping reaches **77.9%** against a Greedy baseline, while PPO+Lookahead reaches **91.9%** with `K=15` and **93.0%** with `K=30`.

## Key Results

All main PPO and PPO+Lookahead results use alternating first player and Wilson confidence intervals.

| Agent | vs Random | vs Greedy | 95% CI vs Greedy | Protocol |
|---|---:|---:|---:|---|
| Score-Based MaskablePPO | 94.8% | 75.8% | [73.0%, 78.4%] | n=1000 |
| Event-Based MaskablePPO | 94.3% | 77.9% | [75.2%, 80.4%] | n=1000 |
| PPO+Lookahead, `K=15` | 97.8% | 91.9% | [90.0%, 93.4%] | n=1000 |
| PPO+Lookahead, `K=30` | 96.9% | 93.0% | [91.2%, 94.4%] | n=1000 |

AlphaZero-style MCTS variants were also implemented and evaluated. They did not approach the PPO baselines; the best AlphaZero checkpoint reached 24.0% against Greedy, while later shaped, warm-started, longer, and combined variants were at or below 5.0% against Greedy.

## Project Narrative

The project follows five stages:

1. **Sparse rewards**: PPO learns basic legality but struggles with long-horizon strategy.
2. **Score-based shaping**: MaskablePPO with score-progress rewards becomes a strong baseline.
3. **Event-based shaping**: semantic rewards for buying cards, reaching 15 points, scarcity taking, blocking, and engine spikes improve stability and slightly improve Greedy win rate.
4. **AlphaZero-style search**: deep MCTS is a poor fit for Splendor under our compute budget because the game has high branching factor, long horizons, stochastic card draws, and hidden deck order.
5. **PPO+Lookahead**: PPO selects top candidate actions, then a 1-step forward simulator re-ranks them using event evaluation and PPO value estimates.

The core conclusion is:

> In Splendor, evaluation quality matters more than search depth. Semantic event evaluation plus shallow lookahead outperforms deeper but noisy tree search.

## Repository Structure

| Path | Purpose |
|---|---|
| `modules/` | Legacy Splendor environment, classic agents, arena tools, and game mechanics. |
| `project/src/` | Current PPO, event shaping, planning adapter, MCTS, MuZero, and utility code. |
| `project/configs/` | YAML configs for PPO, event-based PPO, curriculum runs, AlphaZero, and MuZero. |
| `project/scripts/` | Training, evaluation, debugging, plotting, and helper scripts. |
| `project/experiments/` | Evaluation outputs, robust reports, ablation results, and experiment indices. |
| `project/logs/` | Training run folders, checkpoints, saved models, and run-local configs. |
| `project/web/` | Human-vs-agent web interface and play logs. |
| `docs/` | Project planning and feasibility notes. |
| `legacy/` | Earlier experiments and examples retained for reference. |
| `report.md` | LaTeX final report source copied from Overleaf. |

## Setup

Recommended environment:

- Python 3.10
- CUDA-capable GPU for training, CPU is sufficient for most evaluation scripts
- Windows + WSL2 was used for the main training runs

Install dependencies:

```bash
python -m pip install -r project/requirements.txt
python -m pip install -r project/requirements-dev.txt
```

The repository uses `sitecustomize.py` to add `modules/` to `PYTHONPATH`, so most commands should be run from the repository root.

## Training

Train the score-based MaskablePPO baseline:

```bash
python project/scripts/train/train_ppo.py \
  --config project/configs/training/maskable_ppo_v4a_ent_lr.yaml
```

Train the event-based MaskablePPO baseline:

```bash
python project/scripts/train/train_ppo.py \
  --config project/configs/training/maskable_ppo_event_v1.yaml
```

Each run writes a timestamped folder under `project/logs/` containing checkpoints, `final_model.zip`, evaluation outputs, and a copy of the config.

## Evaluation

Run robust PPO evaluation:

```bash
python project/scripts/evaluate/evaluate_robust.py \
  --model project/logs/<run_dir>/eval/best_model.zip \
  --config project/logs/<run_dir>/config.yaml \
  --games 1000 \
  --batches 10 \
  --tag <tag>
```

Evaluate PPO+Lookahead:

```bash
python project/scripts/evaluate/evaluate_ppo_lookahead.py \
  --ppo-model project/logs/ppo_event_based/maskable_ppo_event_v1_20260309_110155/eval/best_model.zip \
  --games 1000 \
  --depth 1 \
  --top-k 15 \
  --bucket canonical \
  --tag v1_d1_n1000
```

Run the `K=30` best configuration:

```bash
python project/scripts/evaluate/evaluate_ppo_lookahead.py \
  --ppo-model project/logs/ppo_event_based/maskable_ppo_event_v1_20260309_110155/eval/best_model.zip \
  --games 1000 \
  --depth 1 \
  --top-k 30 \
  --bucket ablation \
  --tag ablation_k30
```

Run tests:

```bash
pytest project/tests
```

## Canonical Evidence Files

Score-based PPO:

- `project/experiments/evaluation/robust/ppo_robust/score_based/robust_eval_v4a_20260308_143224_report.md`

Event-based PPO:

- `project/experiments/evaluation/robust/ppo_robust/event_based/robust_eval_v5_event_20260309_211510_report.md`

PPO+Lookahead:

- `project/experiments/evaluation/robust/ppo_lookahead/canonical/ppo_lookahead_v1_d0_n1000_20260408_173700.md`
- `project/experiments/evaluation/robust/ppo_lookahead/canonical/ppo_lookahead_v1_d1_n1000_20260408_184934.md`
- `project/experiments/evaluation/robust/ppo_lookahead/ablation/ppo_lookahead_ablation_k30_20260409_112636.md`

AlphaZero/MCTS:

- `project/experiments/evaluation/robust/mcts/canonical/`
- `project/experiments/evaluation/robust/mcts/archive/`

## Final Report

The final report source is in `report.md`. It uses a LaTeX article/ICLR-style template even though the file extension is `.md`, because it was copied from Overleaf for local editing.

## Notes

- Use the robust evaluation files as the source of truth for reported numbers.
- Some older documents in `docs/` and `project/docs/development/` describe planned directions that were later superseded by final experiments.
- Large model checkpoints and logs are kept in `project/logs/`; depending on submission requirements, these may be shared separately from the source code.

---

Developed for IFT6759, Winter 2026.
