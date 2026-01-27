# FAIRLINK: COPF

This repository accompanies our **deployment-stable counterfactual fairness** for **online link prediction / link recommendation** on evolving graphs.

## This artifact contains

- A unified **OPP** runner that logs utility + counterfactual fairness metrics over **Pre → Deploy → Post**
- Backbones: **EdgeBank**, **TGN**, **GraphMixer**, **TGN_Adv**, **TGN_Penalty**, **TGN_Reweight**
- COPF components:
  - **Online auditing / certificates** (e.g., residual-style OI windowing)
  - **Coverage-driven exploration** (overlap/identification support)
  - **Primal–dual controller** to coordinate utility–fairness trade-offs

---

## Installation

### Requirements
- Python 3.9+ (recommended: 3.11)
- PyTorch (CUDA optional)

## Data (not included)

Due to **dataset size and licensing constraints**, we do **not** redistribute any TGB files, and we also do not upload pre-generated synthetic datasets.

### 1) TGB datasets (tgbl-wiki / tgbl-review)

We evaluate on Temporal Graph Benchmark (TGB) dynamic link property prediction datasets.  
Please download the datasets using the official `py-tgb` package (auto-download supported).

**Install:**
```bash
pip install py-tgb
```

**Download (recommended):**
```bash
python - <<'PY'
from tgb.linkproppred.dataset import LinkPropPredDataset
for name in ["tgbl-wiki", "tgbl-review"]:
    ds = LinkPropPredDataset(name=name, root="data/tgb/datasets", preprocess=True)
    print(name, "downloaded under:", ds.root)
PY
```

**Using our runner:**
- You may run using the repo’s TGB loader (if available) **or** by passing an explicit CSV edgelist path.
- Example (CSV edgelist):
```bash
python -u scripts/run_copf.py \
  --dataset tgb \
  --tgb_edgelist data/tgb/datasets/tgbl_review/tgbl-review_edgelist_v2.csv \
  ...
```

**Note on group labels (no demographics in TGB):**
Because TGB does not ship demographic attributes, we construct static group labels using
`--tgb_group_mode` (e.g., `node_mod`, `src_degree`, `node_degree`) and choose which endpoint is treated as the protected attribute via `--group_on {src|dst}`.

### 2) Synthetic bipartite stream (generated locally)

Synthetic data can be generated locally (small files) and placed anywhere.

**Generate:**
```bash
python scripts/make_synth_bipartite.py \
  --out_dir data/synth/bipartite_v1/seed42 \
  --seed 2026 --n_users 600 --n_items 4000 --n_events 200000
```

This writes (example):
- `edges.csv` (columns: `src,dst,t`)
- `nodes.csv` (columns: `node,group`)
- `meta.json`

**Run on synthetic:**
```bash
python -u scripts/run_copf.py \
  --dataset synth \
  --data_dir data/synth/bipartite_v1/seed42 \
  --model edgebank \
  --group_on src \
  ...
```

> Important: the provided synthetic generator assigns group labels to **users** and sets item groups to `-1`,
> so `--group_on src` is the intended setting for synthetic runs.

---

## Reproducing results

### Output format

Each run writes a folder under an output root (e.g., `out/`) containing:
- `opp_copf_metrics.csv` (phase-wise logs; includes utility + fairness metrics)
- `run.log` (stdout/stderr log)

Example:
```bash
out/<batch_name>/copf/tgn/seed2026/opp_copf_metrics.csv
out/<batch_name>/copf/tgn/seed2026/run.log
```

### Main batch script (TGB-Review)

Below is the exact batch script we use to run **EdgeBank / TGN / GraphMixer** on **tgbl-review** with **COPF enabled**, including fairness-training baselines for TGN (**Adv / Reweight / Penalty**) under the same OPP logging protocol.

To keep the README readable, it is placed in a collapsible block—copy/paste as-is into a shell.

<details>
<summary><b>Click to expand: full batch script for tgbl-review (seeds 2026/2027/2028)</b></summary>

```bash
bash <<'BASH'
set -euo pipefail
export PYTHONUNBUFFERED=1

SEEDS="2026 2027 2028"

PRE_T=20000
DEPLOY_T=20000
POST_T=20000
T_TOTAL=$((PRE_T + DEPLOY_T + POST_T))

NEG=200
HITS_K=10

POLICY="topk_stochastic"
TOPK=10

EPS_PRE=0.20
EPS_DEPLOY=0.02
EPS_POST=0.02

TEMP_PRE=1.0
TEMP_DEPLOY=0.7
TEMP_POST=0.7

LOG_EVERY=1000
OI_WINDOW=50000

AUDIT_EVERY=1000
DEVICE="cuda"

TGN_EMB=128
TGN_MSG=128
ADV_LAMBDA=1.0
PEN_LAMBDA=1.0

GM_FEAT_DIM=64
GM_TIME_FEAT_DIM=64
GM_NUM_TOKENS=20
GM_NUM_LAYERS=2
GM_NUM_NEIGHBORS=10
GM_TIME_GAP=2000
GM_NEG="${NEG}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EDGELIST="data/tgb/datasets/tgbl_review/tgbl-review_edgelist_v2.csv"
OUT_ROOT="out/batch_tgbl_review_${RUN_TAG}"

run_one () {
  local outdir="$1"; shift
  mkdir -p "${outdir}"
  echo "==================== RUN: ${outdir} ===================="
  python -u scripts/run_copf.py --out_dir "${outdir}" "$@" |& tee "${outdir}/run.log"
}

PHASE_ARGS=(--pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}")

POLICY_ARGS=(
  --policy "${POLICY}"
  --topk "${TOPK}"
  --epsilon "${EPS_PRE}"
  --temperature "${TEMP_PRE}"
  --deploy_topk "${TOPK}" --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}"
  --post_topk "${TOPK}" --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}"
)

COMMON_ARGS=(
  --hits_k "${HITS_K}"
  --log_every "${LOG_EVERY}"
  --oi_window "${OI_WINDOW}"
  --outcome_mode bandit
  --group_on dst
  --device "${DEVICE}"
)

DATA_ARGS=(--dataset tgb --tgb_edgelist "${EDGELIST}" --tgb_group_mode node_mod --tgb_group_n 2 --tgb_root "tgb_baselines")

COPF_CTRL_ARGS=(--audit_every "${AUDIT_EVERY}" --pre_apply_calibrator --covexp_enable --pd_enable --pd_apply_phases deploy,post)

for seed in ${SEEDS}; do
  run_one "${OUT_ROOT}/copf/edgebank/seed${seed}" \
    "${DATA_ARGS[@]}" --model edgebank \
    "${PHASE_ARGS[@]}" "${POLICY_ARGS[@]}" \
    --neg "${NEG}" "${COMMON_ARGS[@]}" \
    "${COPF_CTRL_ARGS[@]}" --seed "${seed}"

  run_one "${OUT_ROOT}/copf/tgn/seed${seed}" \
    "${DATA_ARGS[@]}" --model tgn \
    "${PHASE_ARGS[@]}" "${POLICY_ARGS[@]}" \
    --neg "${NEG}" "${COMMON_ARGS[@]}" \
    --emb_dim "${TGN_EMB}" --msg_dim "${TGN_MSG}" --train_every 1 \
    --adv_lambda 0.0 --reweight none --fair_penalty_lambda 0.0 \
    "${COPF_CTRL_ARGS[@]}" --seed "${seed}"

  run_one "${OUT_ROOT}/copf/tgn_adv/seed${seed}" \
    "${DATA_ARGS[@]}" --model tgn \
    "${PHASE_ARGS[@]}" "${POLICY_ARGS[@]}" \
    --neg "${NEG}" "${COMMON_ARGS[@]}" \
    --emb_dim "${TGN_EMB}" --msg_dim "${TGN_MSG}" --train_every 1 \
    --adv_lambda "${ADV_LAMBDA}" --reweight none --fair_penalty_lambda 0.0 \
    "${COPF_CTRL_ARGS[@]}" --seed "${seed}"

  run_one "${OUT_ROOT}/copf/tgn_reweight/seed${seed}" \
    "${DATA_ARGS[@]}" --model tgn \
    "${PHASE_ARGS[@]}" "${POLICY_ARGS[@]}" \
    --neg "${NEG}" "${COMMON_ARGS[@]}" \
    --emb_dim "${TGN_EMB}" --msg_dim "${TGN_MSG}" --train_every 1 \
    --adv_lambda 0.0 --reweight inv_freq --fair_penalty_lambda 0.0 \
    "${COPF_CTRL_ARGS[@]}" --seed "${seed}"

  run_one "${OUT_ROOT}/copf/tgn_penalty/seed${seed}" \
    "${DATA_ARGS[@]}" --model tgn \
    "${PHASE_ARGS[@]}" "${POLICY_ARGS[@]}" \
    --neg "${NEG}" "${COMMON_ARGS[@]}" \
    --emb_dim "${TGN_EMB}" --msg_dim "${TGN_MSG}" --train_every 1 \
    --adv_lambda 0.0 --reweight none --fair_penalty_lambda "${PEN_LAMBDA}" \
    "${COPF_CTRL_ARGS[@]}" --seed "${seed}"

  run_one "${OUT_ROOT}/copf/graphmixer/seed${seed}" \
    "${DATA_ARGS[@]}" --model graphmixer \
    "${PHASE_ARGS[@]}" "${POLICY_ARGS[@]}" \
    --neg "${GM_NEG}" "${COMMON_ARGS[@]}" \
    --emb_dim "${GM_FEAT_DIM}" \
    --gm_time_feat_dim "${GM_TIME_FEAT_DIM}" --gm_num_tokens "${GM_NUM_TOKENS}" --gm_num_layers "${GM_NUM_LAYERS}" \
    --gm_num_neighbors "${GM_NUM_NEIGHBORS}" --gm_time_gap "${GM_TIME_GAP}" \
    "${COPF_CTRL_ARGS[@]}" --seed "${seed}"
done

echo "[DONE] tgbl_review COPF runs finished. OUT_ROOT=${OUT_ROOT}"
BASH
```

</details>
