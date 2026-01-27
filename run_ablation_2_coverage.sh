#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# --- 核心配置 ---
SEEDS="42"
DEVICE="cuda"
EDGELIST="data/tgb/datasets/tgbl_wiki/tgbl-wiki_edgelist_v2.csv"
OUT_ROOT="out/ablations_wiki_parallel"

# 时间步
PRE_T=20000; DEPLOY_T=20000; POST_T=20000
T_TOTAL=$((PRE_T + DEPLOY_T + POST_T))

# 通用参数
COMMON_ARGS=(
  --dataset tgb --tgb_edgelist "${EDGELIST}" --model tgn
  --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}"
  --neg 200 --hits_k 10 --outcome_mode bandit 
  --group_on dst --device "${DEVICE}"
  --log_every 1000 --audit_every 1000 --oi_window 50000
  --emb_dim 128 --msg_dim 128 --train_every 1
  --adv_lambda 0.0 --reweight none --fair_penalty_lambda 0.0
  --pre_apply_calibrator --pd_enable --pd_apply_phases deploy,post
  # Coverage对比时，固定使用 TopK 策略
  --policy "topk_stochastic" --topk 10
  --temperature 1.0 --deploy_temperature 0.7 --post_temperature 0.7
  --tgb_group_mode node_mod --tgb_group_n 2
)

run_one() {
  local exp_name="$1"; shift
  local outdir="${OUT_ROOT}/${exp_name}"
  echo ">>> [Coverage Ablation] Running: ${exp_name}"
  mkdir -p "${outdir}"
  python -u scripts/run_copf.py --out_dir "${outdir}" "${COMMON_ARGS[@]}" "$@" --seed "${SEEDS}" |& tee "${outdir}/run.log"
}

# === (ii) Coverage-driven Exploration ===
# 1. Off (关闭)
run_one "covexp_off" 
# 注意：不传 --covexp_enable 即为关闭

# 2. Target 0.01
run_one "covexp_001" \
  --covexp_enable --covexp_ptar 0.01

# 3. Target 0.05
run_one "covexp_005" \
  --covexp_enable --covexp_ptar 0.05