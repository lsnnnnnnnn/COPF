#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# --- 核心配置 ---
SEEDS="42"
DEVICE="cuda"
EDGELIST="data/tgb/datasets/tgbl_wiki/tgbl-wiki_edgelist_v2.csv"
OUT_ROOT="out/ablations_wiki_parallel" # <--- 关键：所有脚本输出到这里

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
)

run_one() {
  local exp_name="$1"; shift
  local outdir="${OUT_ROOT}/${exp_name}"
  echo ">>> [Policy Ablation] Running: ${exp_name}"
  mkdir -p "${outdir}"
  python -u scripts/run_copf.py --out_dir "${outdir}" "${COMMON_ARGS[@]}" "$@" --seed "${SEEDS}" |& tee "${outdir}/run.log"
}

# === (i) Exploration Policy ===
# 1. Epsilon Greedy
run_one "policy_epsgreedy" \
  --policy "epsilon_greedy" \
  --epsilon 0.20 --deploy_epsilon 0.02 --post_epsilon 0.02 \
  --covexp_enable

# 2. TopK Stochastic (默认)
run_one "policy_topkstochastic" \
  --policy "topk_stochastic" \
  --topk 10 \
  --temperature 1.0 --deploy_temperature 0.7 --post_temperature 0.7 \
  --covexp_enable