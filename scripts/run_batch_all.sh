#!/usr/bin/env bash
set -euo pipefail

# =========================
# Global knobs (你只改这里就行)
# =========================
SEEDS=(0 1 2)

# Phase lengths (对齐 OPP 的 pre/deploy/post)
PRE_T=20000
DEPLOY_T=20000
POST_T=20000
T_TOTAL=$((PRE_T + DEPLOY_T + POST_T))

NEG=200

# Policy knobs (建议 topk>=10，不然 bandit 正样本命中率太低，模型很难学到东西)
POLICY="topk_stochastic"
TOPK=10
EPS_PRE=0.20
EPS_DEPLOY=0.02
EPS_POST=0.02
TEMP_PRE=1.0
TEMP_DEPLOY=0.7
TEMP_POST=0.7

LOG_EVERY=1000

# Baseline: 禁用校准器更新（但仍会记录审计指标/证书）
AUDIT_EVERY_BASE=1000000000

# COPF: 开启审计更新
AUDIT_EVERY_COPF=200

# Audit / DR / OI 的关键超参（显式写出来，方便复现实验）
OI_WINDOW=50000
CF_UPDATE_EVERY=200
AUD_BUCKETS=10
AUD_MIN_MASS=0.02
AUD_BOOTSTRAP_B=200
DR_CLIP=1.0

# COPF 的 PD + coverage（显式写出来）
PD_TE_TARGET=0.05
PD_CAL_TARGET=0.05
PD_GAMMA_P=0.20
PD_GAMMA_I=0.02
PD_OFFSET_SCALE=1.0
COV_PMIN=0.005
COV_PTAR=0.10
COV_EPS=0.10
COV_BUCKETS=10
COV_UPDATE_BUCKETS_EVERY=1000
COV_WARMUP=50

# Paths
SYNTH_DIR="data/synth/bipartite_v1"
OUT_ROOT="out/batch_runs"

# =========================
# Helper: find a wiki-like TGB edge list under ./data
# =========================
find_wiki_edgelist() {
python - <<'PY'
import os
import pandas as pd

def ok_cols(cols):
    cols = [c.lower() for c in cols]
    s = set(cols)
    # accept common schemas: (src,dst,t) or (u,i,ts) or (source,target,time)
    if {"src","dst"}.issubset(s):
        return True
    if {"u","i"}.issubset(s):
        return True
    if {"source","target"}.issubset(s):
        return True
    return False

cands = []
for root, _, files in os.walk("data"):
    for f in files:
        fl = f.lower()
        if not fl.endswith(".csv"):
            continue
        if any(x in fl for x in ["node", "nodes", "meta", "feat", "feature", "label"]):
            continue
        p = os.path.join(root, f)
        if "wiki" not in p.lower():
            continue
        try:
            df = pd.read_csv(p, nrows=1)
        except Exception:
            continue
        if ok_cols(df.columns):
            cands.append(p)

if not cands:
    raise SystemExit("ERROR: 没在 ./data 下面找到包含 src/dst(或 u/i) 列、且路径/文件名带 wiki 的 csv。你可以先 `find data -type f | grep -i wiki` 看看。")

# pick the shortest path (usually the canonical one)
cands.sort(key=lambda x: (len(x), x))
print(cands[0])
PY
}

WIKI_EDGELIST="$(find_wiki_edgelist)"
echo "[INFO] using WIKI_EDGELIST=${WIKI_EDGELIST}"

# =========================
# Runner functions
# =========================
run_synth_base() {
  local model="$1"
  local seed="$2"
  local out_dir="${OUT_ROOT}/synth/base/${model}/seed${seed}"

  if [[ "${model}" == "edgebank" ]]; then
    python scripts/run_copf.py \
      --dataset synth --data_dir "${SYNTH_DIR}" --bipartite \
      --model edgebank --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_BASE}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --device cpu --seed "${seed}"
  elif [[ "${model}" == "tgn" ]]; then
    python scripts/run_copf.py \
      --dataset synth --data_dir "${SYNTH_DIR}" --bipartite \
      --model tgn --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_BASE}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --emb_dim 64 --msg_dim 64 --train_every 1 \
      --device cuda --seed "${seed}"
  elif [[ "${model}" == "graphmixer" ]]; then
    python scripts/run_copf.py \
      --dataset synth --data_dir "${SYNTH_DIR}" --bipartite \
      --model graphmixer --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_BASE}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --gm_time_feat_dim 32 --gm_num_tokens 10 --gm_num_layers 1 \
      --gm_num_neighbors 10 --gm_time_gap 2000 \
      --device cuda --seed "${seed}"
  else
    echo "Unknown model: ${model}" >&2
    exit 1
  fi
}

run_synth_copf() {
  local model="$1"
  local seed="$2"
  local out_dir="${OUT_ROOT}/synth/copf/${model}/seed${seed}"

  if [[ "${model}" == "edgebank" ]]; then
    python scripts/run_copf.py \
      --dataset synth --data_dir "${SYNTH_DIR}" --bipartite \
      --model edgebank --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_COPF}" \
      --pre_apply_calibrator \
      --pd_enable --pd_apply_phases deploy,post \
      --pd_te_target "${PD_TE_TARGET}" --pd_cal_target "${PD_CAL_TARGET}" \
      --pd_gamma_p "${PD_GAMMA_P}" --pd_gamma_i "${PD_GAMMA_I}" --pd_offset_scale "${PD_OFFSET_SCALE}" \
      --covexp_enable \
      --covexp_pmin "${COV_PMIN}" --covexp_ptar "${COV_PTAR}" --covexp_eps "${COV_EPS}" \
      --covexp_buckets_per_group "${COV_BUCKETS}" --covexp_update_buckets_every "${COV_UPDATE_BUCKETS_EVERY}" \
      --covexp_warmup_rounds "${COV_WARMUP}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --device cpu --seed "${seed}"
  elif [[ "${model}" == "tgn" ]]; then
    python scripts/run_copf.py \
      --dataset synth --data_dir "${SYNTH_DIR}" --bipartite \
      --model tgn --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_COPF}" \
      --pre_apply_calibrator \
      --pd_enable --pd_apply_phases deploy,post \
      --pd_te_target "${PD_TE_TARGET}" --pd_cal_target "${PD_CAL_TARGET}" \
      --pd_gamma_p "${PD_GAMMA_P}" --pd_gamma_i "${PD_GAMMA_I}" --pd_offset_scale "${PD_OFFSET_SCALE}" \
      --covexp_enable \
      --covexp_pmin "${COV_PMIN}" --covexp_ptar "${COV_PTAR}" --covexp_eps "${COV_EPS}" \
      --covexp_buckets_per_group "${COV_BUCKETS}" --covexp_update_buckets_every "${COV_UPDATE_BUCKETS_EVERY}" \
      --covexp_warmup_rounds "${COV_WARMUP}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --emb_dim 64 --msg_dim 64 --train_every 1 \
      --device cuda --seed "${seed}"
  elif [[ "${model}" == "graphmixer" ]]; then
    python scripts/run_copf.py \
      --dataset synth --data_dir "${SYNTH_DIR}" --bipartite \
      --model graphmixer --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_COPF}" \
      --pre_apply_calibrator \
      --pd_enable --pd_apply_phases deploy,post \
      --pd_te_target "${PD_TE_TARGET}" --pd_cal_target "${PD_CAL_TARGET}" \
      --pd_gamma_p "${PD_GAMMA_P}" --pd_gamma_i "${PD_GAMMA_I}" --pd_offset_scale "${PD_OFFSET_SCALE}" \
      --covexp_enable \
      --covexp_pmin "${COV_PMIN}" --covexp_ptar "${COV_PTAR}" --covexp_eps "${COV_EPS}" \
      --covexp_buckets_per_group "${COV_BUCKETS}" --covexp_update_buckets_every "${COV_UPDATE_BUCKETS_EVERY}" \
      --covexp_warmup_rounds "${COV_WARMUP}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --gm_time_feat_dim 32 --gm_num_tokens 10 --gm_num_layers 1 \
      --gm_num_neighbors 10 --gm_time_gap 2000 \
      --device cuda --seed "${seed}"
  else
    echo "Unknown model: ${model}" >&2
    exit 1
  fi
}

run_tgb_base() {
  local model="$1"
  local seed="$2"
  local out_dir="${OUT_ROOT}/tgb_wiki/base/${model}/seed${seed}"

  if [[ "${model}" == "edgebank" ]]; then
    python scripts/run_copf.py \
      --dataset tgb --tgb_edgelist "${WIKI_EDGELIST}" --tgb_root "tgb_baselines" \
      --tgb_group_mode node_degree --tgb_group_n 2 --tgb_group_warmup 20000 \
      --group_on dst \
      --model edgebank --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_BASE}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --device cpu --seed "${seed}"
  elif [[ "${model}" == "tgn" ]]; then
    python scripts/run_copf.py \
      --dataset tgb --tgb_edgelist "${WIKI_EDGELIST}" --tgb_root "tgb_baselines" \
      --tgb_group_mode node_degree --tgb_group_n 2 --tgb_group_warmup 20000 \
      --group_on dst \
      --model tgn --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_BASE}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --emb_dim 64 --msg_dim 64 --train_every 1 \
      --device cuda --seed "${seed}"
  elif [[ "${model}" == "graphmixer" ]]; then
    python scripts/run_copf.py \
      --dataset tgb --tgb_edgelist "${WIKI_EDGELIST}" --tgb_root "tgb_baselines" \
      --tgb_group_mode node_degree --tgb_group_n 2 --tgb_group_warmup 20000 \
      --group_on dst \
      --model graphmixer --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_BASE}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --gm_time_feat_dim 32 --gm_num_tokens 10 --gm_num_layers 1 \
      --gm_num_neighbors 10 --gm_time_gap 2000 \
      --device cuda --seed "${seed}"
  else
    echo "Unknown model: ${model}" >&2
    exit 1
  fi
}

run_tgb_copf() {
  local model="$1"
  local seed="$2"
  local out_dir="${OUT_ROOT}/tgb_wiki/copf/${model}/seed${seed}"

  if [[ "${model}" == "edgebank" ]]; then
    python scripts/run_copf.py \
      --dataset tgb --tgb_edgelist "${WIKI_EDGELIST}" --tgb_root "tgb_baselines" \
      --tgb_group_mode node_degree --tgb_group_n 2 --tgb_group_warmup 20000 \
      --group_on dst \
      --model edgebank --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_COPF}" \
      --pre_apply_calibrator \
      --pd_enable --pd_apply_phases deploy,post \
      --pd_te_target "${PD_TE_TARGET}" --pd_cal_target "${PD_CAL_TARGET}" \
      --pd_gamma_p "${PD_GAMMA_P}" --pd_gamma_i "${PD_GAMMA_I}" --pd_offset_scale "${PD_OFFSET_SCALE}" \
      --covexp_enable \
      --covexp_pmin "${COV_PMIN}" --covexp_ptar "${COV_PTAR}" --covexp_eps "${COV_EPS}" \
      --covexp_buckets_per_group "${COV_BUCKETS}" --covexp_update_buckets_every "${COV_UPDATE_BUCKETS_EVERY}" \
      --covexp_warmup_rounds "${COV_WARMUP}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --device cpu --seed "${seed}"
  elif [[ "${model}" == "tgn" ]]; then
    python scripts/run_copf.py \
      --dataset tgb --tgb_edgelist "${WIKI_EDGELIST}" --tgb_root "tgb_baselines" \
      --tgb_group_mode node_degree --tgb_group_n 2 --tgb_group_warmup 20000 \
      --group_on dst \
      --model tgn --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_COPF}" \
      --pre_apply_calibrator \
      --pd_enable --pd_apply_phases deploy,post \
      --pd_te_target "${PD_TE_TARGET}" --pd_cal_target "${PD_CAL_TARGET}" \
      --pd_gamma_p "${PD_GAMMA_P}" --pd_gamma_i "${PD_GAMMA_I}" --pd_offset_scale "${PD_OFFSET_SCALE}" \
      --covexp_enable \
      --covexp_pmin "${COV_PMIN}" --covexp_ptar "${COV_PTAR}" --covexp_eps "${COV_EPS}" \
      --covexp_buckets_per_group "${COV_BUCKETS}" --covexp_update_buckets_every "${COV_UPDATE_BUCKETS_EVERY}" \
      --covexp_warmup_rounds "${COV_WARMUP}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --emb_dim 64 --msg_dim 64 --train_every 1 \
      --device cuda --seed "${seed}"
  elif [[ "${model}" == "graphmixer" ]]; then
    python scripts/run_copf.py \
      --dataset tgb --tgb_edgelist "${WIKI_EDGELIST}" --tgb_root "tgb_baselines" \
      --tgb_group_mode node_degree --tgb_group_n 2 --tgb_group_warmup 20000 \
      --group_on dst \
      --model graphmixer --out_dir "${out_dir}" \
      --pre_T "${PRE_T}" --deploy_T "${DEPLOY_T}" --post_T "${POST_T}" --T "${T_TOTAL}" \
      --neg "${NEG}" \
      --policy "${POLICY}" --topk "${TOPK}" --epsilon "${EPS_PRE}" --temperature "${TEMP_PRE}" \
      --deploy_epsilon "${EPS_DEPLOY}" --deploy_temperature "${TEMP_DEPLOY}" \
      --post_epsilon "${EPS_POST}" --post_temperature "${TEMP_POST}" \
      --log_every "${LOG_EVERY}" --audit_every "${AUDIT_EVERY_COPF}" \
      --pre_apply_calibrator \
      --pd_enable --pd_apply_phases deploy,post \
      --pd_te_target "${PD_TE_TARGET}" --pd_cal_target "${PD_CAL_TARGET}" \
      --pd_gamma_p "${PD_GAMMA_P}" --pd_gamma_i "${PD_GAMMA_I}" --pd_offset_scale "${PD_OFFSET_SCALE}" \
      --covexp_enable \
      --covexp_pmin "${COV_PMIN}" --covexp_ptar "${COV_PTAR}" --covexp_eps "${COV_EPS}" \
      --covexp_buckets_per_group "${COV_BUCKETS}" --covexp_update_buckets_every "${COV_UPDATE_BUCKETS_EVERY}" \
      --covexp_warmup_rounds "${COV_WARMUP}" \
      --oi_window "${OI_WINDOW}" --cf_update_every "${CF_UPDATE_EVERY}" \
      --aud_buckets_per_group "${AUD_BUCKETS}" --aud_min_mass "${AUD_MIN_MASS}" --aud_bootstrap_B "${AUD_BOOTSTRAP_B}" \
      --dr_clip "${DR_CLIP}" --dr_self_norm \
      --gm_time_feat_dim 32 --gm_num_tokens 10 --gm_num_layers 1 \
      --gm_num_neighbors 10 --gm_time_gap 2000 \
      --device cuda --seed "${seed}"
  else
    echo "Unknown model: ${model}" >&2
    exit 1
  fi
}

# =========================
# Execute grid
# =========================
MODELS=(edgebank tgn graphmixer)

for seed in "${SEEDS[@]}"; do
  for m in "${MODELS[@]}"; do
    echo "==================== [SYNTH BASE] model=${m} seed=${seed} ===================="
    run_synth_base "${m}" "${seed}"
    echo "==================== [SYNTH COPF] model=${m} seed=${seed} ===================="
    run_synth_copf "${m}" "${seed}"

    echo "==================== [TGB-WIKI BASE] model=${m} seed=${seed} ===================="
    run_tgb_base "${m}" "${seed}"
    echo "==================== [TGB-WIKI COPF] model=${m} seed=${seed} ===================="
    run_tgb_copf "${m}" "${seed}"
  done
done

echo "[OK] all runs finished. Outputs under: ${OUT_ROOT}"
