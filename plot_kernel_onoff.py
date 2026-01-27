# plot_kernel_onoff.py
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 服务器无显示也能保存图片
import matplotlib.pyplot as plt


COLOR_ON  = "#3eaad0"   # kernel on
COLOR_OFF = "#ffaa2c"   # kernel off

ON_PATH = Path("/data1/shengen/fairlink/out/exp_wiki_tgn_full_copf_20260126_083008_3541548/seed42/opp_copf_metrics.csv")
OFF_PATH = Path("/data1/shengen/fairlink/out/exp_wiki_tgn_certificate_only_20260126_083056_3542066/seed42/opp_copf_metrics.csv")

OUT_DIR = Path("./fig_kernel_ablation_seed42")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 统一字体大小：更清晰，但不“很粗” ----
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
    "figure.dpi": 120,
    "savefig.dpi": 260,
})


def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if c != "phase":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("step").reset_index(drop=True)
    return df


def phase_boundaries(df: pd.DataFrame):
    # 返回 [(phase, start_step, end_step), ...]
    d = df[["step", "phase"]].copy()
    d["phase"] = d["phase"].astype(str)
    chunks = []
    cur_phase = None
    start = None
    last_step = None
    for step, ph in d.itertuples(index=False):
        if cur_phase is None:
            cur_phase = ph
            start = int(step)
        elif ph != cur_phase:
            chunks.append((cur_phase, start, int(last_step)))
            cur_phase = ph
            start = int(step)
        last_step = int(step)
    if cur_phase is not None:
        chunks.append((cur_phase, start, int(last_step)))
    return chunks


def add_phase_lines(ax, df: pd.DataFrame):
    chunks = phase_boundaries(df)
    # 用竖线标出 phase 边界
    for (ph, s, e) in chunks:
        ax.text(
            x=(s + e) / 2.0,
            y=1.01,
            s=ph,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11.5,
        )
    # 画边界线（除了第一个 start）
    for (_, s, _) in chunks[1:]:
        ax.axvline(s, color="0.65", linestyle="--", linewidth=1.0, alpha=0.9)


def plot_overlay(df_on, df_off, col, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    # “描边”技巧：先画粗的 OFF，再画细的 ON（即使完全重合也能看出两条）
    ax.plot(df_off["step"], df_off[col], color=COLOR_OFF, lw=3.6, ls="-", alpha=0.85, label="kernel OFF")
    ax.plot(df_on["step"],  df_on[col],  color=COLOR_ON,  lw=2.0, ls="-", alpha=0.95, label="kernel ON")

    add_phase_lines(ax, df_on)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="normal")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{fname}.png")
    fig.savefig(OUT_DIR / f"{fname}.pdf")
    plt.close(fig)


def plot_delta(df_on, df_off, col, ylabel, title, fname):
    m = df_on[["step", col]].merge(df_off[["step", col]], on="step", suffixes=("_on","_off"))
    delta = (m[f"{col}_on"] - m[f"{col}_off"]).astype(float)
    max_abs = float(np.nanmax(np.abs(delta.to_numpy()))) if len(delta) else float("nan")

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(m["step"], delta, color=COLOR_ON, lw=2.2, alpha=0.95)
    ax.axhline(0.0, color="0.55", lw=1.0, alpha=0.9)

    add_phase_lines(ax, df_on)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}    (max|Δ|={max_abs:.3g})", fontweight="normal")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{fname}.png")
    fig.savefig(OUT_DIR / f"{fname}.pdf")
    plt.close(fig)


def plot_kernel_eps(df_on, df_off, disc_col, kernel_col, title, fname):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    # 左轴：离散审计 eps（两条会重合，用描边显示两条）
    ax.plot(df_off["step"], df_off[disc_col], color=COLOR_OFF, lw=3.6, alpha=0.85, label=f"OFF {disc_col} (discrete)")
    ax.plot(df_on["step"],  df_on[disc_col],  color=COLOR_ON,  lw=2.0, alpha=0.95, label=f"ON  {disc_col} (discrete)")
    ax.set_xlabel("step")
    ax.set_ylabel("discrete eps")

    # 右轴：kernel eps（只有 ON 有）
    ax2 = ax.twinx()
    if kernel_col in df_on.columns:
        ax2.plot(df_on["step"], df_on[kernel_col], color=COLOR_ON, lw=2.4, ls="--", alpha=0.95, label=f"ON  {kernel_col} (kernel)")
    ax2.set_ylabel("kernel eps")

    add_phase_lines(ax, df_on)
    ax.set_title(title, fontweight="normal")
    ax.grid(True, alpha=0.18, linewidth=0.6)

    # 合并 legend（不加粗）
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{fname}.png")
    fig.savefig(OUT_DIR / f"{fname}.pdf")
    plt.close(fig)


def main():
    df_on = load_csv(ON_PATH)
    df_off = load_csv(OFF_PATH)

    # 1) 核心效用/公平：overlay（即使重合也“看得见两条”）
    plot_overlay(df_on, df_off, "mrr", "MRR", "Utility over time", "A_utility_mrr")
    plot_overlay(df_on, df_off, "ndcg@10", "NDCG@10", "Utility over time", "A_utility_ndcg10")

    plot_overlay(df_on, df_off, "gTE_gap", "gTE gap", "Fairness over time", "B_fair_gTE")
    plot_overlay(df_on, df_off, "gCal_max", "gCal max", "Fairness over time", "B_fair_gCal")

    # 2) 差分图：证明“toggle 不改行为”
    plot_delta(df_on, df_off, "mrr", "ΔMRR (ON-OFF)", "Delta", "C_delta_mrr")
    plot_delta(df_on, df_off, "gCal_max", "ΔgCal (ON-OFF)", "Delta", "C_delta_gCal")

    # 3) 真正有差异：kernel eps（诊断更严格）
    plot_kernel_eps(df_on, df_off, "eps_r0", "eps_r0_kernel",
                    "Residual-OI eps: discrete vs kernel (calibration residual r0)",
                    "D_eps_r0_discrete_vs_kernel")
    plot_kernel_eps(df_on, df_off, "eps_r_delta", "eps_r_delta_kernel",
                    "Residual-OI eps: discrete vs kernel (TE residual r_delta)",
                    "D_eps_rdelta_discrete_vs_kernel")

    print(f"[OK] wrote figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()



