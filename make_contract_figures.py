"""Generate Arc A figures from results_contract.json into figures/.

  - ladder_isr.png       : ISR by scenario across ContractGuard levels (the
                           strict necessity ladder H6-H9).
  - lambda_negative.png  : ISR under shortcut forgery vs lambda (H14a).

Run: python make_contract_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def _load():
    with open(os.path.join(HERE, "results_contract.json")) as f:
        return json.load(f)


def fig_ladder(results):
    ladder = results["ladder"]
    scenarios = list(ladder.keys())
    levels = ["L0", "L1", "L2", "L3"]
    labels = {
        "S0_honest": "S0 honest",
        "S1_shortcut_forgery": "S1 shortcut\nforgery",
        "S2_signed_overscope": "S2 signed\nover-scope",
        "S3_effect_divergence": "S3 effect\ndivergence",
    }
    data = np.array([[ladder[s][l] for l in levels] for s in scenarios])

    x = np.arange(len(scenarios))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    colors = ["#b03a2e", "#d68910", "#7d6608", "#1e8449"]
    rung = ["L0 (none)", "L1 provenance", "L2 typed-auth", "L3 runtime"]
    for j, l in enumerate(levels):
        ax.bar(x + (j - 1.5) * w, data[:, j], w, label=rung[j], color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in scenarios])
    ax.set_ylabel("Injection success rate (ISR)")
    ax.set_ylim(0, 1.08)
    ax.set_title("ContractGuard strict necessity ladder (RACG, $\\lambda=2$)")
    ax.legend(ncol=2, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "ladder_isr.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_lambda(results):
    rows = results["lambda_negative"]
    lams = [r["lam"] for r in rows]
    isr = [r["isr_under_attack"] for r in rows]
    succ = [r["benign_success"] for r in rows]

    x = np.arange(len(lams))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(x, isr, "o-", color="#b03a2e", lw=2, label="ISR under shortcut forgery")
    ax.plot(x, succ, "s--", color="#1e8449", lw=2, label="benign task success")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel("risk-penalty $\\lambda$")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("The risk knob cannot remove a contract-forgery attack (H14a)")
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "lambda_negative.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    r = _load()
    a = fig_ladder(r)
    b = fig_lambda(r)
    outs = [a, b]
    if "distributional_suite" in r:
        outs.append(fig_distributional(r))
    if "adaptive_attacker" in r:
        outs.append(fig_adaptive(r))
    if "field_ablation" in r:
        outs.append(fig_field(r))
    for o in outs:
        print("wrote", o)


def fig_distributional(results):
    dist = results["distributional_suite"]
    fams = [f for f in ("A1", "A1p", "A4", "A1A4", "A2", "A3") if f in dist]
    levels = ["L0", "L1", "L2", "L3"]
    data = np.array([[dist[f][l]["isr"] for l in levels] for f in fams])
    x = np.arange(len(fams)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = ["#b03a2e", "#d68910", "#7d6608", "#1e8449"]
    rung = ["L0 (none)", "L1 provenance", "L2 typed attest.", "L3 runtime"]
    for j, l in enumerate(levels):
        ax.bar(x + (j - 1.5) * w, data[:, j], w, label=rung[j], color=colors[j])
    pretty = {"A1": "A1\nshortcut", "A1p": "A1'\nover-scope", "A4": "A4\ndivergence",
              "A1A4": "A1+A4\ncompound", "A2": "A2\nrisk-down", "A3": "A3\nauth-alias"}
    ax.set_xticks(x); ax.set_xticklabels([pretty[f] for f in fams])
    ax.set_ylabel("Injection success rate (ISR)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Distributional ISR by attack family (8 core targets)")
    ax.legend(ncol=2, fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "distributional_isr.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def fig_adaptive(results):
    adp = results["adaptive_attacker"]
    targets = list(adp["by_target"].keys())
    levels = ["L0", "L1", "L2", "L3"]
    data = np.array([[adp["by_target"][t][l] for l in levels] for t in targets])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(data, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(["L0\nnone", "L1\nprovenance", "L2\ntyped attest.", "L3\nruntime"])
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=8)
    for i in range(len(targets)):
        for j in range(len(levels)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="white" if data[i, j] > 0.5 else "black", fontsize=8)
    ax.set_title("Adaptive white-box attacker: attack-induced ISR\n"
                 "(exhaustive best response per guard level)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ISR")
    fig.tight_layout()
    out = os.path.join(FIG, "adaptive_isr.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def fig_field(results):
    fa = results["field_ablation"]
    order = ["produces+requires", "risk", "authorizes", "requires", "produces"]
    order = [f for f in order if f in fa]
    vals = [fa[f] for f in order]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bars = ax.barh(range(len(order)), vals, color="#b03a2e")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("ISR under RACG (no guard)")
    ax.set_xlim(0, 1.08)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=8)
    ax.set_title("Which contract field carries the attack\n(effect integrity is primary)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "field_ablation.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


if __name__ == "__main__":
    main()
