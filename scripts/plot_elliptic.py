"""Render the Elliptic diagnostic figures from a previously generated report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

VARIANTS = {
    "local": ("Local", "#22577a"),
    "structure": ("Structure", "#a34329"),
    "neighborhood": ("Neighborhood", "#537443"),
}
BG = "#fcfaf5"
INK = "#26343b"
MUTED = "#6b7373"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#a7aeaa",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "svg.hashsalt": "fraudguard-elliptic-v1",
        "svg.fonttype": "none",
    }
)


def figure(title, subtitle, figsize):
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.79))
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.06, y=0.985, ha="left").set_in_layout(
        False
    )
    fig.text(0.06, 0.93, subtitle, fontsize=9, color=MUTED, ha="left")
    return fig


def style(ax):
    ax.grid(axis="y", color="#d9ded8", linewidth=0.7)
    ax.set_axisbelow(True)


def pct(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 1.05)


def save(fig, path, name, fmt):
    output = path / f"{name}.{fmt}"
    fig.savefig(
        output,
        format=fmt,
        metadata={"Date": None} if fmt == "svg" else {"Software": "Matplotlib"},
    )
    if fmt == "svg":
        svg = output.read_text(encoding="utf-8")
        output.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8"
        )
    plt.close(fig)


def time_stability(report):
    counts = report["time_step_counts"]
    steps = sorted(map(int, report["test_by_time_step"]["local"]))
    history = report.get("historical_windows", [])
    fig = figure(
        "Performance across time",
        "Labelled cohort only · completed snapshots · diagnostic analysis",
        (11, 10 if history else 7.5),
    )
    axes = fig.subplots(
        3 if history else 2,
        1,
        gridspec_kw={"height_ratios": [1.2, 1, 1.05] if history else [1.2, 1]},
    )
    ax = axes[0]
    for key, (name, color) in VARIANTS.items():
        vals = [report["test_by_time_step"][key][str(s)]["average_precision"] for s in steps]
        ax.plot(
            steps,
            [v if v is not None else float("nan") for v in vals],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=name,
            color=color,
        )
    ax.plot(
        steps,
        [
            counts[str(s)]["prevalence"]
            if counts[str(s)]["prevalence"] is not None
            else float("nan")
            for s in steps
        ],
        linestyle="--",
        color=MUTED,
        linewidth=1.4,
        label="Labelled prevalence (baseline)",
    )
    ax.set(
        title="Test average precision by snapshot",
        ylabel="Average precision / prevalence",
        xticks=steps,
    )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=False)
    pct(ax)
    style(ax)

    ax = axes[1]
    labelled = [counts[str(s)]["labelled"] for s in steps]
    unknown = [counts[str(s)]["unknown"] for s in steps]
    illicit = [counts[str(s)]["illicit"] for s in steps]
    ax.bar(steps, labelled, color="#d8c9ae", label="Labelled")
    ax.bar(steps, unknown, bottom=labelled, color="#e9e8df", label="Unknown")
    ax.set(xlabel="Time step", ylabel="Transactions", xticks=steps, title="Cohort composition")
    style(ax)
    right = ax.twinx()
    right.plot(
        steps,
        illicit,
        color="#a34329",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Illicit (right axis)",
    )
    right.set_ylabel("Illicit transactions", color="#a34329")
    right.tick_params(axis="y", colors="#a34329")
    right.set_ylim(bottom=0)
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0],
        loc="upper right",
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    if history:
        ax = axes[2]
        labels = []
        for i, window in enumerate(history):
            splits = window["splits"]

            def span(split):
                return f"{split['time_min']}–{split['time_max']}"

            labels.append(
                f"Train {span(splits['train'])}\nVal {span(splits['validation'])} / Eval {span(splits['evaluation'])}"
            )
            for j, (key, (name, color)) in enumerate(VARIANTS.items()):
                xs = [i + (j - 1) * 0.17 - 0.05, i + (j - 1) * 0.17 + 0.05]
                vals = [
                    window[f"{split}_metrics"][key]["average_precision"]
                    for split in ("validation", "evaluation")
                ]
                ys = [v if v is not None else float("nan") for v in vals]
                ax.plot(xs, ys, color=color, linewidth=1.1, label=name if i == 0 else None)
                ax.scatter(xs[0], ys[0], color=color, marker="o", s=24)
                ax.scatter(xs[1], ys[1], color=color, marker="s", s=24)
            baselines = [
                splits[split]["illicit"] / splits[split]["labelled"]
                if splits[split]["labelled"]
                else float("nan")
                for split in ("validation", "evaluation")
            ]
            ax.plot(
                [i - 0.05, i + 0.05],
                baselines,
                linestyle="--",
                color=MUTED,
                label="Labelled prevalence" if i == 0 else None,
            )
        ax.scatter([], [], color=INK, marker="o", s=24, label="Validation")
        ax.scatter([], [], color=INK, marker="s", s=24, label="Evaluation")
        ax.set(
            title="Historical windows: paired validation and evaluation AP",
            ylabel="Average precision / prevalence",
            xticks=list(range(len(labels))),
            xticklabels=labels,
        )
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center")
        pct(ax)
        ax.set_ylim(0, 1.28)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        style(ax)
    return fig


def review_budget(report):
    fractions = report["review_policy_config"]["fractions"]
    fig = figure(
        "Review budget trade-offs",
        "Labelled cohort only · completed snapshots · diagnostic analysis",
        (11, 7.6),
    )
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    for col, (policy, title) in enumerate(
        (
            ("pooled_fractions", "Pooled test queue"),
            ("summed_per_step", "Independent queues per step"),
        )
    ):
        for row, metric in enumerate(("precision", "recall")):
            ax = axes[row, col]
            for key, (name, color) in VARIANTS.items():
                data = report["test_review_policies"][key][policy]
                if policy == "summed_per_step":
                    data = data["fractions"]
                vals = [item[metric] for item in data]
                ax.plot(
                    [100 * f for f in fractions],
                    [v if v is not None else float("nan") for v in vals],
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=color,
                    label=name,
                )
            ax.set_xticks([100 * f for f in fractions], [f"{100 * f:g}%" for f in fractions])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(metric.title())
            if row == 1:
                ax.set_xlabel("Prespecified review fraction")
            pct(ax)
            style(ax)
    axes[0, 0].legend(ncol=3, fontsize=9, frameon=False, loc="lower left")
    return fig


def policy_workload(report):
    fig = figure(
        "Workload and illicit transactions found",
        "Summed independent per-step policies · labelled cohort only · completed snapshots",
        (11, 5.4),
    )
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    for ax, (key, (name, color)) in zip(axes, VARIANTS.items(), strict=True):
        policies = report["test_review_policies"][key]["summed_per_step"]
        for kind, marker, desc in (
            ("fractions", "o", "Fraction per step"),
            ("capacities", "s", "Fixed capacity per step"),
        ):
            rows = policies[kind]
            ax.plot(
                [r["reviewed"] for r in rows],
                [r["frauds_found"] for r in rows],
                marker=marker,
                color=color,
                linestyle="-" if kind == "fractions" else "--",
                linewidth=1.5,
                label=desc,
            )
            for r in rows:
                label = f"{r['fraction'] * 100:g}%" if kind == "fractions" else str(r["capacity"])
                ax.annotate(
                    label,
                    (r["reviewed"], r["frauds_found"]),
                    xytext=(4, 7) if kind == "fractions" else (-10, -13),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK,
                )
        threshold = policies["frozen_threshold"]
        ax.scatter(
            [threshold["reviewed"]],
            [threshold["frauds_found"]],
            marker="*",
            s=130,
            color=color,
            edgecolor=INK,
            linewidth=0.6,
            zorder=3,
            label="Validation-frozen threshold",
        )
        ax.set(title=name, xlabel="Total reviewed")
        style(ax)
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Illicit transactions found")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.text(
        0.06,
        0.015,
        "Point labels: fraction of each step or maximum reviews per step. Frozen threshold is not capacity-limited.",
        fontsize=9,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/elliptic.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/elliptic"))
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as f:
        report = json.load(f)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in (
        ("time-stability", time_stability),
        ("review-budget", review_budget),
        ("policy-workload", policy_workload),
    ):
        save(plot(report), args.output_dir, name, args.format)


if __name__ == "__main__":
    main()
