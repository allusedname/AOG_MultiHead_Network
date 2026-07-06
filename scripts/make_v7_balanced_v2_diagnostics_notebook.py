from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(src).strip() + "\n")


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(src).strip() + "\n")


def main() -> None:
    out = ROOT / "experiments" / "notebooks" / "v7_balanced_v2_diagnostics.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    root_literal = str(ROOT)
    nb["cells"] = [
        md("# V7 Balanced V2 Diagnostics"),
        code(f"""
            from pathlib import Path
            import json
            import pandas as pd
            import matplotlib.pyplot as plt

            ROOT = Path({root_literal!r})
            RUN = ROOT / "runs" / "v7_balanced_v2"
            FIG = RUN / "diagnostics"
            FIG.mkdir(parents=True, exist_ok=True)

            summary = json.loads((RUN / "diagnostic_summary.json").read_text())
            flags = pd.read_csv(RUN / "failure_flags_by_sample.csv")
            topn = pd.read_csv(RUN / "candidate_scores_topn.csv")
            scores = pd.read_csv(RUN / "class_score_decomposition.csv")
            conf_long = pd.read_csv(RUN / "confusion_matrix_long.csv")
            name_map = topn[["candidate_class", "candidate_name"]].drop_duplicates().set_index("candidate_class")["candidate_name"].to_dict()
            flags["correct"] = flags["correct"].astype(bool)
            flags["true_name"] = flags["true_class"].map(name_map)
            flags["pred_name"] = flags["pred_class"].map(name_map)
            summary
        """),
        code("""
            baseline = {"accuracy": 0.6439834024896266, "true_class_not_in_top5": 20, "missing_required_pred": 402}
            pd.DataFrame([
                {"metric": k, "v1_balanced": v, "v2_balanced": summary[k], "delta": summary[k] - v}
                for k, v in baseline.items()
            ])
        """),
        code("""
            per_class = []
            for cid, group in flags.groupby("true_class"):
                wrong = group[~group["correct"]]
                per_class.append({
                    "class": name_map.get(cid, str(cid)),
                    "n": len(group),
                    "accuracy": group["correct"].mean(),
                    "wrong": len(wrong),
                    "not_top5": int(group["true_class_not_in_top5"].sum()),
                    "missing_required_pred": int((group["missing_required_pred"] > 0).sum()),
                    "most_common_wrong": None if wrong.empty else wrong["pred_name"].value_counts().idxmax(),
                })
            per_class = pd.DataFrame(per_class).sort_values("accuracy")
            display(per_class)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(per_class["class"], per_class["accuracy"], color="#4C78A8")
            ax.set_ylim(0, 1)
            ax.set_ylabel("accuracy")
            ax.set_title("V7 balanced V2 per-class accuracy")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(FIG / "v2_per_class_accuracy.png", dpi=180)
            plt.show()
        """),
        code("""
            conf = conf_long.pivot(index="true_class", columns="pred_class", values="count").fillna(0)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(conf.values, cmap="Blues")
            ax.set_xticks(range(len(conf.columns)))
            ax.set_yticks(range(len(conf.index)))
            ax.set_xticklabels([name_map.get(i, str(i)) for i in conf.columns], rotation=45, ha="right")
            ax.set_yticklabels([name_map.get(i, str(i)) for i in conf.index])
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title("V7 balanced V2 confusion matrix")
            for i in range(conf.shape[0]):
                for j in range(conf.shape[1]):
                    value = int(conf.iloc[i, j])
                    if value:
                        ax.text(j, i, str(value), ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(FIG / "v2_confusion_matrix.png", dpi=180)
            plt.show()

            off = conf_long[conf_long["true_class"] != conf_long["pred_class"]].copy()
            off["true_name"] = off["true_class"].map(name_map)
            off["pred_name"] = off["pred_class"].map(name_map)
            off.sort_values("count", ascending=False).head(15)
        """),
        code("""
            winners = topn[topn["rank"] == 1].merge(flags[["sample_id", "correct"]], on="sample_id")
            true_rows = scores[scores["candidate_class"] == scores["true_class"]].merge(flags[["sample_id", "correct"]], on="sample_id")
            cols = ["node_presence_score", "node_absence_score", "token_prototype_score", "part_template_score", "required_slot_penalty", "total_score", "missing_required"]
            winner_means = winners.groupby("correct")[cols].mean(numeric_only=True)
            true_means = true_rows.groupby("correct")[cols].mean(numeric_only=True)
            display(winner_means)
            display(true_means)

            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for ax, col in zip(axes.ravel(), ["node_presence_score", "node_absence_score", "token_prototype_score", "required_slot_penalty", "total_score", "missing_required"]):
                vals = winner_means[col].reindex([True, False])
                ax.bar(["correct", "wrong"], vals.values, color=["#4C78A8", "#E45756"])
                ax.set_title(col)
                ax.axhline(0, color="black", linewidth=0.7)
            fig.suptitle("V2 winning-candidate component means")
            fig.tight_layout()
            fig.savefig(FIG / "v2_winner_score_components.png", dpi=180)
            plt.show()
        """),
        code("""
            ranks = flags["true_class_rank"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(ranks.index.astype(str), ranks.values, color="#72B7B2")
            ax.set_xlabel("true class rank")
            ax.set_ylabel("samples")
            ax.set_title("V7 balanced V2 true-class rank")
            fig.tight_layout()
            fig.savefig(FIG / "v2_true_class_rank.png", dpi=180)
            plt.show()
            ranks
        """),
        md("""
            ## Interpretation

            V2 should be judged by whether token prototype evidence and explicit absence
            reduce ambiguity without reintroducing class collapse. The remaining weak
            case is reptile, especially reptile-to-quadruped and reptile-to-bird.
        """),
    ]
    nbf.write(nb, out)
    print(out)


if __name__ == "__main__":
    main()
