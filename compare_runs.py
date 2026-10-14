"""Compare the loss curves of several training runs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(runs: list[Path], out: Path) -> None:
    histories = {run: json.loads((run / "history.json").read_text()) for run in runs}
    style_keys = sorted({k for h in histories.values() for k in h[0] if k.startswith("val_")})

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    header = f"{'run':24s} {'best val':>9s} {'at step':>8s} {'final train':>12s}"
    print(header + "".join(f" {k[4:]:>8s}" for k in style_keys))
    for run, history in histories.items():
        steps = [h["step"] for h in history]
        best = min(history, key=lambda h: h["val"])
        ax_train.plot(steps, [h["train"] for h in history], label=run.name)
        ax_val.plot(steps, [h["val"] for h in history], label=run.name)
        per_style = "".join(f" {best[k]:>8.4f}" if k in best else f" {'-':>8s}" for k in style_keys)
        print(f"{run.name:24s} {best['val']:>9.4f} {best['step']:>8d} {history[-1]['train']:>12.4f}{per_style}")

    if style_keys:
        print("\nPer-style columns are validation losses at each run's best checkpoint.")

    for ax, title in ((ax_train, "train loss"), (ax_val, "validation loss")):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    ax_train.set_ylabel("cross-entropy loss")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/compare.png"))
    args = parser.parse_args()
    main(args.runs, args.out)
