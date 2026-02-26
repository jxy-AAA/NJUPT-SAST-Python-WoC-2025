import json
import math
import re
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def next_output_path(out_dir: Path) -> Path:
    pattern = re.compile(r"^(\d+)\.png$")
    max_idx = 0
    for p in out_dir.glob("*.png"):
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return out_dir / f"{max_idx + 1:02d}.png"


def extract_metrics(rows: List[Dict]) -> Dict[str, List[float]]:
    if not rows:
        raise ValueError("Log file is empty.")

    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
    metrics: Dict[str, List[float]] = {"epoch": epochs}

    first = rows[0]
    for key, value in first.items():
        if key == "epoch":
            continue
        if isinstance(value, (int, float)):
            series: List[float] = []
            for row in rows:
                v = row.get(key)
                if isinstance(v, (int, float)):
                    series.append(float(v))
                else:
                    series.append(float("nan"))
            metrics[key] = series

    if len(metrics) == 1:
        raise ValueError("No numeric metric fields found in log.")
    return metrics


def draw_and_save(metrics: Dict[str, List[float]], save_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = metrics["epoch"]
    names = [k for k in metrics.keys() if k != "epoch"]

    cols = 2
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows))
    flat_axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for i, name in enumerate(names):
        ax = flat_axes[i]
        ax.plot(x, metrics[name], marker="o", linewidth=1.8, markersize=3.5)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    for j in range(len(names), len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def draw_training_curves(log_path: Path, out_dir: Path, title: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(log_path)
    metrics = extract_metrics(rows)
    save_path = next_output_path(out_dir)
    final_title = title if title else f"Training Curves - {log_path.name}"
    draw_and_save(metrics, save_path, final_title)
    return save_path
