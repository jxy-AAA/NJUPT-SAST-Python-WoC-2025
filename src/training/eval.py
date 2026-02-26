import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets import build_classification_dataloaders, build_denoising_dataloaders
from models import MultiTaskNet, Task1DenoiseNet, Task2ClassifierNet
from utils import accuracy, load_checkpoint, load_config, psnr, ssim


def eval_task1(cfg, device):
    _, loader = build_denoising_dataloaders(
        root=cfg["dataset"]["root"],
        noise_std=cfg["dataset"]["noise_std"],
        train_batch_size=cfg["dataset"]["train_batch_size"],
        eval_batch_size=cfg["dataset"]["eval_batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
    )
    model = Task1DenoiseNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["features"],
        num_blocks=cfg["model"]["depth"],
    ).to(device)
    ckpt = load_checkpoint("outputs/checkpoints/task1_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    model.eval()
    n = 0
    l1 = p = s = 0.0
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            b = noisy.size(0)
            n += b
            l1 += F.l1_loss(pred, clean).item() * b
            p += psnr(pred, clean) * b
            s += ssim(pred, clean) * b
    return {"mode": "task1", "l1": l1 / n, "psnr": p / n, "ssim": s / n}


def eval_task2(cfg, device):
    _, loader = build_classification_dataloaders(
        root=cfg["dataset"]["root"],
        train_batch_size=cfg["dataset"]["train_batch_size"],
        eval_batch_size=cfg["dataset"]["eval_batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
    )
    model = Task2ClassifierNet(
        in_channels=3,
        base_channels=cfg["model"]["base_channels"],
        num_blocks=4,
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    ckpt = load_checkpoint("outputs/checkpoints/task2_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    model.eval()
    n = 0
    ce = acc = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            b = x.size(0)
            n += b
            ce += F.cross_entropy(logits, y).item() * b
            acc += accuracy(logits, y) * b
    return {"mode": "task2", "ce": ce / n, "acc": acc / n}


def eval_mtl(cfg, device):
    _, loader = build_denoising_dataloaders(
        root=cfg["dataset"]["root"],
        noise_std=cfg["dataset"]["noise_std"],
        train_batch_size=cfg["dataset"]["train_batch_size"],
        eval_batch_size=cfg["dataset"]["eval_batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
    )
    model = MultiTaskNet(
        in_channels=3,
        base_channels=cfg["model"]["base_channels"],
        num_blocks=4,
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    ckpt = load_checkpoint("outputs/checkpoints/mtl_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    model.eval()
    n = 0
    l1 = p = s = ce = acc = 0.0
    with torch.no_grad():
        for noisy, clean, labels in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            labels = labels.to(device)

            out = model(noisy)
            denoised = out["denoised"]
            logits = out["logits"]
            b = noisy.size(0)
            n += b

            l1 += F.l1_loss(denoised, clean).item() * b
            p += psnr(denoised, clean) * b
            s += ssim(denoised, clean) * b
            ce += F.cross_entropy(logits, labels).item() * b
            acc += accuracy(logits, labels) * b

    return {"mode": "mtl", "l1": l1 / n, "psnr": p / n, "ssim": s / n, "ce": ce / n, "acc": acc / n}


def save_eval_plot(metrics, mode: str, output_dir: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = [k for k in metrics.keys() if k != "mode"]
    metric_values = [float(metrics[k]) for k in metric_names]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = out_dir / f"{mode}_{ts}.png"

    width = max(6, int(1.5 * len(metric_names)))
    fig, ax = plt.subplots(figsize=(width, 4))
    bars = ax.bar(metric_names, metric_values)
    ax.set_title(f"Evaluation Metrics - {mode}")
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["task1", "task2", "mtl"], required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--no-plot", action="store_true", help="only print metrics, do not save plot")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "task1":
        metrics = eval_task1(cfg, device)
    elif args.mode == "task2":
        metrics = eval_task2(cfg, device)
    else:
        metrics = eval_mtl(cfg, device)

    print(metrics)

    if not args.no_plot:
        plot_path = save_eval_plot(metrics, args.mode, args.output_dir)
        print(f"[Plot] saved to {plot_path}")


if __name__ == "__main__":
    main()
