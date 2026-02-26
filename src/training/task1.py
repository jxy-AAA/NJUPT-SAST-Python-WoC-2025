import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim

from datasets import build_denoising_dataloaders
from models import Task1DenoiseNet
from utils import (
    AverageMeter,
    count_trainable_parameters,
    load_config,
    psnr,
    save_checkpoint,
    seed_everything,
    ssim,
)


def evaluate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy)
            loss = F.l1_loss(pred, clean)

            bsz = noisy.size(0)
            loss_meter.update(loss.item(), bsz)
            psnr_meter.update(psnr(pred, clean), bsz)
            ssim_meter.update(ssim(pred, clean), bsz)

    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def train(config_path: str):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    train_loader, val_loader = build_denoising_dataloaders(
        root=cfg["dataset"]["root"],
        noise_std=cfg["dataset"]["noise_std"],
        train_batch_size=cfg["dataset"]["train_batch_size"],
        eval_batch_size=cfg["dataset"]["eval_batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
    )
    print("[Info] dataloader ready")
    print(f"[Info] train batches = {len(train_loader)}")
    print(f"[Info] val batches   = {len(val_loader)}")

    model = Task1DenoiseNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["features"],
        num_blocks=cfg["model"]["depth"],
    ).to(device)
    print(f"[Info] model = {model.__class__.__name__}")
    print(f"[Info] trainable params = {count_trainable_parameters(model):,}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        momentum=cfg["train"].get("momentum", 0.9),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler"]["step_size"],
        gamma=cfg["scheduler"]["gamma"],
    )

    print("[Info] training config:")
    print(f"  epochs        = {cfg['train']['epochs']}")
    print(f"  lr            = {cfg['train']['lr']}")
    print(f"  momentum      = {cfg['train'].get('momentum', 0.9)}")
    print(f"  weight_decay  = {cfg['train']['weight_decay']}")
    print(f"  grad_clip     = {cfg['train']['grad_clip']}")
    print(f"  noise_std     = {cfg['dataset']['noise_std']}")
    print(f"  step_size     = {cfg['scheduler']['step_size']}")
    print(f"  gamma         = {cfg['scheduler']['gamma']}")

    best_psnr = -1.0
    best_epoch = 0
    ckpt_path = Path(cfg.get("output_dir", "outputs")) / "checkpoints" / "task1_best.pt"

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        print(f"\n===== Epoch {epoch}/{cfg['train']['epochs']} =====")
        print(f"[Train] lr = {optimizer.param_groups[0]['lr']:.6f}")

        for step, (noisy, clean, _) in enumerate(train_loader, start=1):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            pred = model(noisy)
            loss = F.l1_loss(pred, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            bsz = noisy.size(0)
            running_loss += loss.item() * bsz
            running_samples += bsz

            if step % 10 == 0 or step == len(train_loader):
                avg_loss = running_loss / max(running_samples, 1)
                print(
                    f"[Train][Epoch {epoch}] "
                    f"step {step:>4}/{len(train_loader)} | "
                    f"l1_loss={avg_loss:.4f}"
                )

        train_loss = running_loss / max(running_samples, 1)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_psnr = val_metrics["psnr"]
        val_ssim = val_metrics["ssim"]

        print(
            f"[Val][Epoch {epoch}] "
            f"loss={val_loss:.4f} | psnr={val_psnr:.4f} | ssim={val_ssim:.4f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "best_psnr": best_psnr,
                },
            )
            print(
                f"[Save] New best checkpoint saved -> {ckpt_path} "
                f"(best_psnr={best_psnr:.4f})"
            )

        scheduler.step()

        print(
            f"[Epoch Summary] epoch={epoch} | "
            f"train_l1={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_psnr={val_psnr:.4f} | "
            f"val_ssim={val_ssim:.4f} | "
            f"best_psnr={best_psnr:.4f} (epoch {best_epoch})"
        )

    print("\n===== Training Finished =====")
    print(f"[Result] best_psnr = {best_psnr:.4f} (epoch {best_epoch})")
    print(f"[Result] best checkpoint = {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task1.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
