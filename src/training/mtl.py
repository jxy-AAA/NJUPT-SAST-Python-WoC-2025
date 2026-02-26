import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim

from datasets import build_denoising_dataloaders
from models import MultiTaskNet, load_single_task_weights
from utils import accuracy, load_checkpoint, load_config, psnr, save_checkpoint, seed_everything, ssim


def evaluate(model, loader, device):
    model.eval()
    denoise_sum = 0.0
    cls_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    acc_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for noisy, clean, labels in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(noisy)
            denoised = out["denoised"]
            logits = out["logits"]

            denoise_loss = F.l1_loss(denoised, clean)
            cls_loss = F.cross_entropy(logits, labels)

            bsz = noisy.size(0)
            denoise_sum += denoise_loss.item() * bsz
            cls_sum += cls_loss.item() * bsz
            psnr_sum += psnr(denoised, clean) * bsz
            ssim_sum += ssim(denoised, clean) * bsz
            acc_sum += accuracy(logits, labels) * bsz
            sample_count += bsz

    denom = max(sample_count, 1)
    return {
        "denoise_loss": denoise_sum / denom,
        "cls_loss": cls_sum / denom,
        "psnr": psnr_sum / denom,
        "ssim": ssim_sum / denom,
        "acc": acc_sum / denom,
    }


def classification_kl_consistency(noisy_logits, clean_logits):
    log_probs_noisy = F.log_softmax(noisy_logits, dim=1)
    probs_clean = F.softmax(clean_logits, dim=1)
    return F.kl_div(log_probs_noisy, probs_clean, reduction="batchmean")


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

    model = MultiTaskNet(
        in_channels=3,
        base_channels=cfg["model"]["base_channels"],
        num_blocks=4,
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    print(f"[Info] model = {model.__class__.__name__}")

    print("[Info] loading single-task checkpoints...")
    task1_ckpt = load_checkpoint(cfg["model"]["task1_ckpt"], map_location="cpu")
    task2_ckpt = load_checkpoint(cfg["model"]["task2_ckpt"], map_location="cpu")

    load_single_task_weights(
        model,
        task1_state_dict=task1_ckpt["model_state"],
        task2_state_dict=task2_ckpt["model_state"],
    )
    print("[Info] single-task weights loaded into MTL model")

    model.freeze_shared_front_half()
    print("[Info] shared front half frozen")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[Info] total params     = {total_count:,}")
    print(f"[Info] trainable params = {trainable_count:,}")

    optimizer = optim.SGD(
        trainable_parameters,
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
    print(f"  epochs               = {cfg['train']['epochs']}")
    print(f"  lr                   = {cfg['train']['lr']}")
    print(f"  momentum             = {cfg['train'].get('momentum', 0.9)}")
    print(f"  weight_decay         = {cfg['train']['weight_decay']}")
    print(f"  grad_clip            = {cfg['train']['grad_clip']}")
    print(f"  noise_std            = {cfg['dataset']['noise_std']}")
    print(f"  lambda_denoise       = {cfg['train']['lambda_denoise']}")
    print(f"  lambda_cls           = {cfg['train']['lambda_cls']}")
    print(f"  lambda_consistency   = {cfg['train']['lambda_consistency']}")
    print(f"  scheduler.step_size  = {cfg['scheduler']['step_size']}")
    print(f"  scheduler.gamma      = {cfg['scheduler']['gamma']}")

    best_score = -1.0
    best_epoch = 0
    ckpt_path = Path("checkpoints") / "best.ptr"

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_total = 0.0
        running_denoise = 0.0
        running_cls = 0.0
        running_cons = 0.0
        running_samples = 0

        print(f"\n===== Epoch {epoch}/{cfg['train']['epochs']} =====")
        print(f"[Train] lr = {optimizer.param_groups[0]['lr']:.6f}")

        for step, (noisy, clean, labels) in enumerate(train_loader, start=1):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(noisy)
            denoised = out["denoised"]
            logits_noisy = out["logits"]

            with torch.no_grad():
                logits_clean = model.predict_clean_logits(clean)

            denoise_loss = F.l1_loss(denoised, clean)
            cls_loss = F.cross_entropy(logits_noisy, labels)
            consistency_loss = classification_kl_consistency(logits_noisy, logits_clean)

            total_loss = (
                cfg["train"]["lambda_denoise"] * denoise_loss
                + cfg["train"]["lambda_cls"] * cls_loss
                + cfg["train"]["lambda_consistency"] * consistency_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, cfg["train"]["grad_clip"])
            optimizer.step()

            bsz = noisy.size(0)
            running_total += total_loss.item() * bsz
            running_denoise += denoise_loss.item() * bsz
            running_cls += cls_loss.item() * bsz
            running_cons += consistency_loss.item() * bsz
            running_samples += bsz

            if step % 10 == 0 or step == len(train_loader):
                avg_total = running_total / max(running_samples, 1)
                avg_denoise = running_denoise / max(running_samples, 1)
                avg_cls = running_cls / max(running_samples, 1)
                avg_cons = running_cons / max(running_samples, 1)
                print(
                    f"[Train][Epoch {epoch}] step {step:>4}/{len(train_loader)} | "
                    f"total={avg_total:.4f} | denoise={avg_denoise:.4f} | "
                    f"cls={avg_cls:.4f} | cons={avg_cons:.4f}"
                )

        train_total = running_total / max(running_samples, 1)
        train_denoise = running_denoise / max(running_samples, 1)
        train_cls = running_cls / max(running_samples, 1)
        train_cons = running_cons / max(running_samples, 1)

        val_metrics = evaluate(model, val_loader, device)
        score = val_metrics["acc"] + val_metrics["psnr"] / 50.0

        print(
            f"[Val][Epoch {epoch}] denoise_loss={val_metrics['denoise_loss']:.4f} | "
            f"cls_loss={val_metrics['cls_loss']:.4f} | psnr={val_metrics['psnr']:.4f} | "
            f"ssim={val_metrics['ssim']:.4f} | acc={val_metrics['acc']:.4f} | "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "best_score": best_score,
                },
            )
            print(f"[Save] New best checkpoint saved -> {ckpt_path} (best_score={best_score:.4f})")

        scheduler.step()

        print(
            f"[Epoch Summary] epoch={epoch} | train_total={train_total:.4f} | "
            f"train_denoise={train_denoise:.4f} | train_cls={train_cls:.4f} | "
            f"train_cons={train_cons:.4f} | val_psnr={val_metrics['psnr']:.4f} | "
            f"val_ssim={val_metrics['ssim']:.4f} | val_acc={val_metrics['acc']:.4f} | "
            f"score={score:.4f} | best_score={best_score:.4f} (epoch {best_epoch})"
        )

    print("\n===== Training Finished =====")
    print(f"[Result] best_score = {best_score:.4f} (epoch {best_epoch})")
    print(f"[Result] best checkpoint = {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mtl.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
