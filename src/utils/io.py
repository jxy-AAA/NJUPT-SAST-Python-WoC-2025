import json
from pathlib import Path
from typing import Dict

import torch


def output_dirs(root: str) :
    base = Path(root)/"outputs"
    dirs = {
        "base": base,
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def append_jsonl(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(payload, f)
        f.write("\n")
#记录数据


def save_checkpoint(path: Path, state: Dict) -> None:
   path = Path(path)
   path.parent.mkdir(parents=True, exist_ok=True)
   torch.save(state, str(path))


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    path = Path(path)
    checkpoint = torch.load(str(path), map_location=map_location)
    return checkpoint


def count_trainable_parameters(model: torch.nn.Module) -> int:
    trainable = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable += param.numel()
    return trainable
