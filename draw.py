import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils.plot import draw_training_curves


def pick_latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No jsonl logs found in: {log_dir}")
    return logs[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw training metrics from jsonl and save numbered png.")
    parser.add_argument("--log", type=str, default="", help="Path to jsonl log. Default: latest in outputs/logs")
    parser.add_argument("--log-dir", type=str, default="outputs/logs", help="Directory of jsonl logs")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to save numbered png")
    parser.add_argument("--title", type=str, default="", help="Custom plot title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    log_path = Path(args.log) if args.log else pick_latest_log(log_dir)
    save_path = draw_training_curves(log_path, out_dir, title=args.title)

    print(f"log: {log_path}")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
