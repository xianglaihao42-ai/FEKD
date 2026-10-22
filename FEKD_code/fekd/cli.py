"""学生训练与学生推理共用的命令行工具。训练参数全部显式输入。"""

import argparse
from dataclasses import asdict, fields
from pathlib import Path
import sys
import torch
from .data import training_loader
from .engine import train
from .models.distiller import Options, build_model, StudentInference
from .settings import TrainingSettings
from .utils import seed_all, write_json, file_hash, now


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu or install a CUDA PyTorch build")
    return torch.device(name)


def training_parser():
    parser = argparse.ArgumentParser(
        description="Train a FEKD student with a user-supplied teacher."
    )
    parser.add_argument(
        "--teacher-factory", required=True, help="external module:function or file.py:function"
    )
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--teacher-prior-raw", required=True, type=float, help="teacher log prior b"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--download", action="store_true", help="download CIFAR-100 through torchvision"
    )
    for flag in (
        "epochs",
        "batch-size",
        "workers",
        "cpu-threads",
        "seed",
        "factor",
        "extra-paths",
        "feature-warmup",
        "output-warmup",
    ):
        parser.add_argument("--" + flag, required=True, type=int)
    for flag in (
        "learning-rate",
        "momentum",
        "weight-decay",
        "lr-decay",
        "max-grad-norm",
        "student-prior-raw",
        "label-weight",
        "output-weight",
        "feature-weight",
        "auxiliary-weight",
    ):
        parser.add_argument("--" + flag, required=True, type=float)
    parser.add_argument(
        "--milestones",
        required=True,
        type=int,
        nargs="*",
        help="decay epochs; leave empty for none",
    )
    parser.add_argument("--nesterov", required=True, action=argparse.BooleanOptionalAction)
    return parser


def training_main():
    parser = training_parser()
    args = parser.parse_args()
    options = Options(**{field.name: getattr(args, field.name) for field in fields(Options)})
    settings = TrainingSettings(
        **{field.name: getattr(args, field.name) for field in fields(TrainingSettings)}
    )
    try:
        options.validate()
        settings.validate()
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if args.output_dir.exists():
        parser.error("--output-dir already exists; select a new directory")
    torch.set_num_threads(settings.cpu_threads)
    seed_all(options.seed)
    device = resolve_device(args.device)
    model = build_model(
        args.teacher_checkpoint, args.teacher_factory, args.teacher_prior_raw, options
    ).to(device)
    loader = training_loader(
        args.data_root, settings.batch_size, settings.workers, options.seed, device, args.download
    )
    model.class_names = list(loader.dataset.classes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "parameters.json",
        {
            "method": asdict(options),
            "training": asdict(settings),
            "teacher_factory": args.teacher_factory,
            "teacher_prior_raw": args.teacher_prior_raw,
            "teacher_checkpoint_sha256": file_hash(args.teacher_checkpoint),
            "training_samples": len(loader.dataset),
            "device": str(device),
            "created_at": now(),
            "python": sys.version,
            "torch": str(torch.__version__),
        },
    )
    print(f"Output: {args.output_dir.resolve()}", flush=True)
    try:
        train(model, loader, args.output_dir, device, settings)
    except BaseException as error:
        write_json(
            args.output_dir / "status.json",
            {"status": "failed", "error": repr(error), "failed_at": now()},
        )
        raise


def load_inference(checkpoint, device):
    """部署学生独立加载；推理过程使用学生文件中保存的分类器。"""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return StudentInference(payload).to(device).eval()
