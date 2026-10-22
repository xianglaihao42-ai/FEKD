"""对单张图片或目录中的图片推理，导出期望概率和总浓度。"""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import _bootstrap
from fekd.cli import load_inference, resolve_device
from fekd.data import image_transform
from fekd.utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="an image or a directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--cpu-threads", required=True, type=int)
    parser.add_argument("--device", required=True, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100:
        parser.error("--top-k must be in [1,100]")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.input.is_dir():
        files = sorted(
            p
            for p in args.input.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
    elif args.input.is_file():
        files = [args.input]
    else:
        raise FileNotFoundError(args.input)
    if not files:
        raise ValueError("no images found")
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    model = load_inference(args.checkpoint, device)
    preprocessing = transforms.Compose([transforms.Resize((32, 32)), image_transform()])
    names = getattr(model, "class_names", None)
    records = []
    with torch.inference_mode():
        for path in files:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                tensor = preprocessing(image).unsqueeze(0).to(device)
            alpha = model.alpha(model(tensor))
            strength = alpha.sum(-1, keepdim=True)
            probabilities = alpha / strength  # Dirichlet 期望；不对 alpha 再做 softmax。
            confidence, indices = probabilities[0].topk(args.top_k)
            records.append(
                {
                    "image": str(path),
                    "alpha0": float(strength[0, 0]),
                    "top_k": [
                        {
                            "class_index": int(index),
                            "class_name": names[int(index)] if names else None,
                            "probability": float(probability),
                        }
                        for probability, index in zip(confidence.cpu(), indices.cpu())
                    ],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"checkpoint": str(args.checkpoint), "predictions": records})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
