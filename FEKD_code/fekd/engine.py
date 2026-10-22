"""FEKD 蒸馏训练与部署学生导出。"""

import json
import time
import torch
from .utils import now, write_json, save_torch, state_hash


def train(model, loader, directory, device, settings):
    """使用显式传入的参数训练；每轮覆盖保存最新的部署学生。"""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=settings.learning_rate,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay,
        nesterov=settings.nesterov,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.milestones, gamma=settings.lr_decay
    )
    teacher_before = state_hash(model.teacher.state_dict())
    for epoch in range(1, settings.epochs + 1):
        began = time.time()
        model.train()
        count = correct = 0
        sums = {}
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            logits, loss, terms, _ = model.training_loss(images, labels, epoch)
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
                raise FloatingPointError(f"nonfinite gradient at epoch {epoch}")
            if settings.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    parameters, settings.max_grad_norm, error_if_nonfinite=True
                )
            optimizer.step()
            count += len(labels)
            correct += int(logits.detach().argmax(-1).eq(labels).sum())
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * len(labels)
        if count == 0:
            raise ValueError("empty training data")
        row = dict(
            epoch=epoch,
            learning_rate=learning_rate,
            train_top1=correct / count,
            epoch_seconds=time.time() - began,
            **{key: value / count for key, value in sums.items()},
        )
        with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        # 部署文件保存学生、适配器与复用分类器；教师骨干和交互模块留在训练内存。
        exported = model.export_student()
        exported["epoch"] = epoch
        save_torch(directory / "student.pth", exported)
        scheduler.step()
        write_json(directory / "status.json", dict(status="training", **row))
        print(
            f"Epoch {epoch}/{settings.epochs} lr={learning_rate:.6g} train={100*row['train_top1']:.2f}% loss={sum(sums.values())/count:.6g}",
            flush=True,
        )
    if state_hash(model.teacher.state_dict()) != teacher_before:
        raise RuntimeError("frozen teacher parameters or normal-path BN statistics changed")
    write_json(
        directory / "status.json",
        dict(status="completed", finished_at=now(), epochs=settings.epochs),
    )
