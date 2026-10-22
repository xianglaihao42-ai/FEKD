"""CIFAR ResNets: SimKD head + FCFD paths + evidential outputs.

References: DefangChen/SimKD models/util.py; LiuDongyang6/FCFD
classification/models/meta.py. Mixed-path BN statistics remain separate during training.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

from .resnet import resnet8x4
from ..external import build_external_model
from ..losses import alpha_from_logits, evidential_ce, dirichlet_kl

CANDIDATES = (("s2t", 1), ("s2t", 2), ("t2s", 1), ("t2s", 2))


@dataclass
class Options:
    """由使用者显式给出的训练参数；完整 FEKD 固定使用特征交互与 Dirichlet KL。"""

    seed: int
    student_prior_raw: float
    factor: int
    extra_paths: int
    label_weight: float
    output_weight: float
    auxiliary_weight: float
    feature_weight: float
    feature_warmup: int
    output_warmup: int
    reuse_teacher_head: ClassVar[bool] = True
    use_fcfd: ClassVar[bool] = True
    evidence_mapping: ClassVar[str] = "exp"

    def validate(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.factor) is not int or not 1 <= self.factor <= 256:
            raise ValueError("factor must be an integer in [1,256]")
        if type(self.extra_paths) is not int or self.extra_paths not in range(1, 5):
            raise ValueError("extra_paths must be an integer in [1,4]")
        for key in ("feature_warmup", "output_warmup"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if (
            not math.isfinite(self.student_prior_raw)
            or not 0 < math.exp(self.student_prior_raw) < math.inf
        ):
            raise ValueError("student prior must yield a finite positive exp(b)")
        for key in ("label_weight", "output_weight", "auxiliary_weight", "feature_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"invalid {key}")


def pool(x):
    return F.adaptive_avg_pool2d(F.relu(x), 1).flatten(1)


def backbone_features(model, image):
    x = F.relu(model.bn1(model.conv1(image)))
    features = [x]
    for k in (1, 2, 3):
        x, _ = getattr(model, f"layer{k}")(x)
        features.append(x)
    return features


def bridge(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.1)
    )


def simkd_projector(factor, cin=256, cout=256):
    mid = cout // factor
    return nn.Sequential(
        nn.Conv2d(cin, mid, 1, bias=False),
        nn.BatchNorm2d(mid),
        nn.ReLU(),
        nn.Conv2d(mid, mid, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid),
        nn.ReLU(),
        nn.Conv2d(mid, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


class BNTrace(nn.Module):
    """一条交互路径专用的 BN 运行统计量；仿射参数仍与主网络共享。"""

    def __init__(self, bn):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros_like(bn.running_mean))
        self.register_buffer("running_var", torch.ones_like(bn.running_var))
        self.register_buffer("num_batches_tracked", torch.zeros_like(bn.num_batches_tracked))


class DistillationModel(nn.Module):
    """训练模型：冻结教师，优化学生、特征桥与最终适配器。"""

    def __init__(self, student, teacher, teacher_prior_raw, options):
        super().__init__()
        options.validate()
        self.options = options
        self.student, self.teacher = student, teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.register_buffer(
            "prior_s", torch.tensor(options.student_prior_raw, dtype=torch.float64)
        )
        self.register_buffer("prior_t", torch.tensor(teacher_prior_raw, dtype=torch.float64))
        self.register_buffer("path_draw_count", torch.zeros((), dtype=torch.long))
        if options.reuse_teacher_head:
            self.student.fc = nn.Identity()  # remove unused classifier from optimizer/export
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.seed + 1701)
            self.final_adapter = (
                simkd_projector(factor=options.factor)
                if options.reuse_teacher_head
                else nn.Identity()
            )
            self.s2t, self.t2s = nn.ModuleDict(), nn.ModuleDict()
            if options.use_fcfd:
                for k, ch in ((1, 64), (2, 128)):
                    self.s2t[str(k)] = bridge(ch, ch)
                    self.t2s[str(k)] = bridge(ch, ch)
                # With head reuse, final_adapter is ALSO the final FCFD bridge.
                if not options.reuse_teacher_head:
                    self.s2t["3"] = bridge(256, 256)
            for group in (self.final_adapter, self.s2t, self.t2s):
                for m in group.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        self.traces = nn.ModuleDict()
        if options.use_fcfd:
            for direction, k in CANDIDATES:
                target = teacher if direction == "s2t" else student
                for stage in range(k + 1, 4):
                    for name, bn in getattr(target, f"layer{stage}").named_modules():
                        if isinstance(bn, nn.BatchNorm2d):
                            self.traces[self.trace_key(direction, k, stage, name)] = BNTrace(bn)
            if options.reuse_teacher_head:
                for direction, k in CANDIDATES:
                    if direction != "t2s":
                        continue
                    for name, bn in self.final_adapter.named_modules():
                        if isinstance(bn, nn.BatchNorm2d):
                            self.traces[self.trace_key(direction, k, 4, name)] = BNTrace(bn)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def trace_key(direction, k, stage, name):
        return f"{direction}_{k}_{stage}_" + name.replace(".", "_")

    @contextmanager
    def mixed_bn(self, module, direction, k, stage):
        """临时切换到交互路径统计量；finally 保证主路径状态被恢复。"""
        saved = []
        try:
            for name, bn in module.named_modules():
                if not isinstance(bn, nn.BatchNorm2d):
                    continue
                trace = self.traces[self.trace_key(direction, k, stage, name)]
                saved.append(
                    (bn, bn.training, bn.running_mean, bn.running_var, bn.num_batches_tracked)
                )
                bn.running_mean, bn.running_var = trace.running_mean, trace.running_var
                bn.num_batches_tracked = trace.num_batches_tracked
                bn.training = self.training
            yield
        finally:
            for bn, mode, mean, var, count in saved:
                bn.training = mode
                bn.running_mean, bn.running_var, bn.num_batches_tracked = mean, var, count

    def student_predict(self, final_feature):
        if self.options.reuse_teacher_head:
            return self.teacher.fc(pool(self.final_adapter(final_feature)))
        return self.student.fc(pool(final_feature))

    def forward(self, image):
        return self.student_predict(backbone_features(self.student, image)[-1])

    def alpha(self, logits):
        return alpha_from_logits(logits, self.prior_s, self.options.evidence_mapping)

    def sample_paths(self):
        """按使用者的种子与运行计数采样交互路径。"""
        paths = random.Random(self.options.seed + int(self.path_draw_count)).sample(
            CANDIDATES, self.options.extra_paths
        )
        self.path_draw_count.add_(1)
        return paths

    def output_loss(self, logits, teacher_logits):
        a = alpha_from_logits(teacher_logits.detach(), self.prior_t, self.options.evidence_mapping)
        b = self.alpha(logits)
        return dirichlet_kl(a, b).mean()

    def training_loss(self, image, labels, epoch, paths=None):
        o = self.options
        # 正常教师路径作为固定目标；交互教师后缀仍需对输入保留梯度。
        with torch.no_grad():
            ft = backbone_features(self.teacher, image)
            zt = self.teacher.fc(pool(ft[-1]))
        fs = backbone_features(self.student, image)
        final = self.final_adapter(fs[3]) if o.reuse_teacher_head else fs[3]
        zs = self.teacher.fc(pool(final)) if o.reuse_teacher_head else self.student.fc(pool(final))
        label = evidential_ce(self.alpha(zs), labels)
        main = self.output_loss(zs, zt)
        zero = zs.new_zeros((), dtype=torch.float64)
        app = feature_func = teacher_output = student_output = zero
        selected = []
        mixed = []
        if o.use_fcfd:
            adapted = {k: self.s2t[str(k)](fs[k]) for k in (1, 2)}
            adapted[3] = final if o.reuse_teacher_head else self.s2t["3"](fs[3])
            # L_app：三个 stage 的元素平均 MSE，加最终 ReLU/GAP 表示的 MSE。
            app = sum(F.mse_loss(adapted[k], ft[k].detach()).double() for k in (1, 2, 3))
            app = app + F.mse_loss(pool(adapted[3]), pool(ft[3])).double()
            selected = self.sample_paths() if paths is None else paths
            if len(set(selected)) != len(selected) or any(p not in CANDIDATES for p in selected):
                raise ValueError("invalid mixed paths")
            for direction, k in selected:
                target = self.teacher if direction == "s2t" else self.student
                x = adapted[k] if direction == "s2t" else self.t2s[str(k)](ft[k].detach())
                for stage in range(k + 1, 4):
                    module = getattr(target, f"layer{stage}")
                    with self.mixed_bn(module, direction, k, stage):
                        x, _ = module(x)
                    # L_func 的中间特征监督来自正常教师；反向交互由辅助输出监督。
                    if direction == "s2t":
                        feature_func = feature_func + F.mse_loss(x, ft[stage].detach()).double()
                        if stage == 3:
                            feature_func = feature_func + F.mse_loss(pool(x), pool(ft[3])).double()
                if direction == "s2t":
                    zm = self.teacher.fc(pool(x))
                    teacher_output = teacher_output + self.output_loss(zm, zt)
                else:
                    if o.reuse_teacher_head:
                        with self.mixed_bn(self.final_adapter, direction, k, 4):
                            x = self.final_adapter(x)
                        zm = self.teacher.fc(pool(x))
                    else:
                        zm = self.student.fc(pool(x))
                    student_output = student_output + self.output_loss(zm, zt)
                mixed.append((direction, k, zm))
        wf = min(1.0, epoch / o.feature_warmup) if o.feature_warmup else 1.0
        wo = min(1.0, epoch / o.output_warmup) if o.output_warmup else 1.0
        # 完整 FEKD 使用 Dirichlet KL 监督正常预测与交互预测。
        terms = dict(
            loss_label=o.label_weight * label,
            loss_main=o.output_weight * wo * main,
            loss_appearance=o.feature_weight * wf * app,
            loss_function_feature=o.feature_weight * wf * feature_func,
            loss_teacher_path=o.auxiliary_weight * wo * teacher_output,
            loss_student_path=o.auxiliary_weight * wo * student_output,
        )
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite combined loss")
        stats = dict(
            raw_label=float(label.detach()),
            raw_main=float(main.detach()),
            raw_appearance=float(app.detach()),
            raw_function_feature=float(feature_func.detach()),
            raw_teacher_path=float(teacher_output.detach()),
            raw_student_path=float(student_output.detach()),
            feature_warmup_factor=wf,
            output_warmup_factor=wo,
            paths=selected,
            teacher_strength_mean=float(
                alpha_from_logits(zt, self.prior_t, o.evidence_mapping).sum(-1).mean()
            ),
            student_strength_mean=float(self.alpha(zs).detach().sum(-1).mean()),
        )
        return zs, loss, terms, stats

    def export_student(self):
        """导出推理组件和必要的结构信息，省去教师骨干与训练参数。"""
        return dict(
            format="fekd_student_v2",
            architecture="resnet8x4",
            dataset="cifar100",
            class_names=getattr(self, "class_names", None),
            inference=dict(
                factor=self.options.factor, reuse_teacher_head=True, evidence_mapping="exp"
            ),
            prior_raw=float(self.prior_s),
            prior_concentration=float(self.prior_s.exp()),
            student={k: v.detach().cpu() for k, v in self.student.state_dict().items()},
            adapter={k: v.detach().cpu() for k, v in self.final_adapter.state_dict().items()},
            classifier={k: v.detach().cpu() for k, v in self.teacher.fc.state_dict().items()},
        )


def build_model(teacher_checkpoint, teacher_factory, teacher_prior_raw, options):
    """读取外部教师工厂与权重；教师参数始终冻结。"""
    raw = float(teacher_prior_raw)
    if not math.isfinite(raw) or not 0 < math.exp(raw) < math.inf:
        raise ValueError("teacher prior must yield a finite positive exp(b)")
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("teacher checkpoint must contain a state dictionary")
    if "prior_raw" in checkpoint and not math.isclose(
        float(checkpoint["prior_raw"]), raw, rel_tol=1e-6, abs_tol=1e-8
    ):
        raise ValueError("teacher prior differs from checkpoint metadata")
    if "prior_concentration" in checkpoint and not math.isclose(
        float(checkpoint["prior_concentration"]), math.exp(raw), rel_tol=1e-6
    ):
        raise ValueError("teacher prior concentration differs from checkpoint metadata")
    if checkpoint.get("evidence_mapping", "exp") != "exp":
        raise ValueError("this FEKD implementation uses exponential evidence")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        teacher = build_external_model(teacher_factory)
        teacher.load_state_dict(state, strict=True)
        torch.manual_seed(options.seed)
        student = resnet8x4(num_classes=100)
    return DistillationModel(student, teacher, raw, options)


class StudentInference(nn.Module):
    """独立学生推理模型，兼容此前导出的 v1 学生文件。"""

    def __init__(self, checkpoint):
        super().__init__()
        if checkpoint.get("format") == "fekd_student_v2":
            settings = checkpoint["inference"]
        elif checkpoint.get("format") == "simkd_fcfd_dirichlet_student_v1":
            settings = checkpoint["options"]
        else:
            raise ValueError("expected an exported FEKD student checkpoint")
        self.class_names = checkpoint.get("class_names")
        self.reuse_teacher_head = settings["reuse_teacher_head"]
        self.evidence_mapping = settings["evidence_mapping"]
        factor = settings["factor"]
        if type(factor) is not int or not 1 <= factor <= 256:
            raise ValueError("invalid adapter factor")
        if self.evidence_mapping not in ("exp", "softplus"):
            raise ValueError("invalid student evidence mapping")
        self.student = resnet8x4(num_classes=100)
        if self.reuse_teacher_head:
            self.student.fc = nn.Identity()
            self.adapter = simkd_projector(factor=factor)
            self.adapter.load_state_dict(checkpoint["adapter"], strict=True)
            self.classifier = nn.Linear(256, 100)
            self.classifier.load_state_dict(checkpoint["classifier"], strict=True)
            self.classifier.requires_grad_(False)
        else:
            self.adapter = nn.Identity()
            self.classifier = nn.Identity()
        self.student.load_state_dict(checkpoint["student"], strict=True)
        self.register_buffer("prior_s", torch.tensor(checkpoint["prior_raw"], dtype=torch.float64))

    def forward(self, image):
        final = backbone_features(self.student, image)[-1]
        if self.reuse_teacher_head:
            return self.classifier(pool(self.adapter(final)))
        return self.student.fc(pool(final))

    def alpha(self, logits):
        return alpha_from_logits(logits, self.prior_s, self.evidence_mapping)
