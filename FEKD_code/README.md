# FEKD

Aligning Features and Evidence for Knowledge Distillation

## 目录

```text
FEKD_code/
├── fekd/
│   ├── models/
│   │   ├── resnet.py        # 学生骨干
│   │   └── distiller.py     # FEKD、特征交互与独立学生推理模型
│   ├── external.py         # 使用者外部教师的加载接口
│   ├── losses.py           # EDL 与 Dirichlet KL
│   ├── data.py             # 训练数据与图片预处理
│   ├── settings.py         # 使用者输入的参数校验
│   ├── engine.py           # 学生训练和导出
│   ├── cli.py              # 训练参数解析及学生加载
│   └── utils.py            # 保存、哈希与随机种子
├── scripts/
│   ├── _bootstrap.py       # 本地包导入辅助模块
│   ├── train_student.py    # 训练脚本
│   └── predict.py          # 推理脚本
├── licenses/
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## 配置环境

```bash
conda env create -f environment.yml
conda activate fekd
```

NVIDIA GPU 环境安装 CUDA 12.6 版 PyTorch：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

CPU 环境将第一条安装命令替换为：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

## 准备外部教师与数据

教师由 `--teacher-factory` 指定，格式为 `package.module:function` 或 `/path/to/model.py:function`。该函数接受 `num_classes=100`，并返回已由使用者实现的 `torch.nn.Module`。

外部教师应暴露 `conv1`、`bn1`、`layer1`、`layer2`、`layer3` 和 `fc`。三阶段输出通道依次为 64、128、256，空间尺寸对应 32、16、8；每个阶段返回 `(feature, preactivation)`。分类器 `fc` 将 256 维表示映射到 100 类。此接口用于将学生特征送入教师的后续阶段，包内不包含教师网络定义或教师训练程序。

`--teacher-checkpoint` 指向使用者自己的教师权重，支持纯 `state_dict`、包含 `model` 的字典或包含 `state_dict` 的字典。`module.` 前缀会在加载时去除。`--teacher-prior-raw` 由使用者输入教师的对数先验 b，对应每类先验为 exp(b)；已有文件若记录先验信息，程序会核对其一致性。教师证据映射应为指数函数。

CIFAR-100 目录结构为：

```text
DATA_ROOT/
└── cifar-100-python/
    ├── train
    ├── test
    └── meta
```

训练读取 `train`。首次运行时可在训练命令末尾添加 `--download`，由 torchvision 下载数据。训练采用随机裁剪和水平翻转，训练与推理使用相同的 CIFAR-100 归一化。

## 训练命令

下面的大写项均为占位符，请替换为自己的值。所有训练参数均需显式输入；`NESTEROV_FLAG` 替换为 `--nesterov` 或 `--no-nesterov`。`--milestones` 后填写所需的衰减轮次；使用空日程时保留该选项并省去其后的数字。


```powershell
python scripts/train_student.py `
  --teacher-factory "FACTORY_SPEC" `
  --teacher-checkpoint "TEACHER_CHECKPOINT" `
  --teacher-prior-raw TEACHER_PRIOR_RAW `
  --data-root "DATA_ROOT" `
  --output-dir "OUTPUT_DIR" `
  --device DEVICE `
  --epochs EPOCHS `
  --batch-size BATCH_SIZE `
  --learning-rate LEARNING_RATE `
  --momentum MOMENTUM `
  --weight-decay WEIGHT_DECAY `
  --milestones MILESTONE_1 MILESTONE_2 `
  --lr-decay LR_DECAY `
  NESTEROV_FLAG `
  --max-grad-norm MAX_GRAD_NORM `
  --workers WORKERS `
  --cpu-threads CPU_THREADS `
  --seed SEED `
  --student-prior-raw STUDENT_PRIOR_RAW `
  --factor FACTOR `
  --extra-paths EXTRA_PATHS `
  --label-weight LABEL_WEIGHT `
  --output-weight OUTPUT_WEIGHT `
  --feature-weight FEATURE_WEIGHT `
  --auxiliary-weight AUXILIARY_WEIGHT `
  --feature-warmup FEATURE_WARMUP `
  --output-warmup OUTPUT_WARMUP
```

Linux/macOS 使用相同参数，将行末反引号替换为反斜杠，或将命令合并为一行。

| 参数 | 含义 |
|---|---|
| `--device` | `cuda`、`cpu` 或 `auto` |
| `--epochs`、`--batch-size` | 训练轮数和每批样本数 |
| `--learning-rate`、`--momentum`、`--weight-decay` | SGD 参数 |
| `--milestones`、`--lr-decay` | 学习率衰减轮次和乘法因子 |
| `--nesterov` / `--no-nesterov` | 显式开启或关闭 Nesterov |
| `--max-grad-norm` | 梯度裁剪阈值；输入 0 表示关闭 |
| `--workers`、`--cpu-threads` | 数据加载进程数和计算线程数 |
| `--seed` | 随机种子 |
| `--student-prior-raw` | 学生对数先验 b，每类先验为 exp(b) |
| `--factor` | 最终适配器中间通道的缩减因子 |
| `--extra-paths` | 每批从四条交互路径中选择的数量 |
| `--label-weight` | EDL 标签监督权重 |
| `--output-weight` | 正常学生输出的 Dirichlet KL 权重 |
| `--feature-weight` | 直接特征对齐和功能匹配的共同权重 |
| `--auxiliary-weight` | 交互辅助输出的 Dirichlet KL 权重 |
| `--feature-warmup`、`--output-warmup` | 特征项和输出项的预热轮数；输入 0 表示直接使用完整权重 |

教师参数固定，训练更新学生骨干、最终适配器及特征映射模块。每轮将部署学生保存至 `OUTPUT_DIR/student.pth`，同名文件随训练更新。训练脚本读取训练集并记录训练损失及训练准确率。

输出目录还保存 `history.jsonl`、`status.json` 和 `parameters.json`。后者记录本次使用者实际输入的参数，供实验留档。部署文件包含学生骨干、最终适配器和复用分类器，不含完整教师骨干或训练交互路径。

## 推理命令

将占位符替换为自己的路径和推理设置：

```powershell
python scripts/predict.py `
  --checkpoint "OUTPUT_DIR/student.pth" `
  --input "IMAGE_OR_FOLDER" `
  --output "PREDICTION_JSON" `
  --device DEVICE `
  --top-k TOP_K `
  --cpu-threads CPU_THREADS
```

输入可为单张图片或图片目录，支持 JPG、PNG、BMP 和 WebP。图片转为 RGB 后缩放至 32×32。输出 JSON 包含类别索引、类别名称、Dirichlet 期望类别概率和总浓度 `alpha0`。推理使用部署学生文件即可。

## 查看参数说明

```bash
python scripts/train_student.py --help
python scripts/predict.py --help
```

第三方代码来源及许可证见 `THIRD_PARTY_NOTICES.md` 和 `licenses/`。
