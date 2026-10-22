# FEKD

Aligning Features and Evidence for Knowledge Distillation

## Directory Structure

```text
FEKD_code/
├── fekd/
│   ├── models/
│   │   ├── resnet.py        # Student backbone
│   │   └── distiller.py     # FEKD, feature interactions, and standalone student inference
│   ├── external.py         # Interface for loading an external teacher
│   ├── losses.py           # EDL and Dirichlet KL losses
│   ├── data.py             # Training data and image preprocessing
│   ├── settings.py         # Validation of user-specified parameters
│   ├── engine.py           # Student training and export
│   ├── cli.py              # Training argument parsing and student loading
│   └── utils.py            # File saving, hashing, and random seeds
├── scripts/
│   ├── _bootstrap.py       # Helper for importing the local package
│   ├── train_student.py    # Training script
│   └── predict.py          # Inference script
├── licenses/
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## Environment Setup

Run the following commands from the `FEKD_code` directory:

```bash
conda env create -f environment.yml
conda activate fekd
```

For NVIDIA GPUs, install the CUDA 12.6 build of PyTorch:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For a CPU environment, replace the first installation command with:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

## External Teacher and Data Preparation

Specify the teacher using `--teacher-factory` in the format `package.module:function` or `/path/to/model.py:function`. The function must accept `num_classes=100` and return a user-implemented `torch.nn.Module`.

The external teacher must expose `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, and `fc`. The three stages must have 64, 128, and 256 output channels, with spatial resolutions of 32×32, 16×16, and 8×8, respectively. Each stage must return `(feature, preactivation)`. The classifier `fc` maps a 256-dimensional representation to 100 classes. This interface allows student features to pass through subsequent teacher stages. Teacher network definitions and teacher training code are not included in this package.

Use `--teacher-checkpoint` to specify your teacher weights. Supported formats include a plain `state_dict`, a dictionary containing a `model` entry, or a dictionary containing a `state_dict` entry. The `module.` prefix is removed during loading. Specify the teacher's log prior b using `--teacher-prior-raw`, where the prior for each class is exp(b). If the checkpoint contains prior metadata, the provided value is checked for consistency. The teacher must use an exponential evidence mapping.

Organize CIFAR-100 as follows:

```text
DATA_ROOT/
└── cifar-100-python/
    ├── train
    ├── test
    └── meta
```

## Training

Replace all uppercase placeholders below with your own values. All training parameters must be provided explicitly. Replace `NESTEROV_FLAG` with either `--nesterov` or `--no-nesterov`. List the learning-rate decay epochs after `--milestones`. To use an empty schedule, include `--milestones` without any numbers after it.

The following command uses PowerShell syntax, with backticks for line continuation:

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

On Linux/macOS, use the same arguments and replace the trailing backticks with backslashes, or combine the command into a single line.

| Argument | Description |
|---|---|
| `--device` | `cuda`, `cpu`, or `auto` |
| `--epochs`, `--batch-size` | Number of training epochs and samples per batch |
| `--learning-rate`, `--momentum`, `--weight-decay` | SGD parameters |
| `--milestones`, `--lr-decay` | Learning-rate decay epochs and multiplicative decay factor |
| `--nesterov` / `--no-nesterov` | Explicitly enable or disable Nesterov momentum |
| `--max-grad-norm` | Gradient clipping threshold; set to 0 to disable clipping |
| `--workers`, `--cpu-threads` | Number of data-loading worker processes and computation threads |
| `--seed` | Random seed |
| `--student-prior-raw` | Student log prior b; the prior for each class is exp(b) |
| `--factor` | Channel reduction factor for the intermediate layers of the final adapter |
| `--extra-paths` | Number of paths selected from the four feature interaction paths per batch |
| `--label-weight` | Weight of the EDL label loss |
| `--output-weight` | Weight of the Dirichlet KL loss for the standard student output |
| `--feature-weight` | Shared weight for direct feature alignment and functional matching |
| `--auxiliary-weight` | Weight of the Dirichlet KL loss for auxiliary interaction outputs |
| `--feature-warmup`, `--output-warmup` | Warmup epochs for feature and output losses; set to 0 to apply the full weights immediately |

The teacher parameters remain frozen during training. The student backbone, final adapter, and feature mapping modules are updated. The deployable student is saved to `OUTPUT_DIR/student.pth` after each epoch, replacing the previous file. The training script reads the training set and records the training loss and accuracy.

The output directory also contains `history.jsonl`, `status.json`, and `parameters.json`. The last file records the parameters supplied for the run. The exported student contains the student backbone, final adapter, and reused classifier, without the full teacher backbone or training interaction paths.

## Inference

Replace the placeholders with your paths and inference settings:

```powershell
python scripts/predict.py `
  --checkpoint "OUTPUT_DIR/student.pth" `
  --input "IMAGE_OR_FOLDER" `
  --output "PREDICTION_JSON" `
  --device DEVICE `
  --top-k TOP_K `
  --cpu-threads CPU_THREADS
```

The input can be a single image or an image directory. Supported formats include JPG, PNG, BMP, and WebP. Images are converted to RGB and resized to 32×32. The output JSON contains class indices, class names, expected class probabilities under the Dirichlet distribution, and the total concentration `alpha0`. Inference requires only the exported student checkpoint.

## Argument Reference

```bash
python scripts/train_student.py --help
python scripts/predict.py --help
```

See `THIRD_PARTY_NOTICES.md` and `licenses/` for third-party code attribution and licenses.

