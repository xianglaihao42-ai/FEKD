# Third-party attribution

`fekd/models/resnet.py` contains the CIFAR student backbone adapted from MDistiller / RepDistiller:

- https://github.com/megvii-research/mdistiller
- https://github.com/HobbitLong/RepDistiller

The corresponding MIT license is preserved in `licenses/MDistiller_MIT.txt`.

Method references: SimKD (Chen et al., CVPR 2022), FCFD (Liu et al., ICLR 2023), and EKD (Xiang et al., ICCV 2025).

This package provides FEKD student training and inference. The teacher implementation and all model checkpoints are supplied separately by the user and are not distributed with this package.
