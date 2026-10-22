"""加载使用者提供的外部模型工厂；包内不定义教师网络。"""

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from torch import nn


def build_external_model(specification):
    """支持 package.module:function 和 /path/model.py:function。"""
    try:
        module_name, factory_name = specification.rsplit(":", 1)
    except ValueError as error:
        raise ValueError("teacher factory must be module:function or file.py:function") from error
    if not module_name or not factory_name:
        raise ValueError("teacher factory module and function are required")
    module_path = Path(module_name)
    inserted_path = None
    try:
        if module_path.suffix == ".py":
            module_path = module_path.resolve(strict=True)
            name = "_fekd_external_" + hashlib.sha256(str(module_path).encode()).hexdigest()[:16]
            specification = importlib.util.spec_from_file_location(name, module_path)
            module = importlib.util.module_from_spec(specification)
            sys.modules[name] = module
            inserted_path = str(module_path.parent)
            sys.path.insert(0, inserted_path)
            specification.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        model = getattr(module, factory_name)(num_classes=100)
    finally:
        if inserted_path is not None:
            sys.path.remove(inserted_path)
    if not isinstance(model, nn.Module):
        raise TypeError("external factory must return a torch.nn.Module")
    for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "fc"):
        if not isinstance(getattr(model, name, None), nn.Module):
            raise ValueError(f"external teacher must expose module {name}")
    if not isinstance(model.fc, nn.Linear) or (model.fc.in_features, model.fc.out_features) != (
        256,
        100,
    ):
        raise ValueError("external teacher classifier must map 256 features to 100 classes")
    return model
