"""允许从任意工作目录直接运行脚本，也支持 pip install -e .。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
