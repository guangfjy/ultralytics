# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.4.8"

# --------------------------------------------------------------------
# mtml fork 版本增强（带本地 git 信息 + dirty 标记 + 缓存）
# --------------------------------------------------------------------
import sys  # 使用 sys 模块做简单缓存（进程内只计算一次）
if not hasattr(sys, '_mtml_version_cache'):
    try:
        import subprocess
        from pathlib import Path

        # 执行 git describe，只匹配上游风格的 tag（v8.x、v9.x 等）
        cmd = [
            "git",
            "describe",
            "--tags",
            "--dirty",
            "--always",
            "--match", "v8.*",  # 匹配 v8.3.248、v8.4.0 等官方 tag
        ]
        git_desc = subprocess.check_output(
            cmd,
            cwd=Path(__file__).resolve().parents[1],  # 项目根目录
            stderr=subprocess.DEVNULL,
            text=True,  # 返回字符串，不需要decode()
        ).strip()
    except Exception:
        git_desc = "nogit"

    # 最终版本格式：上游版本 + mtml 本地标识
    sys._mtml_version_cache = f"{__version__}+mtml.{git_desc}"

__version__ = sys._mtml_version_cache
# --------------------------------------------------------------------

import importlib
import os
from typing import TYPE_CHECKING

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from ultralytics.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
