"""配置加载器

负责从 YAML 文件中加载应用配置和策略配置，
在模块导入时即完成加载，供其他模块直接引用 CONFIG / STRATEGY。
"""
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).parent.resolve()
_CONFIG_PATH = _PROJECT_ROOT / "resources/config.yml"
_STRATEGY_PATH = _PROJECT_ROOT / "resources/strategy.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {path}\n"
            f"请确保 {path.name} 位于 {path.parent} 目录。"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


CONFIG: dict[str, Any] = _load_yaml(_CONFIG_PATH)
STRATEGY: dict[str, Any] = _load_yaml(_STRATEGY_PATH)
