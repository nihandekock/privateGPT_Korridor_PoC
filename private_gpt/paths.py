from pathlib import Path

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.settings import settings


def _absolute_or_from_project_root(path: str) -> Path:
    if path.startswith("/"):
        return Path(path)
    return PROJECT_ROOT_PATH / path


models_path: Path = PROJECT_ROOT_PATH / "models"
models_cache_path: Path = models_path / "cache"
docs_path: Path = PROJECT_ROOT_PATH / "docs"
ui_path: Path = PROJECT_ROOT_PATH / "ui"
model: str = settings().local.llm_hf_model_file if settings().llm.mode == "local" else settings().openai.model
local_data_path: Path = _absolute_or_from_project_root(f"{settings().data.local_data_folder}/{settings().llm.mode}/{model}")

