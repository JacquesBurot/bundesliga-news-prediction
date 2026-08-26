"""Build and run structured local-LLM news annotations."""

from .config import DEFAULT_ANNOTATION_CONFIG_PATH, load_annotation_config
from .ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
)
from .runner import annotate_news_tasks
from .selection import (
    NEWS_ANNOTATION_PILOT_SELECTION_VERSION,
    select_stratified_annotation_tasks,
)
from .tasks import build_news_annotation_tasks


__all__ = (
    "DEFAULT_ANNOTATION_CONFIG_PATH",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_NUM_CTX",
    "NEWS_ANNOTATION_PILOT_SELECTION_VERSION",
    "annotate_news_tasks",
    "build_news_annotation_tasks",
    "load_annotation_config",
    "select_stratified_annotation_tasks",
)
