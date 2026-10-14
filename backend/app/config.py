"""Настройки приложения: пути и переменные окружения (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
FRONTEND_DIR = ROOT / "frontend"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://traceon:traceon@localhost:5433/traceon")

# Необязательно: текстовое заключение через LLM (NVIDIA API, совместимый с OpenAI)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_URL = os.getenv("NVIDIA_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "10"))
