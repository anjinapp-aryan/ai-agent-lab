"""Select and load the environment file for the current APP_ENV (local | dev | prod).

Order (first definition wins, real process env always wins over files):
  1. .env.<APP_ENV>   e.g. .env.local, .env.dev, .env.prod   (git-ignored; copy from *.example)
  2. .env             optional shared base values (git-ignored)
APP_ENV comes from the process environment; default is "local".
"""
import os
from pathlib import Path

from dotenv import load_dotenv

VALID_ENVS = ("local", "dev", "prod")
REPO_ROOT = Path(__file__).resolve().parents[3]


def current_env() -> str:
    env = os.environ.get("APP_ENV", "local").strip().lower()
    if env not in VALID_ENVS:
        raise RuntimeError(f"APP_ENV={env!r} invalid; use one of {VALID_ENVS}")
    return env


def load_env() -> str:
    """Load env files for APP_ENV; returns the environment name."""
    env = current_env()
    load_dotenv(REPO_ROOT / f".env.{env}", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    os.environ["APP_ENV"] = env
    return env
