import os
from pathlib import Path
from dotenv import load_dotenv

# =====================================================================
# Project Base Directory
# =====================================================================

# Resolve the absolute path to the project root (parent of the 'configs' directory)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from a '.env' file located at the project root
# This makes variables defined in the .env file accessible via os.getenv()
load_dotenv(BASE_DIR / ".env")


# =====================================================================
# Directory Path Settings
# =====================================================================

# The root directory of the project
PROJECT_ROOT = BASE_DIR

# Directory for storing raw, intermediate, and processed datasets
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Directory for saving model checkpoints, weights, and configurations
MODELS_DIR = PROJECT_ROOT / "models"

# Directory for saving runtime logs and execution history
LOGS_DIR = PROJECT_ROOT / "logs"

# Directory for saving model outputs, predictions, and visualization plots
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# =====================================================================
# Directory Setup
# =====================================================================

# Automatically ensure that all foundational directories exist.
# This prevents potential 'FileNotFoundError' when logs or outputs are written.
for directory in [DATASETS_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Configuration Variables & Environment Placeholders
# =====================================================================

# Current environment context (e.g., 'development', 'staging', 'production')
# Defaults to 'development' if not specified in the environment or .env file
ENV = os.getenv("ENV", "development")

# Toggle debug mode (evaluates to True/False)
# Defaults to True for development convenience
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t", "y", "yes")

# --- Placeholders for future environment configurations ---
# (Secrets/API Keys should always be loaded from the environment, never hardcoded here)
# API_KEY_EXAMPLE = os.getenv("API_KEY_EXAMPLE", None)
