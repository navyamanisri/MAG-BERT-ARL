"""
MAG-BERT-ARL Backend Constants

This file defines project-wide constants, placeholder model names,
and directory structures. To maintain consistency, we align path constants
with the central settings defined in configs/settings.py.
"""

from pathlib import Path
import sys

# Import core configurations if available to ensure alignment across packages
try:
    # Adding the project root to sys.path if not present to allow importing config
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from configs.settings import PROJECT_ROOT, DATASETS_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR, ENV, DEBUG
except ImportError:
    # Fallback to local definitions if settings.py cannot be imported
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATASETS_DIR = PROJECT_ROOT / "datasets"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    ENV = "development"
    DEBUG = True

# Project Identification
PROJECT_NAME = "MAG-BERT-ARL"
PROJECT_DESCRIPTION = (
    "Fair Automated Video Interview Assessment using a Multimodal Adaptation Gate "
    "and Adversarial Representation Learning."
)
VERSION = "0.1.0"

# Backend specific directories
BACKEND_DIR = PROJECT_ROOT / "backend"
API_DIR = BACKEND_DIR / "api"
CORE_DIR = BACKEND_DIR / "core"
SERVICES_DIR = BACKEND_DIR / "services"
UTILS_DIR = BACKEND_DIR / "utils"
PIPELINES_DIR = BACKEND_DIR / "pipelines"

# Placeholder / Default Model Names (Hugging Face / Pre-trained Hub Keys)
# Verbal (Text) Modality Base Model
DEFAULT_TEXT_MODEL = "bert-base-uncased"
ALTERNATIVE_TEXT_MODEL = "roberta-base"

# Vocal (Audio) Modality Base Model
DEFAULT_AUDIO_MODEL = "facebook/wav2vec2-base-960h"
ALTERNATIVE_AUDIO_MODEL = "superb/wav2vec2-base-superb-er"  # Emotion Recognition

# Visual (Video) Modality Config/Model placeholders
DEFAULT_VISUAL_MODEL = "resnet3d-50"
FACEMESH_MAX_FACES = 1

# Pipeline Configs & Defaults
DEFAULT_AUDIO_SAMPLE_RATE = 16000  # 16kHz for Wav2Vec2 and openSMILE
DEFAULT_VIDEO_FPS = 30
MAX_VIDEO_DURATION_SECONDS = 300  # 5 minutes maximum for interview answers

# Competency & Fairness Targets
# Standard competencies assessed in structured interviews
ASSESSED_COMPETENCIES = [
    "communication_skills",
    "problem_solving",
    "leadership",
    "technical_aptitude",
    "professionalism"
]

# Demographic Groups targeted for fairness auditing & ARL de-biasing
SENSITIVE_ATTRIBUTES = [
    "gender",
    "age_group",
    "ethnicity"
]
