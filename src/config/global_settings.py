"""Global configuration settings for the project."""

import logging
import os
from pathlib import Path

import dotenv
import toml

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

# Config file paths
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent.parent
ENV_FILE_PATH = CONFIG_DIR / ".env"
SETTINGS_TOML_FILE_PATH = PROJECT_ROOT / "settings.toml"

# Load .env
if ENV_FILE_PATH.exists():
    dotenv.load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)
    logging.info(f"[global_settings] .env file loaded from {ENV_FILE_PATH}")
else:
    logging.warning(f"[global_settings] .env file not found at {ENV_FILE_PATH}")

# Load settings.toml
config_from_toml = {}
if SETTINGS_TOML_FILE_PATH.exists():
    with open(SETTINGS_TOML_FILE_PATH, "r", encoding="utf-8") as f:
        config_from_toml = toml.load(f)
    logging.info(
        f"[global_settings] settings.toml loaded from {SETTINGS_TOML_FILE_PATH}"
    )

# Main variables (priority to .env)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SMART_Object_Detection")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

PICSELLIA_API_TOKEN = os.getenv("PICSELLIA_API_TOKEN")
PICSELLIA_DATASET_ID = os.getenv("PICSELLIA_DATASET_ID")
PICSELLIA_DATASET_VERSION_NAME = os.getenv("PICSELLIA_DATASET_VERSION_NAME", "latest")
PICSELLIA_HOST = os.getenv("PICSELLIA_HOST", "https://app.picsellia.com")

# Class names from settings.toml
DEFAULT_CLASS_NAMES_FOR_YAML = config_from_toml.get("dataset", {}).get(
    "class_names", []
)


# Utility functions
def str_to_bool(value, default=False):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return default


def str_to_int(value, default=0):
    """Convert string to integer with fallback."""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def str_to_float(value, default=0.0):
    """Convert string to float with fallback."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def detect_device():
    """Automatically detect the best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            device = "0"  # First GPU (YOLO prefers "0" over "cuda")
            logging.info(f"[global_settings] CUDA detected, using device: {device}")
            return device
        else:
            logging.info("[global_settings] CUDA not available, using CPU")
            return "cpu"
    except ImportError:
        logging.warning("[global_settings] PyTorch not installed, using CPU by default")
        return "cpu"


def get_device_setting():
    """Get the device from env or detect automatically."""
    device_env = os.getenv("DEVICE", "auto")
    if device_env == "auto":
        return detect_device()
    elif device_env == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "0"
            else:
                logging.warning(
                    "[global_settings] DEVICE=cuda requested but CUDA not available, falling back to CPU"
                )
                return "cpu"
        except ImportError:
            logging.warning(
                "[global_settings] PyTorch not available, falling back to CPU"
            )
            return "cpu"
    return device_env


# Settings object
class SimpleSettings:
    """A simple settings class to hold configuration values."""

    def __init__(self, **kwargs):
        """Initialize settings with keyword arguments."""
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        """Get a setting value by key, supporting nested keys with dot notation."""
        if "." in key:
            keys = key.split(".")
            value = self.__dict__
            for k in keys:
                value = value.get(k, {}) if isinstance(value, dict) else {}
            return value if value != {} else default
        return self.__dict__.get(key, default)


settings = SimpleSettings(
    # MLflow
    MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME=MLFLOW_EXPERIMENT_NAME,
    # Picsellia
    PICSELLIA_API_TOKEN=PICSELLIA_API_TOKEN,
    PICSELLIA_DATASET_ID=PICSELLIA_DATASET_ID,
    PICSELLIA_DATASET_VERSION_NAME=PICSELLIA_DATASET_VERSION_NAME,
    PICSELLIA_HOST=PICSELLIA_HOST,
    # Paths and organization
    OUTPUT_DATA_DIR=os.getenv("OUTPUT_DATA_DIR", "data"),
    # Data preparation
    PREPARE_TRAIN_RATIO=str_to_float(os.getenv("PREPARE_TRAIN_RATIO"), 0.6),
    PREPARE_VAL_RATIO=str_to_float(os.getenv("PREPARE_VAL_RATIO"), 0.2),
    PREPARE_TEST_RATIO=str_to_float(os.getenv("PREPARE_TEST_RATIO"), 0.2),
    PREPARE_SEED=str_to_int(os.getenv("PREPARE_SEED"), 42),
    # Basic training config
    TRAIN_MODEL_TYPE=os.getenv("TRAIN_MODEL_TYPE", "yolo11n.pt"),
    TRAIN_EPOCHS=str_to_int(os.getenv("TRAIN_EPOCHS"), 1),
    TRAIN_BATCH_SIZE=str_to_int(os.getenv("TRAIN_BATCH_SIZE"), 4),
    TRAIN_IMG_SIZE=str_to_int(os.getenv("TRAIN_IMG_SIZE"), 320),
    TRAIN_PROJECT_NAME=os.getenv("TRAIN_PROJECT_NAME", MLFLOW_EXPERIMENT_NAME),
    TRAIN_RUN_NAME=os.getenv("TRAIN_RUN_NAME", "run_from_pipeline"),
    TRAIN_CLOSE_MOSAIC=str_to_int(os.getenv("TRAIN_CLOSE_MOSAIC"), 0),
    TRAIN_SEED=str_to_int(os.getenv("TRAIN_SEED"), 42),
    # GPU / performance config
    DEVICE=get_device_setting(),
    TRAIN_WORKERS=str_to_int(os.getenv("TRAIN_WORKERS"), 8),
    TRAIN_AMP=str_to_bool(os.getenv("TRAIN_AMP"), True),
    TRAIN_CACHE=str_to_bool(os.getenv("TRAIN_CACHE"), False),
    TRAIN_PATIENCE=str_to_int(os.getenv("TRAIN_PATIENCE"), 50),
    TRAIN_OPTIMIZER=os.getenv("TRAIN_OPTIMIZER", "auto"),
    TRAIN_LR0=str_to_float(os.getenv("TRAIN_LR0"), 0.01),
    # Dataset from TOML
    dataset=config_from_toml.get("dataset", {}),
)


# Config validation
def validate_config():
    """Validate configuration and display warnings if needed."""
    warnings = []
    errors = []

    if not PICSELLIA_API_TOKEN:
        errors.append("Missing PICSELLIA_API_TOKEN")

    if not PICSELLIA_DATASET_ID:
        errors.append("Missing PICSELLIA_DATASET_ID")

    total_ratio = (
        settings.PREPARE_TRAIN_RATIO
        + settings.PREPARE_VAL_RATIO
        + settings.PREPARE_TEST_RATIO
    )
    if abs(total_ratio - 1.0) > 0.001:
        warnings.append(
            f"Sum of train/val/test ratios ({total_ratio}) is not equal to 1.0"
        )

    device = settings.DEVICE.lower()
    if device in ["cuda", "0", "1", "2", "3"]:
        try:
            import torch

            if not torch.cuda.is_available():
                warnings.append(
                    f"DEVICE={settings.DEVICE} set but CUDA is not available"
                )
        except ImportError:
            warnings.append("PyTorch not installed, cannot check CUDA")

    if errors:
        for error in errors:
            logging.error(f"❌ Config: {error}")
        raise ValueError(f"Invalid config: {', '.join(errors)}")

    if warnings:
        for warning in warnings:
            logging.warning(f"⚠️ Config: {warning}")

    logging.info("✅ Configuration validated")


# Auto-validate on import
try:
    validate_config()
except Exception as e:
    logging.error(f"Configuration validation error: {e}")

# Config summary
logging.info("📋 Configuration summary:")
logging.info(f"  Device: {settings.DEVICE}")
logging.info(f"  Model: {settings.TRAIN_MODEL_TYPE}")
logging.info(f"  Epochs: {settings.TRAIN_EPOCHS}")
logging.info(f"  Batch size: {settings.TRAIN_BATCH_SIZE}")
logging.info(f"  MLflow URI: {MLFLOW_TRACKING_URI or 'Not set'}")
logging.info(f"  Dataset classes: {len(DEFAULT_CLASS_NAMES_FOR_YAML)} classes defined")
