"""main_pipeline.py - Entry point for the SMART end-to-end training and deployment pipeline."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import boto3
import mlflow
from botocore.client import Config

# Load variables and settings object from the central config module
from config.global_settings import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_S3_ENDPOINT_URL,
    MLFLOW_TRACKING_URI,
    PICSELLIA_API_TOKEN,
    PICSELLIA_DATASET_ID,
    settings,
)
from deploy_model import BentoMLDeployer
from download_dataset import download_picsellia_dataset
from prepare_dataset import create_splits, generate_yolo_config_file
from train_model import train_yolo_model

# Logging configuration (configured once here)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)

logging.info("===== STARTING MAIN PIPELINE VIA main_pipeline.py =====")


def ensure_minio_bucket_exists(bucket_name: str = "mlflow"):
    """Ensure the MinIO bucket exists; create it if it doesn't.

    Args:
        bucket_name (str): Name of the S3/MinIO bucket.
    """
    if not all([MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        logging.warning(
            "⚠️ S3 variables are not configured (MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY)."
        )
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )
    try:
        s3.head_bucket(Bucket=bucket_name)
        logging.info(f"✅ Bucket '{bucket_name}' already exists on MinIO.")
    except s3.exceptions.ClientError:
        logging.info(f"🚧 Creating bucket '{bucket_name}' on MinIO…")
        s3.create_bucket(Bucket=bucket_name)
        logging.info(f"✅ Bucket '{bucket_name}' successfully created.")


# MLflow configuration
if MLFLOW_TRACKING_URI:
    logging.info(f"[main_pipeline] Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            logging.info(
                f"[main_pipeline] Creating MLflow experiment: {MLFLOW_EXPERIMENT_NAME}"
            )
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logging.info(
            f"[main_pipeline] MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}"
        )
    except Exception as e:
        logging.error(
            f"[main_pipeline] Failed to set up MLflow experiment: {e}", exc_info=True
        )
else:
    logging.warning(
        "[main_pipeline] MLFLOW_TRACKING_URI not found. MLflow logging is disabled."
    )

# --- Pipeline Parameters ---
PICSELLIA_DATASET_VERSION = settings.get("PICSELLIA_DATASET_VERSION_NAME", "latest")
DOWNLOADED_DATA_ROOT_DIR = Path(settings.get("OUTPUT_DATA_DIR", "data"))

PREPARE_TRAIN_RATIO = settings.get("PREPARE_TRAIN_RATIO", 0.6)
PREPARE_VAL_RATIO = settings.get("PREPARE_VAL_RATIO", 0.2)
PREPARE_TEST_RATIO = settings.get("PREPARE_TEST_RATIO", 0.2)
PREPARE_SEED = settings.get("PREPARE_SEED", 42)
DEFAULT_CLASS_NAMES_FOR_YAML = settings.get("dataset.class_names", [])

# Training parameters
TRAIN_MODEL_TYPE = settings.get("TRAIN_MODEL_TYPE", "yolo11n.pt")
TRAIN_EPOCHS = settings.get("TRAIN_EPOCHS", 1)
TRAIN_BATCH_SIZE = settings.get("TRAIN_BATCH_SIZE", 4)
TRAIN_IMG_SIZE = settings.get("TRAIN_IMG_SIZE", 320)
TRAIN_PROJECT_NAME = settings.get("TRAIN_PROJECT_NAME", MLFLOW_EXPERIMENT_NAME)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAIN_RUN_NAME = f"{TRAIN_MODEL_TYPE}_e{TRAIN_EPOCHS}_b{TRAIN_BATCH_SIZE}_{timestamp}"

TRAIN_CLOSE_MOSAIC = settings.get("TRAIN_CLOSE_MOSAIC", 0)
TRAIN_SEED = settings.get("TRAIN_SEED", 42)

# GPU / performance settings
DEVICE = settings.get("DEVICE", "auto")
TRAIN_WORKERS = settings.get("TRAIN_WORKERS", 8)
TRAIN_AMP = settings.get("TRAIN_AMP", True)
TRAIN_CACHE = settings.get("TRAIN_CACHE", False)
TRAIN_PATIENCE = settings.get("TRAIN_PATIENCE", 50)
TRAIN_OPTIMIZER = settings.get("TRAIN_OPTIMIZER", "auto")
TRAIN_LR0 = settings.get("TRAIN_LR0", 0.01)


def run_complete_pipeline(deploy_to_cloud: bool = True, serve_locally: bool = False):
    """Run the full SMART pipeline from dataset download to BentoML deployment.

    Args:
        deploy_to_cloud (bool): Deploy to BentoML Cloud.
        serve_locally (bool): Serve locally after training.

    Returns:
        bool: True if pipeline succeeds, False otherwise.
    """
    logging.info("===== STARTING FULL SMART PIPELINE =====")

    logging.info("Training Configuration:")
    logging.info(f"  Device: {DEVICE}")
    logging.info(f"  Model: {TRAIN_MODEL_TYPE}")
    logging.info(f"  Epochs: {TRAIN_EPOCHS}")
    logging.info(f"  Batch size: {TRAIN_BATCH_SIZE}")
    logging.info(f"  Image size: {TRAIN_IMG_SIZE}")
    logging.info(f"  Workers: {TRAIN_WORKERS}")
    logging.info(f"  AMP: {TRAIN_AMP}")
    logging.info(f"  Cache: {TRAIN_CACHE}")

    ensure_minio_bucket_exists("mlflow")

    # Step 1: Download Dataset
    logging.info("--- Step 1: Downloading Dataset from Picsellia ---")
    try:
        downloaded_dataset_base_path_str = download_picsellia_dataset(
            api_token=PICSELLIA_API_TOKEN,
            dataset_id=PICSELLIA_DATASET_ID,
            version_identifier=PICSELLIA_DATASET_VERSION,
            output_root_dir=DOWNLOADED_DATA_ROOT_DIR,
            host=settings.get("PICSELLIA_HOST", "https://app.picsellia.com"),
        )
        if not downloaded_dataset_base_path_str:
            raise Exception("Dataset download returned an empty path.")
        raw_dataset_path = Path(downloaded_dataset_base_path_str)
        logging.info(f"Dataset successfully downloaded to: {raw_dataset_path}")
    except Exception as e:
        logging.error(f"Dataset download failed: {e}", exc_info=True)
        return False

    # Step 2: Prepare Dataset
    logging.info("--- Step 2: Preparing Dataset for YOLO ---")
    prepared_dataset_path = (
        raw_dataset_path.parent / f"{raw_dataset_path.name}_yolo_prepared"
    )
    data_yaml_path = prepared_dataset_path / "data.yaml"
    try:
        create_splits(
            base_path=raw_dataset_path,
            output_path=prepared_dataset_path,
            train_ratio=PREPARE_TRAIN_RATIO,
            val_ratio=PREPARE_VAL_RATIO,
            test_ratio=PREPARE_TEST_RATIO,
            seed=PREPARE_SEED,
        )
        generate_yolo_config_file(
            dataset_output_path=prepared_dataset_path,
            class_names=DEFAULT_CLASS_NAMES_FOR_YAML,
        )
        logging.info(f"Dataset prepared. YOLO config file: {data_yaml_path}")
    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}", exc_info=True)
        return False

    # Step 3: Train YOLO Model
    logging.info("--- Step 3: Training YOLO Model ---")
    try:
        best_model_path_str = train_yolo_model(
            data_config_path=str(data_yaml_path),
            model_type=TRAIN_MODEL_TYPE,
            project_name=TRAIN_PROJECT_NAME,
            run_name=TRAIN_RUN_NAME,
            epochs=TRAIN_EPOCHS,
            img_size=TRAIN_IMG_SIZE,
            batch_size=TRAIN_BATCH_SIZE,
            seed=TRAIN_SEED,
            close_mosaic_epochs=TRAIN_CLOSE_MOSAIC,
            device=DEVICE,
            workers=TRAIN_WORKERS,
            amp=TRAIN_AMP,
            cache=TRAIN_CACHE,
            patience=TRAIN_PATIENCE,
            optimizer=TRAIN_OPTIMIZER,
            lr0=TRAIN_LR0,
        )
        if not best_model_path_str:
            raise Exception("Training did not return a path to best.pt.")
        logging.info(f"Training completed. Best model path: {best_model_path_str}")
    except Exception as e:
        logging.error(f"Model training failed: {e}", exc_info=True)
        return False

    # Step 4: Deploy with BentoML
    logging.info("--- Step 4: Deploying Model with BentoML ---")
    try:
        deployer = BentoMLDeployer()

        deployed_model_path = deployer.download_best_model(best_model_path_str)
        logging.info(f"Model ready for deployment: {deployed_model_path}")

        bento_tag = deployer.build_bento()
        logging.info(f"BentoML service built: {bento_tag}")

        if deploy_to_cloud:
            success = deployer.deploy_to_cloud(bento_tag, "smart-object-detection")
            if success:
                logging.info("✅ Model successfully deployed to BentoML Cloud!")
            else:
                logging.warning(
                    "⚠️ Cloud deployment failed, but model is ready locally."
                )

        if serve_locally:
            logging.info("🚀 Starting local BentoML server...")
            deployer.serve_locally(bento_tag, port=3000)

    except Exception as e:
        logging.error(f"Deployment failed: {e}", exc_info=True)
        logging.info("Model is available, but deployment failed.")
        # Not considered a fatal error
        pass

    logging.info("===== SMART PIPELINE COMPLETED =====")
    logging.info(f"Raw dataset: {raw_dataset_path}")
    logging.info(f"Prepared YOLO dataset: {prepared_dataset_path}")
    logging.info(f"Best model: {best_model_path_str}")
    logging.info("✅ Training pipeline completed successfully!")
    logging.info("🔄 To run inference: python src/inference_pipeline.py")

    return True


def main():
    """CLI entry point for the SMART project pipeline."""
    parser = argparse.ArgumentParser(description="Full SMART Project Pipeline")
    parser.add_argument(
        "--no-cloud-deploy",
        action="store_true",
        help="Skip BentoML Cloud deployment",
    )
    parser.add_argument(
        "--serve-locally",
        action="store_true",
        help="Start local BentoML server after building",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        help="Override the default Picsellia dataset ID",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        help="Override the default dataset version",
    )

    args = parser.parse_args()

    # Override config if CLI args provided
    if args.dataset_id:
        global PICSELLIA_DATASET_ID
        PICSELLIA_DATASET_ID = args.dataset_id
        logging.info(f"Dataset ID overridden: {PICSELLIA_DATASET_ID}")

    if args.dataset_version:
        global PICSELLIA_DATASET_VERSION
        PICSELLIA_DATASET_VERSION = args.dataset_version
        logging.info(f"Dataset version overridden: {PICSELLIA_DATASET_VERSION}")

    try:
        success = run_complete_pipeline(
            deploy_to_cloud=not args.no_cloud_deploy, serve_locally=args.serve_locally
        )
        if success:
            logging.info("🎉 Pipeline executed successfully!")
            return 0
        else:
            logging.error("❌ Pipeline failed")
            return 1
    except KeyboardInterrupt:
        logging.info("⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"❌ Fatal pipeline error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
