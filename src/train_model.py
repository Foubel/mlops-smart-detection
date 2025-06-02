"""Train a YOLO model using Ultralytics and log the results to MLflow."""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
from ultralytics import YOLO

from config.global_settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

# Logging configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)


def train_yolo_model(
    data_config_path: str,
    model_type: str,
    project_name: str,
    run_name: str,
    epochs: int,
    img_size: int,
    batch_size: int,
    seed: int,
    close_mosaic_epochs: int,
    device: str = "auto",
    workers: int = 8,
    amp: bool = True,
    cache: bool = False,
    patience: int = 50,
    optimizer: str = "auto",
    lr0: float = 0.01,
) -> str | None:
    """Train a YOLO model with the given configuration and log to MLflow.

    Args:
        data_config_path (str): Path to YOLO data.yaml config.
        model_type (str): YOLO model to use (e.g., 'yolo11n.pt').
        project_name (str): Folder to store YOLO training runs.
        run_name (str): Run name.
        epochs (int): Number of training epochs.
        img_size (int): Input image size.
        batch_size (int): Training batch size.
        seed (int): Seed for reproducibility.
        close_mosaic_epochs (int): Number of final epochs without mosaic.
        device (str): Device to use (e.g., 'cuda', 'cpu', 'auto').
        workers (int): Number of data loader workers.
        amp (bool): Use Automatic Mixed Precision.
        cache (bool): Cache images during training.
        patience (int): Early stopping patience.
        optimizer (str): Optimizer to use.
        lr0 (float): Initial learning rate.

    Returns:
        str | None: Path to best.pt if training succeeded, else None.
    """
    logging.info(f"Starting training with model {model_type}...")
    logging.info("Training configuration:")
    logging.info(f"  Data config: {data_config_path}")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Image size: {img_size}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Workers: {workers}")
    logging.info(f"  AMP: {amp}")
    logging.info(f"  Cache: {cache}")
    logging.info(f"  Seed: {seed}")
    logging.info(f"  Close mosaic: {close_mosaic_epochs}")
    logging.info(f"  Patience: {patience}")
    logging.info(f"  Optimizer: {optimizer}")
    logging.info(f"  LR0: {lr0}")

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logging.info(f"MLflow tracking URI set: {MLFLOW_TRACKING_URI}")

    try:
        exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if exp is None:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            logging.info(f"MLflow experiment created: {MLFLOW_EXPERIMENT_NAME}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logging.info(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
    except Exception as e:
        logging.error(f"Error configuring MLflow experiment: {e}")

    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        logging.info(f"MLflow run started: {run.info.run_id}")

        mlflow.log_params(
            {
                "model_type": model_type,
                "epochs": epochs,
                "img_size": img_size,
                "batch_size": batch_size,
                "device": device,
                "workers": workers,
                "amp": amp,
                "cache": cache,
                "seed": seed,
                "close_mosaic_epochs": close_mosaic_epochs,
                "patience": patience,
                "optimizer": optimizer,
                "lr0": lr0,
            }
        )

        model = YOLO(model_type)
        logging.info(f"Loaded model: {model_type}")

        try:
            model.train(
                data=data_config_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                seed=seed,
                close_mosaic=close_mosaic_epochs,
                project=project_name,
                name=run_name,
                device=device,
                workers=workers,
                amp=amp,
                cache=cache,
                patience=patience,
                optimizer=optimizer,
                lr0=lr0,
                deterministic=True,
                verbose=True,
            )
            logging.info("Training completed successfully.")
        except Exception as e:
            logging.critical(f"Training failed: {e}", exc_info=True)
            return None

        save_dir = Path(project_name) / run_name
        best_model = save_dir / "weights" / "best.pt"

        if not best_model.is_file():
            logging.error(f"best.pt not found: {best_model}")
            fallback_paths = [
                save_dir / "best.pt",
                Path(project_name) / "weights" / "best.pt",
                Path("runs") / "detect" / run_name / "weights" / "best.pt",
            ]
            for path in fallback_paths:
                if path.exists():
                    best_model = path
                    logging.info(f"Found model at: {best_model}")
                    break
            else:
                logging.error("Could not find best.pt anywhere.")
                return None

        mlflow.log_artifact(str(best_model), artifact_path="model")
        logging.info(f"Model logged to MLflow: {best_model}")

        try:
            model_uri = f"runs:/{run.info.run_id}/model/best.pt"
            model_name = "SMART_YOLO_Object_Detection"
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

            logging.info(
                f"Model registered in Model Registry: {model_name}, version: {model_version.version}"
            )

            try:
                client = mlflow.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=f"YOLO model ({model_type}) trained on SMART dataset with {epochs} epochs on {device}",
                )
                logging.info("Model description updated.")
            except Exception as desc_error:
                logging.warning(f"Could not update model description: {desc_error}")

            try:
                client.transition_model_version_stage(
                    name=model_name, version=model_version.version, stage="Production"
                )
                logging.info(
                    f"Model version {model_version.version} marked as 'Production'"
                )
            except Exception as stage_error:
                logging.warning(
                    f"Could not transition model to Production: {stage_error}"
                )

        except Exception as e:
            logging.error(f"Model registration failed: {e}")
            logging.info("Training succeeded despite Model Registry error.")

        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd

            try:
                df = pd.read_csv(results_csv)
                last_row = df.iloc[-1]
                for col in df.columns:
                    try:
                        value = float(last_row[col])
                        clean_name = (
                            col.replace("(", "_").replace(")", "").replace("/", "_")
                        )
                        mlflow.log_metric(clean_name, value)
                    except (ValueError, TypeError):
                        continue
                logging.info("YOLO metrics logged to MLflow from results.csv")
            except Exception as e:
                logging.warning(f"Could not log metrics: {e}")
        else:
            logging.warning("results.csv not found — no metrics logged.")

        req_file = Path("requirements.txt")
        if req_file.exists():
            mlflow.log_artifact(str(req_file), artifact_path="environment")

        return str(best_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model with GPU support.")
    parser.add_argument(
        "--data_config", type=str, required=True, help="Path to data.yaml."
    )
    parser.add_argument(
        "--model_type", type=str, default="yolo11n.pt", help="YOLO model to use."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--img_size", type=int, default=640, help="Image size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--close_mosaic_epochs",
        type=int,
        default=0,
        help="Disable mosaic on last N epochs.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="SMART_YOLO_Training",
        help="YOLO project folder.",
    )
    parser.add_argument(
        "--run_name", type=str, default="run_default", help="Training run name."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Training device (auto, cpu, cuda)."
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of workers.")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP.")
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false", help="Disable AMP."
    )
    parser.add_argument(
        "--cache", action="store_true", help="Cache images during training."
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--lr0", type=float, default=0.01, help="Initial learning rate."
    )

    args = parser.parse_args()
    data_yaml = Path(args.data_config)
    if not data_yaml.is_file():
        logging.error(f"data.yaml not found: {args.data_config}")
        exit(1)

    best_model_path = train_yolo_model(
        data_config_path=str(data_yaml),
        model_type=args.model_type,
        project_name=args.project_name,
        run_name=args.run_name,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        close_mosaic_epochs=args.close_mosaic_epochs,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        cache=args.cache,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
    )

    if best_model_path:
        logging.info("✅ Training completed successfully!")
        logging.info(f"Best model path: {best_model_path}")
        exit(0)
    else:
        logging.error("❌ Training failed")
        exit(1)
