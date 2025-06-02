"""Deployment module for SMART_YOLO_Object_Detection using BentoML and MLflow."""

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow

from config.global_settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class BentoMLDeployer:
    """Handles downloading, building, and deploying the YOLO model using BentoML."""

    def __init__(self):
        """Initialize the deployer and configure the MLflow tracking URI."""
        self.model_name = "SMART_YOLO_Object_Detection"

        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def download_best_model(self, local_model_path: str = None) -> str:
        """Download the best model either from MLflow or from a local path.

        Args:
            local_model_path (str, optional): Path to a local model file.

        Returns:
            str: Path to the downloaded or copied model.
        """
        target_path = "deployed_model.pt"

        if local_model_path and Path(local_model_path).exists():
            logging.info(f"Using local model: {local_model_path}")
            shutil.copy2(local_model_path, target_path)
            return target_path

        try:
            logging.info("Downloading model from MLflow Model Registry...")

            client = mlflow.MlflowClient()
            latest_versions = client.get_latest_versions(
                name=self.model_name, stages=["Production"]
            )

            if not latest_versions:
                latest_versions = client.get_latest_versions(name=self.model_name)

            if not latest_versions:
                raise Exception(f"No version found for {self.model_name}")

            model_version = latest_versions[0]
            logging.info(f"Downloading version {model_version.version}")

            model_uri = f"models:/{self.model_name}/{model_version.version}"

            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"{model_uri}/best.pt", dst_path=temp_dir
                )
                shutil.copy2(downloaded_path, target_path)

            logging.info(f"Model downloaded to: {target_path}")
            return target_path

        except Exception as e:
            logging.error(f"Error downloading from MLflow: {e}")

            try:
                logging.info("Trying to fetch from the latest run's artifacts...")
                experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1,
                    )

                    if not runs.empty:
                        latest_run_id = runs.iloc[0].run_id
                        artifact_uri = f"runs:/{latest_run_id}/model/best.pt"

                        with tempfile.TemporaryDirectory() as temp_dir:
                            downloaded_path = mlflow.artifacts.download_artifacts(
                                artifact_uri=artifact_uri, dst_path=temp_dir
                            )
                            shutil.copy2(downloaded_path, target_path)

                        logging.info("Model retrieved from latest run artifacts")
                        return target_path

            except Exception as e2:
                logging.error(f"Failed to retrieve the model: {e2}")
                raise Exception("No available model for deployment")

        return target_path

    def build_bento(self, service_name: str = "yolo_service") -> str:
        """Build a BentoML service from the current working directory.

        Args:
            service_name (str): Name prefix of the BentoML service tag.

        Returns:
            str: Tag of the built bento.
        """
        logging.info("Building BentoML service...")
        subprocess.run(["bentoml", "build"], check=True)

        raw_json = subprocess.run(
            ["bentoml", "list", "--output", "json"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout
        bentos = json.loads(raw_json)

        pattern = re.compile(rf"^{re.escape(service_name)}:")
        candidates = [b for b in bentos if pattern.match(b["tag"])]

        if not candidates:
            raise RuntimeError(
                f"No bento found with tag starting with '{service_name}:'"
            )

        tag = max(candidates, key=lambda b: b.get("creation_time", ""))["tag"]
        logging.info(f"Bento built: {tag}")
        return tag

    def check_bentoml_cloud_auth(self) -> bool:
        """Check if the user is authenticated with BentoML Cloud.

        Returns:
            bool: True if authenticated, False otherwise.
        """
        try:
            subprocess.run(
                ["bentoml", "cloud", "current-context"],
                check=True,
            )
            logging.info("✅ Authenticated with BentoML Cloud")
            return True
        except subprocess.CalledProcessError:
            logging.error("❌ Not authenticated with BentoML Cloud")
            logging.info("💡 Run: bentoml cloud login")
            return False

    def deploy_to_cloud(
        self, bento_tag: str, deployment_name: str | None = None
    ) -> bool:
        """Deploy the given bento to BentoML Cloud.

        Args:
            bento_tag (str): The BentoML tag to deploy.
            deployment_name (str, optional): Deployment name on the cloud.

        Returns:
            bool: True if deployment succeeded, False otherwise.
        """
        deployment_name = deployment_name or "smart-object-detection"
        logging.info(f"🚀 Deploying to BentoML Cloud: {deployment_name}")

        if not self.check_bentoml_cloud_auth():
            return False

        try:
            subprocess.run(
                ["bentoml", "deploy", "--name", deployment_name, bento_tag],
                check=True,
            )
            logging.info("✅ Deployment successful!")

            list_result = subprocess.run(
                ["bentoml", "deployment", "list"],
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
            if deployment_name in list_result.stdout:
                logging.info("🔗 Deployment found — check BentoCloud for the exact URL")

            return True

        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Deployment error: {e}")
            return False

    def serve_locally(self, bento_tag: str, port: int = 3000):
        """Start a local BentoML server to serve the bento.

        Args:
            bento_tag (str): Tag of the bento to serve.
            port (int): Local port to serve on.
        """
        logging.info(f"Starting local server on port {port}")

        try:
            subprocess.run(
                ["bentoml", "serve", bento_tag, "--port", str(port)], check=True
            )
        except KeyboardInterrupt:
            logging.info("Server stopped by user")
        except subprocess.CalledProcessError as e:
            logging.error(f"Server error: {e}")


def main():
    """CLI entry point to download the model, build the bento, and optionally deploy or serve locally."""
    parser = argparse.ArgumentParser(description="Deploy the SMART model using BentoML")
    parser.add_argument(
        "--model_path", type=str, help="Path to local best.pt model (optional)"
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        default="smart-object-detection",
        help="Name of the deployment",
    )
    parser.add_argument(
        "--serve_locally",
        action="store_true",
        help="Serve locally instead of deploying to the cloud",
    )
    parser.add_argument("--port", type=int, default=3000, help="Port to serve locally")
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Only build the bento without deploying",
    )

    args = parser.parse_args()

    deployer = BentoMLDeployer()

    try:
        model_path = deployer.download_best_model(args.model_path)
        logging.info(f"Model ready: {model_path}")

        bento_tag = deployer.build_bento()

        if args.build_only:
            logging.info(f"Build completed: {bento_tag}")
            return 0

        if args.serve_locally:
            deployer.serve_locally(bento_tag, args.port)
        else:
            success = deployer.deploy_to_cloud(bento_tag, args.deployment_name)
            if success:
                logging.info("🎉 Deployment successful!")
                return 0
            else:
                logging.error("❌ Deployment failed")
                return 1

    except Exception as e:
        logging.error(f"Deployment error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
