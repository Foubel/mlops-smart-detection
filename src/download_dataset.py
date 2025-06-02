"""This script downloads a dataset from Picsellia in YOLO format."""

import logging
import os
import random
import shutil
import zipfile
from pathlib import Path

from picsellia import Client
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.dataset_version import AnnotationFileType

from config.global_settings import (
    PICSELLIA_API_TOKEN,
    PICSELLIA_DATASET_ID,
    PICSELLIA_DATASET_VERSION_NAME,
    settings,
)

# Logger configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Load Picsellia host from .env via Dynaconf
PICSELLIA_HOST = settings.get("PICSELLIA_HOST", "https://app.picsellia.com")


def download_picsellia_dataset(
    api_token: str = None,
    dataset_id: str = None,
    version_identifier: str = None,
    output_root_dir: str = "data",
    host: str = "https://app.picsellia.com",
) -> str:
    """Download a dataset and its YOLO annotations from Picsellia, preparing the folder structure.

    Args:
        api_token (str, optional): Picsellia API token.
        dataset_id (str, optional): ID of the dataset to download.
        version_identifier (str, optional): Dataset version name or ID, or 'latest'.
        output_root_dir (str): Root output folder to store the dataset.
        host (str): Picsellia host URL.

    Returns:
        str: Absolute path to the prepared dataset directory.

    Raises:
        ValueError: If no API token is provided.
        ResourceNotFoundError: If the dataset or version cannot be found.
        Exception: For any download or extraction failure.
    """
    p_api_token = api_token or PICSELLIA_API_TOKEN
    p_dataset_id = dataset_id or PICSELLIA_DATASET_ID
    p_version_identifier = version_identifier or PICSELLIA_DATASET_VERSION_NAME

    if not p_api_token:
        logging.error("Picsellia API token is missing.")
        raise ValueError("Picsellia API token is required.")

    logging.info("Initializing Picsellia client...")
    client = Client(api_token=p_api_token, host=host)

    try:
        dataset = client.get_dataset_by_id(p_dataset_id)
        logging.info(f"Dataset '{dataset.name}' (ID: {dataset.id}) found.")
    except Exception as e:
        logging.error(f"Failed to fetch dataset with ID '{p_dataset_id}': {e}")
        raise

    try:
        logging.info(f"Retrieving version '{p_version_identifier}'...")
        if p_version_identifier.lower() == "latest":
            dsv = dataset.get_last_version()
            if not dsv:
                raise ResourceNotFoundError(
                    f"No 'latest' version found for dataset {dataset.id}"
                )
        else:
            dsv = client.get_dataset_version_by_id(p_version_identifier)

        logging.info(f"Using version: {dsv.version} (Name: {dsv.name}, ID: {dsv.id})")
    except Exception as e:
        logging.error(f"Failed to retrieve version '{p_version_identifier}': {e}")
        raise

    dataset_name_slug = dsv.name.lower().replace(" ", "_").replace("-", "_")
    dataset_output_dir = Path(output_root_dir) / dataset_name_slug
    images_dir = dataset_output_dir / "images"
    labels_dir = dataset_output_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Preparing dataset in: {dataset_output_dir}")

    # Download images
    logging.info(f"Downloading images to {images_dir}...")
    try:
        if hasattr(dsv, "download_assets"):
            dsv.download_assets(target_folder=str(images_dir))
        elif hasattr(dsv, "list_assets"):
            dsv.list_assets().download(target_path=str(images_dir))
        else:
            raise NotImplementedError("No method found to download assets.")
        logging.info("Images successfully downloaded.")
    except Exception as e:
        logging.error(f"Image download failed: {e}")
        raise

    # Export YOLO annotations
    logging.info(f"Exporting YOLO annotations to {labels_dir}...")
    try:
        # Clean up any old ZIPs
        for old_zip in list(labels_dir.rglob("*.zip")):
            try:
                old_zip.unlink()
                logging.info(f"Deleted old ZIP: {old_zip}")
            except Exception as e:
                logging.warning(f"Could not delete old ZIP {old_zip}: {e}")

        dsv.export_annotation_file(
            annotation_file_type=AnnotationFileType.YOLO, target_path=str(labels_dir)
        )
        logging.info("Annotations exported from Picsellia.")

        # Process ZIPs or directly extracted .txt
        zip_files = list(labels_dir.rglob("*.zip"))
        if not zip_files:
            logging.info("No ZIP file found. Checking for .txt files...")

            txt_found = 0
            for directory in [labels_dir] + [
                d for d in labels_dir.iterdir() if d.is_dir()
            ]:
                for txt_file in directory.glob("*.txt"):
                    target_path = labels_dir / txt_file.name
                    if txt_file != target_path:
                        try:
                            shutil.move(str(txt_file), str(target_path))
                            logging.info(f"Moved: {txt_file.name}")
                        except shutil.Error as e:
                            logging.warning(f"Failed to move {txt_file.name}: {e}")
                    txt_found += 1

                if directory != labels_dir and not any(directory.iterdir()):
                    try:
                        directory.rmdir()
                        logging.info(f"Removed empty subdirectory: {directory}")
                    except OSError as e:
                        logging.warning(f"Could not remove folder {directory}: {e}")

            if txt_found == 0:
                logging.warning("No .txt files found after annotation export.")

        else:
            for zip_file in zip_files:
                temp_dir = (
                    labels_dir
                    / f"temp_extract_{zip_file.stem}_{random.randint(1000, 9999)}"
                )
                temp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                    logging.info(f"Extracted: {zip_file}")

                    moved_count = 0
                    for root, _, files in os.walk(temp_dir):
                        for f in files:
                            if f.endswith(".txt"):
                                src = Path(root) / f
                                dst = labels_dir / f
                                try:
                                    shutil.move(str(src), str(dst))
                                    moved_count += 1
                                except shutil.Error as e:
                                    logging.warning(f"Could not move {f}: {e}")

                    logging.info(f"{moved_count} .txt files moved from ZIP.")

                except Exception as e:
                    logging.error(f"Error processing ZIP {zip_file}: {e}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    zip_file.unlink(missing_ok=True)

        final_txt_files = list(labels_dir.glob("*.txt"))
        if not final_txt_files:
            logging.error(
                f"CRITICAL: No .txt files found in {labels_dir} after all processing."
            )
        else:
            logging.info(
                f"SUCCESS: {len(final_txt_files)} annotation files available in {labels_dir}."
            )

    except Exception as e:
        logging.error(f"Annotation export or processing failed: {e}")
        raise

    logging.info("Dataset download and preparation completed successfully.")
    logging.info(f"  ➡️  Images in: {images_dir.resolve()}")
    logging.info(f"  ➡️  YOLO annotations in: {labels_dir.resolve()}")
    return str(dataset_output_dir.resolve())


if __name__ == "__main__":
    logging.info("Starting Picsellia dataset download script...")

    output_root_dir = settings.get("OUTPUT_DATA_DIR", "data")
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = download_picsellia_dataset(
            api_token=PICSELLIA_API_TOKEN,
            dataset_id=PICSELLIA_DATASET_ID,
            version_identifier=PICSELLIA_DATASET_VERSION_NAME,
            output_root_dir=output_root_dir,
            host=PICSELLIA_HOST,
        )
        if downloaded_path:
            logging.info(f"Dataset available at: {downloaded_path}")
        else:
            logging.error("Dataset download failed. See logs above.")

    except Exception as e:
        logging.critical(f"Unhandled error occurred: {e}", exc_info=True)

    logging.info("Dataset download script completed.")
