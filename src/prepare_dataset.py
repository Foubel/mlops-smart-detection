"""Utilities to split a dataset into train/val/test and generate YOLO 'data.yaml' configuration."""

import argparse
import logging
import random
import shutil
from pathlib import Path

import yaml

from config.global_settings import DEFAULT_CLASS_NAMES_FOR_YAML as DEFAULT_CLASS_NAMES

# Logger configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def create_splits(
    base_path: Path,
    output_path: Path,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42,
):
    """Split a dataset into train, val, and test folders with corresponding labels.

    Args:
        base_path (Path): Path to the dataset with 'images' and 'labels' folders.
        output_path (Path): Directory to save the structured dataset.
        train_ratio (float): Train split ratio.
        val_ratio (float): Validation split ratio.
        test_ratio (float): Test split ratio.
        seed (int): Random seed for reproducibility.

    Returns:
        Path: Path to the structured output dataset.

    Raises:
        FileNotFoundError: If input folders are missing or contain no images.
        ValueError: If not enough annotated images are found.
    """
    random.seed(seed)
    logging.info(
        f"Splitting dataset from {base_path} to {output_path} with seed {seed}"
    )
    logging.info(f"Ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")

    source_images_path = base_path / "images"
    source_labels_path = base_path / "labels"

    if not source_images_path.is_dir() or not source_labels_path.is_dir():
        raise FileNotFoundError(f"Missing 'images' or 'labels' folder in {base_path}")

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(source_images_path.glob(f"*{ext}"))
        all_image_files.extend(source_images_path.glob(f"*{ext.upper()}"))

    if not all_image_files:
        raise FileNotFoundError(f"No images found in {source_images_path}")

    logging.info(f"{len(all_image_files)} images found.")

    valid_image_files = []
    missing_annotations = []

    for img_path in all_image_files:
        label_path = source_labels_path / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_image_files.append(img_path)
        else:
            missing_annotations.append(img_path.name)

    if missing_annotations:
        logging.warning(
            f" {len(missing_annotations)} images without annotations will be excluded."
        )
        for missing in missing_annotations[:5]:
            logging.warning(f"  - {missing}")
        if len(missing_annotations) > 5:
            logging.warning(f"  ...and {len(missing_annotations) - 5} more")

        if len(valid_image_files) < 10:
            raise ValueError(
                f"Only {len(valid_image_files)} valid images found, at least 10 required."
            )

    logging.info(f" {len(valid_image_files)} valid annotated images will be used.")

    random.shuffle(valid_image_files)

    num_train = int(len(valid_image_files) * train_ratio)
    num_val = int(len(valid_image_files) * val_ratio)

    splits = {
        "train": valid_image_files[:num_train],
        "val": valid_image_files[num_train : num_train + num_val],
        "test": valid_image_files[num_train + num_val :],
    }

    output_path.mkdir(parents=True, exist_ok=True)

    for split, images in splits.items():
        logging.info(f"Creating split '{split}': {len(images)} images.")
        split_img_dir = output_path / split / "images"
        split_lbl_dir = output_path / split / "labels"

        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            shutil.copy(img, split_img_dir / img.name)
            label_file = source_labels_path / f"{img.stem}.txt"
            shutil.copy(label_file, split_lbl_dir / f"{img.stem}.txt")

    logging.info(f" Splitting completed. Structured dataset in {output_path}")
    logging.info("Final Stats:")
    logging.info(f"  - Train: {len(splits['train'])} images")
    logging.info(f"  - Val: {len(splits['val'])} images")
    logging.info(f"  - Test: {len(splits['test'])} images")
    logging.info(f"  - Total used: {len(valid_image_files)} images")
    logging.info(f"  - Excluded: {len(missing_annotations)} images")

    return output_path


def generate_yolo_config_file(dataset_output_path: Path, class_names: list):
    """Generate YOLO 'data.yaml' config file.

    Args:
        dataset_output_path (Path): Path to root folder with train/val/test subfolders.
        class_names (list): List of class names for YOLO.

    Returns:
        None
    """
    if not class_names:
        logging.error("Class name list is empty. Cannot generate data.yaml.")
        return

    content = {
        "path": str(dataset_output_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": list(class_names),
    }

    config_path = dataset_output_path / "data.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            content, f, sort_keys=False, default_flow_style=False, allow_unicode=True
        )

    logging.info(f"YOLO config file 'data.yaml' generated at: {config_path.resolve()}")
    logging.info("data.yaml content:")
    with open(config_path, "r", encoding="utf-8") as f:
        logging.info(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO dataset by splitting images/labels into train/val/test and generating data.yaml."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset containing 'images' and 'labels'.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for the structured dataset. Defaults to '<dataset_path>_yolo_prepared'.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.6, help="Training split ratio."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation split ratio."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="Test split ratio."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    input_path = Path(args.dataset_path).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else input_path.parent / f"{input_path.name}_yolo_prepared"
    )

    logging.info(f"Input dataset path: {input_path}")
    logging.info(f"Output structured dataset path: {output_path}")

    if (args.train_ratio + args.val_ratio + args.test_ratio) > 1.00001:
        raise ValueError("The sum of train, val, and test ratios must not exceed 1.")

    try:
        create_splits(
            base_path=input_path,
            output_path=output_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        generate_yolo_config_file(output_path, DEFAULT_CLASS_NAMES)

        logging.info(" Dataset preparation completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File not found during preparation: {e}")
    except ValueError as e:
        logging.error(f"Value error during preparation: {e}")
    except Exception as e:
        logging.critical(
            f"Unexpected error during dataset preparation: {e}", exc_info=True
        )
