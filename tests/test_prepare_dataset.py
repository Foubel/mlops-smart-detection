from pathlib import Path

import yaml

from src.prepare_dataset import create_splits, generate_yolo_config_file


def make_dummy_dataset(root: Path, n=30):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / "images" / f"img_{i}.jpg").touch()
        (root / "labels" / f"img_{i}.txt").touch()


def test_split_image_inclusion_only(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "yolo"
    raw.mkdir()
    out.mkdir()

    # Create 30 images + labels
    make_dummy_dataset(raw, n=30)

    # Run the split function (seed=42 by default for reproducibility)
    create_splits(raw, out, seed=42)

    # 1) Original image filenames
    orig_images = {p.name for p in (raw / "images").glob("*.jpg")}

    # 2) All filenames in out/**/images
    split_paths = list(out.glob("**/images/*.jpg"))
    split_names = {p.name for p in split_paths}

    # 3) Each original image must appear at least once in the splits
    assert orig_images.issubset(
        split_names
    ), f"Some original images are missing: {orig_images - split_names}"

    # 4) No unexpected filenames should appear
    assert split_names.issubset(
        orig_images
    ), f"Unexpected files found: {split_names - orig_images}"

    # 5) Ensure train/images, val/images, test/images exist and contain at least 1 image
    for subset in ("train", "val", "test"):
        subset_folder = out / subset / "images"
        assert subset_folder.exists(), f"Directory {subset_folder} does not exist"
        assert any(subset_folder.glob("*.jpg")), f"No images found in {subset_folder}"


def test_yaml_generation(tmp_path):
    """Check that generate_yolo_config_file creates a valid .yaml file with expected keys."""
    out = tmp_path / "yolo"
    out.mkdir()

    # Call the function (we don't rely on return value, we scan the directory)
    generate_yolo_config_file(out, class_names=[f"cls{i}" for i in range(10)])

    # 1) After the call, exactly one .yaml file should exist at the root of out/
    yaml_files = list(out.glob("*.yaml"))
    assert len(yaml_files) == 1, f"Expected 1 .yaml file, found: {yaml_files}"

    # 2) Load the YAML file
    yaml_path = yaml_files[0]
    data = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))

    # 3) "train" key must end with ".../train/images"
    assert "train" in data, f"Missing 'train' key in {yaml_path}"
    assert data["train"].endswith(
        "train/images"
    ), f"'train' should end with 'train/images', got: {data['train']}"

    # 4) "val" key must end with ".../val/images"
    assert "val" in data, f"Missing 'val' key in {yaml_path}"
    assert data["val"].endswith(
        "val/images"
    ), f"'val' should end with 'val/images', got: {data['val']}"

    # 5) "names" key must be a list of length 10
    assert isinstance(
        data.get("names"), list
    ), f"'names' should be a list, got: {type(data.get('names'))}"
    assert (
        len(data["names"]) == 10
    ), f"'names' should contain 10 classes, got: {len(data['names'])}"
