# tests/test_download_dataset_stub.py
import shutil
import sys
import tempfile
import types
from pathlib import Path


def test_download_function_exists():
    # --- 1) Full stub of the "picsellia" module and its subpackages ---
    fake_sdk = types.ModuleType("picsellia")

    # 1.a) Submodule "picsellia.exceptions" with ResourceNotFoundError
    fake_exceptions = types.ModuleType("picsellia.exceptions")
    fake_exceptions.ResourceNotFoundError = Exception
    fake_sdk.exceptions = fake_exceptions

    # 1.b) Submodule "picsellia.sdk.dataset_version" with AnnotationFileType
    fake_sdk_pkg = types.ModuleType("picsellia.sdk")
    fake_dataset_version = types.ModuleType("picsellia.sdk.dataset_version")

    class FakeAnnotationFileType:
        JSON = "json"
        TXT = "txt"
        YOLO = "yolo"

    fake_dataset_version.AnnotationFileType = FakeAnnotationFileType
    fake_sdk_pkg.dataset_version = fake_dataset_version

    # 1.c) Client stub
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_dataset_by_id(self, dataset_id):
            class FakeDataset:
                def __init__(self):
                    self.name = "Test Dataset"
                    self.id = dataset_id

                def get_last_version(self):
                    return FakeDatasetVersion()

            return FakeDataset()

        def get_dataset_version_by_id(self, version_id):
            return FakeDatasetVersion()

    class FakeDatasetVersion:
        def __init__(self):
            self.version = "1.0"
            self.name = "test_version"
            self.id = "test_id"

        def download_assets(self, target_folder):
            Path(target_folder).mkdir(parents=True, exist_ok=True)
            (Path(target_folder) / "test_image.jpg").touch()

        def export_annotation_file(self, annotation_file_type, target_path):
            Path(target_path).mkdir(parents=True, exist_ok=True)
            (Path(target_path) / "test_annotation.txt").touch()

    fake_sdk.Client = FakeClient

    # 1.d) Register stubs in sys.modules
    sys.modules["picsellia"] = fake_sdk
    sys.modules["picsellia.exceptions"] = fake_exceptions
    sys.modules["picsellia.sdk"] = fake_sdk_pkg
    sys.modules["picsellia.sdk.dataset_version"] = fake_dataset_version

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # --- 2) Try importing the function ---
            from src.download_dataset import download_picsellia_dataset

            assert callable(
                download_picsellia_dataset
            ), "download_picsellia_dataset must be a callable function"

            # Basic test using fake arguments in a temporary directory
            try:
                result = download_picsellia_dataset(
                    api_token="fake_token",
                    dataset_id="fake_id",
                    version_identifier="latest",
                    output_root_dir=temp_dir,
                    host="https://fake.host",
                )

                assert isinstance(
                    result, (str, type(None))
                ), "Function should return a string or None"

                if result:
                    result_path = Path(result)
                    assert (
                        result_path.exists()
                    ), f"The returned path {result} must exist"

            except Exception as e:
                if "Token API Picsellia" in str(e) or "fake_token" in str(e):
                    pass  # Acceptable for fake context
                else:
                    raise  # Re-raise unexpected exceptions

        finally:
            # Clean up fake modules
            for module in [
                "picsellia",
                "picsellia.exceptions",
                "picsellia.sdk",
                "picsellia.sdk.dataset_version",
            ]:
                if module in sys.modules:
                    del sys.modules[module]

    cleanup_temp_dirs()


def cleanup_temp_dirs():
    for temp_pattern in ["temp_test", "test_*"]:
        for path in Path(".").glob(temp_pattern):
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                    print(f"🧹 Cleaned: {path}")
                except Exception:
                    pass  # Ignore cleanup errors
