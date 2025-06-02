import asyncio
import importlib
import types

import numpy as np
from PIL import Image


def patch_ultralytics_bentoml_mlflow(monkeypatch):
    # --- 1) Fake for ultralytics.YOLO ---
    class FakeResult:
        def __init__(self):
            self.names = {0: "Canette", 1: "Compote", 2: "Brownie"}
            self.speed = {"inference": 25.4}  # Simulated inference time

            class FakeBox:
                def __init__(self, xyxy, conf, cls):
                    self.xyxy = [np.array(xyxy)]
                    self.conf = [np.array([conf])]
                    self.cls = [np.array([cls])]

            # Simulate 1 detection as in the original service
            self.boxes = [FakeBox([10, 20, 100, 120], 0.85, 0)]  # Canette

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return [FakeResult()]

    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.YOLO = FakeYOLO
    monkeypatch.setitem(importlib.sys.modules, "ultralytics", fake_ultra)

    # --- 2) Fake for bentoml ---
    fake_bentoml = types.ModuleType("bentoml")
    # bentoml.service used as @bentoml.service
    fake_bentoml.service = lambda cls: cls

    # bentoml.api used as @bentoml.api or @bentoml.api(...)
    def fake_api(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    fake_bentoml.api = fake_api
    monkeypatch.setitem(importlib.sys.modules, "bentoml", fake_bentoml)

    # --- 3) Fake for mlflow ---
    fake_mlflow = types.ModuleType("mlflow")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_latest_versions(self, *args, **kwargs):
            raise Exception("No model in registry")

    fake_mlflow.MlflowClient = FakeClient
    fake_mlflow.pytorch = types.ModuleType("mlflow.pytorch")
    fake_mlflow.pytorch.load_model = lambda path: None
    monkeypatch.setitem(importlib.sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(importlib.sys.modules, "mlflow.pytorch", fake_mlflow.pytorch)


def test_predict_dict_structure(monkeypatch):
    # Apply stubs before importing service.py
    patch_ultralytics_bentoml_mlflow(monkeypatch)

    from service import YOLOService

    svc = YOLOService()
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    out = asyncio.run(svc.predict(img))

    # 1) Output must be a dict
    assert isinstance(out, dict), "predict must return a dict"

    # 2) Expected keys according to the actual service
    assert "boxes" in out, "Missing key 'boxes' in output"
    assert "inference_time" in out, "Missing key 'inference_time' in output"

    # 3) inference_time must be a number
    assert isinstance(
        out["inference_time"], (float, int)
    ), "'inference_time' must be a number"

    # 4) boxes must be a list
    boxes = out["boxes"]
    assert isinstance(boxes, list), "'boxes' must be a list"

    # 5) If boxes exist, validate their structure
    if boxes:
        first_box = boxes[0]
        required_keys = ["xyxy", "confidence", "class_id", "class_name"]
        for key in required_keys:
            assert key in first_box, f"Missing key '{key}' in the box"

        assert (
            first_box["class_name"] == "Canette"
        ), f"Expected 'Canette', got '{first_box['class_name']}'"
        assert isinstance(
            first_box["confidence"], float
        ), "'confidence' must be a float"
        assert isinstance(first_box["class_id"], int), "'class_id' must be an int"
        assert isinstance(first_box["xyxy"], list), "'xyxy' must be a list"


def test_predict_empty_boxes(monkeypatch):
    # Simulate a result with no detection
    class EmptyResult:
        def __init__(self):
            self.names = {0: "Canette"}
            self.speed = {"inference": 15.2}
            self.boxes = None  # No detection

    class EmptyYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return [EmptyResult()]

    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.YOLO = EmptyYOLO
    monkeypatch.setitem(importlib.sys.modules, "ultralytics", fake_ultra)

    # Stubs for bentoml
    fake_bentoml = types.ModuleType("bentoml")
    fake_bentoml.service = lambda cls: cls
    fake_bentoml.api = lambda fn: fn
    monkeypatch.setitem(importlib.sys.modules, "bentoml", fake_bentoml)

    from service import YOLOService

    svc = YOLOService()
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    out = asyncio.run(svc.predict(img))

    # Ensure structure is correct even with no detections
    assert isinstance(out, dict), "predict must return a dict"
    assert "boxes" in out, "Missing 'boxes'"
    assert "inference_time" in out, "Missing 'inference_time'"
    assert out["boxes"] == [], "'boxes' must be empty when no detections"
    assert isinstance(
        out["inference_time"], (float, int)
    ), "'inference_time' must be a number"
