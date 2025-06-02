import logging
import os
import bentoml
import numpy as np
from PIL import Image

logger = logging.getLogger("bentoml")


@bentoml.service
class YOLOService:
    def __init__(self) -> None:
        self.model = self.load_model()

    def load_model(self):
        """Load the YOLO model."""
        from ultralytics import YOLO

        # Chercher le modèle déployé
        if os.path.exists("deployed_model.pt"):
            logger.info("Loading deployed model: deployed_model.pt")
            try:
                return YOLO("deployed_model.pt")
            except Exception as e:
                logger.warning(f"Failed to load deployed model: {e}")

        # Fallback vers le modèle de base
        logger.info("Using base YOLO model")
        return YOLO("yolo11n.pt")

    @bentoml.api
    async def predict(self, image: Image.Image) -> dict:
        """Handle prediction requests.

        Args:
            image: PIL Image object

        Returns:
            dict: Prediction results including boxes, scores, and class labels
        """
        img_array = np.array(image)
        results = self.model.predict(img_array)
        result = results[0]

        boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                boxes.append(
                    {
                        "xyxy": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0].item()),
                        "class_id": int(box.cls[0].item()),
                        "class_name": result.names[int(box.cls[0].item())],
                    }
                )

        return {
            "boxes": boxes,
            "inference_time": float(result.speed.get("inference", 0)) if hasattr(result, 'speed') else 0
        }