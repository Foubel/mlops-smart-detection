"""SMART Inference Pipeline module."""

import argparse
import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

# Logging configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)


class SMARTInferencePipeline:
    """Performs inference using a YOLO model trained in the SMART project.

    Supports image, video, and real-time webcam inference. The model is loaded
    from a local path.
    """

    def __init__(self):
        """Initialize the inference pipeline."""
        self.model = None

    def load_model(self, model_path: str):
        """Load a local YOLO model.

        Args:
            model_path (str): Path to a local YOLO model file (e.g., 'best.pt').
        """
        if not model_path:
            raise ValueError("model_path is required")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logging.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        logging.info("Model successfully loaded!")

    def predict_image(self, image_path: str, save_results: bool = True):
        """Run inference on a static image.

        Args:
            image_path (str): Path to the image file.
            save_results (bool, optional): Whether to save annotated image. Defaults to True.

        Returns:
            List: Detection results from YOLO model.

        Raises:
            Exception: If no model is loaded.
            FileNotFoundError: If image file does not exist.
        """
        if not self.model:
            raise Exception("No model loaded")

        logging.info(f"Running inference on image: {image_path}")

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(image_path, save=save_results, conf=0.25)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                logging.info(f"Detections found: {len(boxes)}")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    logging.info(f"  - {class_name}: {confidence:.2f}")
            else:
                logging.info("No detections found")

        return results

    def predict_video(self, video_path: str, save_results: bool = True):
        """Run inference on a video file.

        Args:
            video_path (str): Path to the video file.
            save_results (bool, optional): Whether to save annotated video. Defaults to True.

        Returns:
            List: Detection results for each frame.

        Raises:
            Exception: If no model is loaded.
            FileNotFoundError: If video file does not exist.
        """
        if not self.model:
            raise Exception("No model loaded")

        logging.info(f"Running inference on video: {video_path}")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        results = self.model.predict(video_path, save=save_results, conf=0.25)
        logging.info(f"Video processing completed, {len(results)} frames processed")
        return results

    def predict_webcam(self, camera_index: int = 0):
        """Run inference in real-time using the webcam.

        Args:
            camera_index (int, optional): Index of the webcam. Defaults to 0.

        Raises:
            Exception: If no model is loaded or if the camera cannot be opened.
        """
        if not self.model:
            raise Exception("No model loaded")

        logging.info(f"Starting webcam inference (camera {camera_index})")
        logging.info("Press 'q' to quit")

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise Exception(f"Unable to open camera {camera_index}")

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to read frame from camera")
                    break

                frame_count += 1

                if frame_count % 30 == 0:
                    logging.info(f"📹 Frame {frame_count} processed")

                results = self.model.predict(frame, conf=0.1, verbose=False)

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    logging.info(f"{len(boxes)} detection(s) found!")
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = results[0].names[class_id]
                        logging.info(f"  #{i+1}: {class_name} ({confidence:.2f})")
                elif frame_count % 60 == 0:
                    logging.info("No detections yet...")
                    logging.info(
                        f"Target classes: {list(results[0].names.values())}"
                    )

                annotated_frame = results[0].plot()
                cv2.imshow("SMART Object Detection", annotated_frame)

                if frame_count == 1:
                    cv2.setWindowProperty(
                        "SMART Object Detection", cv2.WND_PROP_TOPMOST, 1
                    )
                    cv2.moveWindow("SMART Object Detection", 100, 100)
                    logging.info(
                        "Window created — look for 'SMART Object Detection'"
                    )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logging.info("Exit requested by user")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Webcam closed")


def main():
    """Parse CLI arguments and run the inference pipeline.

    Returns:
        int: Exit status code (0 if success, 1 if error).
    """
    parser = argparse.ArgumentParser(description="SMART Inference Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["IMAGE", "VIDEO", "WEBCAM"],
        required=True,
        help="Inference mode",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to image or video (required for IMAGE and VIDEO modes)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local YOLO model (required)",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index for WEBCAM mode (default: 0)",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save annotated results",
    )

    args = parser.parse_args()

    if args.mode in ["IMAGE", "VIDEO"] and not args.input_path:
        parser.error(f"--input_path is required for mode {args.mode}")

    pipeline = SMARTInferencePipeline()

    try:
        pipeline.load_model(args.model_path)

        if args.mode == "IMAGE":
            pipeline.predict_image(args.input_path, args.save_results)
            logging.info("Image inference completed")

        elif args.mode == "VIDEO":
            pipeline.predict_video(args.input_path, args.save_results)
            logging.info("Video inference completed")

        elif args.mode == "WEBCAM":
            pipeline.predict_webcam(args.camera_index)

    except Exception as e:
        logging.error(f"Inference error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
