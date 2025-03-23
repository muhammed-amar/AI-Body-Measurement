# single_person_processor.py

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import io
import logging
import numpy as np
import tensorflow as tf
from rembg import new_session, remove
from PIL import Image
from typing import Tuple, Dict
from tensorflow.keras.preprocessing.image import img_to_array

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImagePreprocessor:
    """Class to preprocess images by removing background."""

    def __init__(self, img_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the image preprocessor.

        Parameters:
        - img_size: Target size for images (default is (128, 128)).
        """
        self.img_size = img_size
        self.session = new_session()  # Create a new session for rembg
        logging.info("Image processor initialized.")

    def process_single_person(self, front_path: str, side_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process two images (front and side views).

        Parameters:
        - front_path: Path to the front image.
        - side_path: Path to the side image.

        Returns:
        - A tuple containing processed front and side images as NumPy arrays.
        """
        return (
            self._process_image(front_path),
            self._process_image(side_path)
        )

    def _process_image(self, image_path: str) -> np.ndarray:
        """
        Remove background and process a single image.

        Parameters:
        - image_path: Path to the image file.

        Returns:
        - Processed image as a NumPy array.
        """
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()

            # Remove background using rembg
            bg_removed = Image.open(io.BytesIO(remove(img_bytes, session=self.session)))

            # Convert person to white and background to black
            white_img = Image.new("RGBA", bg_removed.size, (255, 255, 255, 255))
            white_img.putalpha(bg_removed.getchannel('A'))
            final_img = Image.alpha_composite(
                Image.new("RGBA", white_img.size, (0, 0, 0, 255)),
                white_img
            )

            # Prepare the image for the model
            return self._prepare_image(final_img)

        except Exception as e:
            logging.error(f"Failed to process {image_path}: {str(e)}")
            raise

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """
        Convert the image to the model's input format.

        Parameters:
        - image: PIL Image object.

        Returns:
        - Image as a normalized NumPy array.
        """
        return img_to_array(
            image.convert('L').resize(self.img_size)
        ).astype(np.float32) / 255.0


class SinglePersonPredictor:
    """Class to predict body measurements and clothing sizes."""

    MEASUREMENT_INDICES = {
        'ankle': 0, 'arm-length': 1, 'bicep': 2, 'calf': 3,
        'chest': 4, 'forearm': 5, 'height': 6, 'hip': 7,
        'leg-length': 8, 'shoulder-breadth': 9,
        'shoulder-to-crotch': 10, 'thigh': 11, 'waist': 12, 'wrist': 13
    }

    SIZE_CHARTS = {
        'male': {
            'tshirt': [
                (97, 42, 'S'), (104, 45, 'M'), (112, 48, 'L'),
                (120, 51, 'XL'), (128, 54, 'XXL'), (136, 57, 'XXXL')
            ],
            'pants': [
                (76, 102, 30), (81, 107, 32), (86, 112, 34),
                (91, 117, 36), (97, 122, 38), (102, 127, 40), (107, 132, 42)
            ]
        },
        'female': {
            'tshirt': [
                (89, 38, 'S'), (96, 41, 'M'), (104, 44, 'L'),
                (112, 47, 'XL'), (120, 50, 'XXL'), (128, 53, 'XXXL')
            ],
            'pants': [
                (66, 92, 26), (71, 97, 28), (76, 102, 30),
                (81, 107, 32), (86, 112, 34), (91, 117, 36), (97, 122, 38)
            ]
        }
    }

    def __init__(self, model_path: str = 'best_model.keras'):
        """
        Initialize the predictor.

        Parameters:
        - model_path: Path to the trained model.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = ImagePreprocessor()
        logging.info("Model loaded successfully.")

    def predict_measurements(self,
                             front_img_path: str,
                             side_img_path: str,
                             gender: int,
                             height_cm: float,
                             weight_kg: float,
                             apparel_type: str = "all") -> Dict:
        """
        Perform predictions for a single person and calculate clothing sizes.

        Parameters:
        - front_img_path: Path to the front image.
        - side_img_path: Path to the side image.
        - gender: 0 for male, 1 for female.
        - height_cm: Height in centimeters.
        - weight_kg: Weight in kilograms.
        - apparel_type: Specify "tshirt", "pants", or "all".

        Returns:
        - Dictionary containing predicted measurements and clothing sizes.
        """
        try:
            # Validate inputs
            if gender not in (0, 1):
                raise ValueError("Gender must be 0 (male) or 1 (female).")
            if not 100 <= height_cm <= 250:
                raise ValueError("Height must be between 100-250 cm.")
            if not 30 <= weight_kg <= 300:
                raise ValueError("Weight must be between 30-300 kg.")
            if apparel_type not in ["tshirt", "pants", "all"]:
                raise ValueError("Apparel type must be 'tshirt', 'pants', or 'all'.")

            # Process images
            front_arr, side_arr = self.preprocessor.process_single_person(front_img_path, side_img_path)
            meta_arr = np.array([[gender, height_cm, weight_kg]], dtype=np.float32)

            # Make predictions
            prediction = self.model.predict([
                np.expand_dims(front_arr, axis=0),
                np.expand_dims(side_arr, axis=0),
                meta_arr
            ])

            # Convert predictions to dictionary
            measurements = {
                name: round(float(prediction[0][idx]), 2)
                for name, idx in self.MEASUREMENT_INDICES.items()
            }

            result = {}

            if apparel_type == "tshirt":
                result["tshirt_size"] = self.calculate_tshirt_size(gender, measurements, weight_kg)

            elif apparel_type == "pants":
                result["pants_size"] = self.calculate_pants_size(gender, measurements, weight_kg)

            elif apparel_type == "all":
                result["body_measurements"] = measurements
                tshirt_size, pants_size = self.calculate_apparel_size(gender, measurements, weight_kg)
                result["tshirt_size"] = tshirt_size
                result["pants_size"] = pants_size

            return result

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def calculate_tshirt_size(self, gender: int, measurements: Dict, weight: float) -> str:
        """
        Calculate t-shirt size based on chest and shoulder-breadth measurements.

        Parameters:
        - gender: 0 for male, 1 for female.
        - measurements: Dictionary of body measurements.
        - weight: Weight in kilograms.

        Returns:
        - T-Shirt size as a string.
        """
        gender_str = 'male' if gender == 0 else 'female'
        chart = self.SIZE_CHARTS[gender_str]['tshirt']

        chest = measurements['chest']
        shoulder_breadth = measurements['shoulder-breadth']

        base_size = next(
            (size for max_chest, max_shoulder, size in chart
             if chest <= max_chest and shoulder_breadth <= max_shoulder),
            'XXXL'
        )

        if weight > 95 and gender == 0 or weight > 80 and gender == 1:
            base_size = 'XXL'

        if measurements['height'] > 180:
            base_size = f"Tall {base_size}"

        return base_size

    def calculate_pants_size(self, gender: int, measurements: Dict, weight: float) -> int:
        """
        Calculate pants size based on waist and hip measurements.

        Parameters:
        - gender: 0 for male, 1 for female.
        - measurements: Dictionary of body measurements.
        - weight: Weight in kilograms.

        Returns:
        - Pants size as an integer.
        """
        gender_str = 'male' if gender == 0 else 'female'
        chart = self.SIZE_CHARTS[gender_str]['pants']

        waist = measurements['waist']
        hip = measurements['hip']

        base_size = next(
            (size for max_waist, max_hip, size in chart
             if waist <= max_waist and hip <= max_hip),
            chart[-1][2]
        )

        if measurements['height'] > 180 and gender == 0:
            base_size += 2 if base_size < 40 else 0

        return base_size

    def calculate_apparel_size(self, gender: int, measurements: Dict, weight: float) -> Tuple[str, int]:
        """
        Calculate both t-shirt and pants sizes.

        Parameters:
        - gender: 0 for male, 1 for female.
        - measurements: Dictionary of body measurements.
        - weight: Weight in kilograms.

        Returns:
        - A tuple containing t-shirt size and pants size.
        """
        tshirt_size = self.calculate_tshirt_size(gender, measurements, weight)
        pants_size = self.calculate_pants_size(gender, measurements, weight)
        return tshirt_size, pants_size