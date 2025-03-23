from fastapi import FastAPI, File, UploadFile, Form
import json
import tempfile
import shutil
import os
from tesssst.single_person_processor import SinglePersonPredictor

app = FastAPI()

# Initialize the predictor
predictor = SinglePersonPredictor(model_path="best_model.keras")

@app.post("/predict/")
async def predict(
    front_image: UploadFile = File(...),  
    side_image: UploadFile = File(...),  
    input_data: str = Form(...)  
):
    try:
        try:
            input_dict = json.loads(input_data)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format in input_data"}

        # Validate file types
        if front_image.content_type not in ["image/jpeg", "image/png"]:
            return {"error": "Front image must be a JPEG or PNG file"}
        if side_image.content_type not in ["image/jpeg", "image/png"]:
            return {"error": "Side image must be a JPEG or PNG file"}

        # Create temporary files for images
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as front_temp:
            shutil.copyfileobj(front_image.file, front_temp)
            front_temp_path = front_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as side_temp:
            shutil.copyfileobj(side_image.file, side_temp)
            side_temp_path = side_temp.name

        # Perform predictions
        results = predictor.predict_measurements(
            front_img_path=front_temp_path,
            side_img_path=side_temp_path,
            gender=input_dict.get("gender"),
            height_cm=input_dict.get("height_cm"),
            weight_kg=input_dict.get("weight_kg"),
            apparel_type=input_dict.get("apparel_type")
        )

        # Clean up temporary files
        os.remove(front_temp_path)
        os.remove(side_temp_path)
        
        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
