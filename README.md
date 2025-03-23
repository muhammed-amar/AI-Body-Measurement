# AI-Powered Body Measurement & Apparel Sizing  

This project is a **FastAPI-based AI system** that predicts **body measurements** and recommends **clothing sizes** using deep learning. It processes **front and side images**, removes backgrounds, and utilizes a trained **Keras model** for accurate predictions.  

---

## 📌 Features  
✅ AI-powered **body measurement estimation** using images.  
✅ Automatic **background removal** for clean image processing.  
✅ Deep learning-based **predictions** using a trained model.  
✅ **FastAPI integration** for quick and efficient API responses.  

---


## 🚀 Installation  

To set up the project, follow these steps:  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/CloozyBrands/AI-BodyMeasurement.git
```
---
### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the FastAPI Server**  
```bash
uvicorn app.main:app --reload
```
Then open **http://127.0.0.1:8000/docs** to test the API.

---

## 🛠️ How It Works  

### **1️⃣ Upload Images**  
- The API accepts **two images**:  
  - **Front view**
  - **Side view**  

### **2️⃣ AI-Based Processing**  
- The images are processed using **`single_person_processor.py`**, which:  
  ✅ Removes the background using `rembg`.  
  ✅ Converts the images into a **model-compatible format**.  
  ✅ Predicts **body measurements** like chest, waist, and height.  

### **3️⃣ Clothing Size Prediction**  
- Based on the body measurements, the system suggests a **T-shirt and pants size** using predefined **size charts**.

---

## 🔍 API Endpoints  

### **1️⃣ `/predict/` (POST) - Predict Body Measurements**  
📌 **Example Request:**  
```http
POST /predict/
```

📌 **Request Parameters:**  
| Parameter     | Type   | Description |
|---------------|--------|-------------|
| `front_image` | File   | Front view image (JPEG/PNG) |
| `side_image`  | File   | Side view image (JPEG/PNG) |
| `input_data`  | JSON   | User data (height, weight, gender) |

📌 **Example JSON Payload:**  
```json
{
  "gender": 0, 
  "height_cm": 175, 
  "weight_kg": 70, 
  "apparel_type": "all"
}
```

📌 **Example Response:**  
```json
{
  "results": {
    "body_measurements": {
      "chest": 100.5,
      "waist": 80.2,
      "hip": 97.3
    },
    "tshirt_size": "L",
    "pants_size": 34
  }
}
```

---

## 🎯 Model & AI Processing  

📌 The model used in this project is a **TensorFlow/Keras** model stored as `best_model.keras`.  
📌 The AI processing is handled inside **`single_person_processor.py`**, which:  
- Loads the trained model using `tf.keras.models.load_model`.  
- Extracts measurements based on input images.  
- Maps the measurements to standard clothing sizes.  

---

## 📜 License  

This project is licensed under the **MIT License** – feel free to modify and use it.  

---

## 🤝 Contributing  

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Submit a pull request  

We welcome contributions and improvements! 🚀  
```
