from flask import Flask, request, jsonify, make_response
from keras.models import load_model
import numpy as np
import os
from PIL import Image
import io

app = Flask(__name__)

# ✅ แก้ไข CORS Manual Headers (กันเหนียว)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 🔹 โหลดโมเดล (สมองก้อนใหม่ของคุณ)
try:
    # ตรวจสอบ path ให้ชัวร์
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_ai", "tomato_disease_model.h5")
    print(f"⏳ Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 🔹 รายชื่อคลาส (ต้องตรงกับตอนเทรนเป๊ะๆ)
class_name = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

# 🔹 ฐานข้อมูลโรค + คำแนะนำ (ตอบโจทย์โครงงาน)
disease_info = {
    "Tomato___Bacterial_spot": {
        "thai": "โรคใบจุดแบคทีเรีย",
        "advice": "ตัดใบที่เป็นโรคเผาทำลาย, ใช้สารป้องกันกำจัดแบคทีเรีย เช่น คอปเปอร์ไฮดรอกไซด์"
    },
    "Tomato___Early_blight": {
        "thai": "โรคใบไหม้ระยะแรก",
        "advice": "หมั่นตัดแต่งใบ, ฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น แมนโคเซบ หรือ คลอโรทาโลนิล"
    },
    "Tomato___healthy": {
        "thai": "ใบปกติ (สุขภาพดี)",
        "advice": "ต้นมะเขือเทศแข็งแรงดี ควรหมั่นรดน้ำและใส่ปุ๋ยตามระยะเวลาที่เหมาะสม"
    },
    "Tomato___Late_blight": {
        "thai": "โรคใบไหม้ระยะสุดท้าย",
        "advice": "ระบาดรุนแรง! ให้รีบตัดส่วนที่เป็นโรคทิ้งทันที และพ่นสารเมทาแลกซิลสลับกับแมนโคเซบ"
    },
    "Tomato___Leaf_Mold": {
        "thai": "โรคราแป้ง",
        "advice": "ลดความชื้นในแปลง, ตัดแต่งใบให้อากาศถ่ายเท, ใช้กำมะถันผงชนิดละลายน้ำฉีดพ่น"
    },
    "Tomato___Septoria_leaf_spot": {
        "thai": "โรคใบจุดเซปโทเรีย",
        "advice": "เก็บเศษพืชที่เป็นโรคเผาทำลาย, พ่นสารป้องกันเชื้อรากลุ่ม Azoxystrobin"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "thai": "ไรแดง",
        "advice": "ใช้น้ำฉีดพ่นใต้ใบแรงๆ เพื่อล้างไรแดง, หากระบาดหนักใช้สารกำจัดไร เช่น อะบาเมกติน"
    },
    "Tomato___Target_Spot": {
        "thai": "โรคใบจุดวงกลม",
        "advice": "ดูแลแปลงให้สะอาด, พ่นสารป้องกันเชื้อรา เช่น คลอโรทาโลนิล หรือ ไดฟีโนโคนาโซล"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "thai": "ไวรัสใบม้วนเหลือง",
        "advice": "เกิดจากแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง และพ่นสารกำจัดแมลงจำพวกอิมิดาคลอพริด"
    }
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ API Ready!"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    if "image" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์รูปภาพ"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "ไม่ได้เลือกไฟล์"}), 400

    try:
        # อ่านภาพจาก RAM
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224)) # ⚠️ แก้ขนาดเป็น 224 ตามโมเดลใหม่
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ทำนายผล
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        
        if predicted_class_index < len(class_name):
            disease_en = class_name[predicted_class_index]
        else:
            disease_en = "Unknown"

        # ดึงข้อมูลจากฐานข้อมูล
        info = disease_info.get(disease_en, {"thai": "ไม่รู้จัก", "advice": "ไม่พบข้อมูลคำแนะนำ"})
        
        confidence = float(np.max(prediction) * 100)
        
        return jsonify({
            "prediction": disease_en,
            "disease_th": info["thai"],
            "advice": info["advice"],  # ✅ ส่งคำแนะนำกลับไปด้วย
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)