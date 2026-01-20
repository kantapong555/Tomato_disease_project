import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 🎯 ส่วนที่ 1: เตรียมข้อมูล (ตอบโจทย์: ใช้ข้อมูลจากแหล่งเชื่อถือได้)
# ============================================================
DATASET_PATH = "segmented"  # โฟลเดอร์รูปภาพจาก PlantVillage
IMG_SIZE = (224, 224)       # ขนาดมาตรฐานโมเดลระดับโลก
BATCH_SIZE = 32
EPOCHS = 20                 # เทรน 20 รอบ เพื่อให้แม่นยำเกิน 70%

# Data Augmentation (แก้ปัญหาภาพมัว/มีเงา ตามเกณฑ์ประเมิน)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # หมุนภาพจำลองการถ่ายเบี้ยว
    brightness_range=[0.8, 1.2], # จำลองแสงมาก/น้อย (แก้เรื่องเงา)
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2    # แบ่ง 20% ไว้สอบ (Validation Set)
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# โหลดข้อมูล
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False # ห้ามสุ่มตอนวัดผล
)

# ============================================================
# 🎯 ส่วนที่ 2: สร้างโมเดล CNN (ตอบโจทย์: พัฒนาโมเดล CNN)
# ============================================================
# ใช้ MobileNetV2 (CNN) เพราะแม่นยำและเหมาะกับมือถือ (Prototype)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy']) # ตัวชี้วัดหลัก

# ============================================================
# 🎯 ส่วนที่ 3: เริ่มสอน AI (ตอบโจทย์: ฝึกสอนด้วย TensorFlow)
# ============================================================
print(f"🚀 เริ่มเทรนโมเดล... เป้าหมาย Accuracy > 70%")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3)
    ]
)

# บันทึกโมเดล
os.makedirs("model_ai", exist_ok=True)
model.save("model_ai/tomato_disease_model.h5")
print("✅ บันทึกโมเดลสำเร็จ!")

# ============================================================
# 🎯 ส่วนที่ 4: ประเมินผลละเอียด (ตอบโจทย์: Accuracy, Precision, Recall, F1)
# ============================================================
print("\n📊 กำลังประมวลผลเพื่อสร้างรายงานการวัดผล...")
y_pred = np.argmax(model.predict(val_generator), axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# 1. รายงานค่าทางสถิติ (เอาไปใส่บทที่ 4 ของโครงงาน)
print("\n" + "="*50)
print("📝 Classification Report (ตอบโจทย์ Precision, Recall, F1)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_labels))

# 2. คำนวณ Accuracy รวม
accuracy = np.mean(y_pred == y_true) * 100
print(f"🏆 ความแม่นยำรวม (Accuracy): {accuracy:.2f}%")
if accuracy > 70:
    print("✅ ผ่านเกณฑ์การประเมิน (> 70%)")
else:
    print("⚠️ ต้องปรับปรุงเพิ่ม")

# ============================================================
# 🎯 ส่วนที่ 5: วิเคราะห์ข้อผิดพลาด (ตอบโจทย์: Analyze Errors)
# ============================================================
# วาด Confusion Matrix (ดูว่า AI สับสนโรคไหนกับโรคไหน)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (วิเคราะห์ความผิดพลาด)')
plt.ylabel('ความจริง (True)')
plt.xlabel('ทำนาย (Predicted)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# กราฟ Accuracy/Loss (เอาไปใส่เล่ม)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Graph')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()