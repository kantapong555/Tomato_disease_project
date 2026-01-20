console.log("✅ script.js loaded!");
alert("✅ script.js is working!");

const startCameraBtn = document.getElementById("start-camera");
const toggleCameraBtn = document.getElementById("toggle-camera");
const takePhotoBtn = document.getElementById("take-photo");
const uploadPhotoBtn = document.getElementById("upload-photo");
const fileInput = document.getElementById("file-input");
const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const preview = document.getElementById("preview");
const statusText = document.getElementById("status");

let stream;
let useFrontCamera = true;

// ✅ เปิดกล้อง
async function startCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    video: {
      facingMode: useFrontCamera ? "user" : "environment"
    }
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.style.display = "block";
    preview.style.display = "none";
    statusText.textContent = "✅ กล้องเปิดแล้ว";
  } catch (err) {
    console.error(err);
    statusText.textContent = "❌ ไม่สามารถเปิดกล้องได้";
  }
}

startCameraBtn.addEventListener("click", startCamera);

// ✅ สลับกล้องหน้า–หลัง
toggleCameraBtn.addEventListener("click", () => {
  useFrontCamera = !useFrontCamera;
  startCamera();
});

// ✅ ถ่ายภาพ
takePhotoBtn.addEventListener("click", () => {
  if (!stream) return alert("กรุณาเปิดกล้องก่อน!");

  const context = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataURL = canvas.toDataURL("image/jpeg");
  preview.src = dataURL;
  preview.style.display = "block";
  video.style.display = "none";
  statusText.textContent = "📸 ถ่ายภาพเสร็จแล้ว";
});

// ✅ อัปโหลดภาพไป backend
uploadPhotoBtn.addEventListener("click", async () => {
  statusText.textContent = "⏳ กำลังอัปโหลด...";
  console.log("🚀 เริ่มอัปโหลดภาพ...");

  const formData = new FormData();

  // ✅ 1. ถ้ามีรูปจากกล้อง (preview.src เริ่มด้วย data:image)
  if (preview.src.startsWith("data:image")) {
    const blob = await (await fetch(preview.src)).blob();
    formData.append("image", blob, "capture.jpg");
  } 
  // ✅ 2. ถ้าเลือกจากไฟล์
  else if (fileInput.files.length > 0) {
    formData.append("image", fileInput.files[0]);
  } 
  else {
    alert("ยังไม่มีภาพที่จะอัปโหลด!");
    statusText.textContent = "⚠️ กรุณาเลือกรูปภาพหรือถ่ายภาพก่อน";
    return;
  }

  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    console.log("📡 Response status:", res.status);
    const result = await res.json();
    console.log("📩 Response JSON:", result);
    
    if (res.ok) {
      document.getElementById("disease-th").textContent = result.disease_th;
      document.getElementById("disease-en").textContent = `(${result.prediction})`;
      document.getElementById("confidence").textContent = `ความมั่นใจ: ${result.confidence}`;
    
      // ✅ เพิ่มส่วนแสดงคำแนะนำ (ถ้าคุณมี element id="advice" ใน html)
      // หรือจะเอาไปต่อท้ายชื่อโรคเลยก็ได้ครับ แบบง่ายๆ:
      statusText.innerText = "💡 คำแนะนำ: " + result.advice; 
      statusText.style.color = "blue"; // เปลี่ยนสีหน่อยให้เด่นๆ
    } else {
      statusText.textContent = `❌ ผิดพลาด: ${result.error}`;
    }
  } catch (err) {
    console.error(err);
    statusText.textContent = "❌ อัปโหลดล้มเหลว (เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ)";
  }
});


// ✅ เลือกไฟล์จากเครื่อง
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = "block";
    video.style.display = "none";
  };
  reader.readAsDataURL(file);
});
