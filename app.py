import streamlit as st
from PIL import Image as PILImage
import numpy as np
import cv2
from ultralytics import YOLO
import os
import time
import requests
from collections import Counter
from io import BytesIO
import base64
import streamlit.components.v1 as components
import gdown

@st.cache_resource
def load_model():
    model_path = "ishaan_model.pt"
    google_drive_id = "1eE1BocN7wiTfMuZAxSPmcktuU-7F765F"

    if not os.path.exists(model_path):
        with st.spinner("🔄 Downloading model from Google Drive..."):
            try:
                url = f"https://drive.google.com/uc?id={google_drive_id}"
                gdown.download(url, model_path, quiet=False)
                
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                return None

    model = YOLO(model_path)
    return model



model = load_model()

# Object detection function
def detect_objects(image, selected_classes, conf_thresh):
    if model is None:
        return np.array(image), None

    results = model(image)
    img_np = np.array(image)
    detected_classes = []

    colors = {"crater": (0, 255, 0), "boulder": (255, 0, 0)}

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confidences = r.boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            if conf < conf_thresh:
                continue

            class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)
            if selected_classes and class_name not in selected_classes:
                continue

            detected_classes.append(class_name)
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(class_name, (255, 255, 0))
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if detected_classes:
        class_counts = Counter(detected_classes)
        st.write("### 📌 Detected Object Counts:")
        for cls, count in class_counts.items():
            st.write(f"- {cls}: {count}")
    else:
        st.warning("⚠️ No objects detected with current settings.")

    pil_img = PILImage.fromarray(img_np)
    return img_np, pil_img

# Sidebar
st.sidebar.title("🔧 Settings")
nav = st.sidebar.radio("Navigate", ["Home", "Detect", "Report"])



# ---- Home ----
if nav == "Home":
    st.title("🌕 Lunar Vision: Crater & Boulder Detection")
    st.markdown("---")
    st.image("https://images-assets.nasa.gov/image/PIA02442/PIA02442~orig.jpg",
             caption="Apollo 17 – Lunar Surface View (NASA)", use_container_width=True)

    st.subheader("🚀 Mission Overview")
    st.markdown("""
    Welcome to **Lunar Vision**, your intelligent interface for **automated crater and boulder detection** 
    on high-resolution lunar surface imagery.

    This tool is developed as part of the **SOI Data Science Challenge 2025** and leverages a trained deep learning model to:
    - 🧠 Automatically detect surface features
    - 📍 Assist space missions in planning safe routes and identifying scientific sites
    - 🌟 Enable real-time analysis from satellite or rover feeds
    """)

    st.subheader("✨ Features & Innovations")
    st.markdown("""
    ✅ **YOLO-based detection model** optimized for lunar terrain  
    ✅ **Real-time UI** – Upload and detect features with a click  
    ✅ **Customizable confidence threshold**  
    ✅ **Multi-class selection:** Detect only craters, boulders, or both  
    ✅ **Downloadable output** 
    """)

    st.subheader("🧪 Bonus Features for Extra Credit")
    st.markdown("""
    💡 **1. Instant Detection Interface**  
    Upload a lunar image on the **Detect** tab, bounding boxes generate with minimal clicks.

    🌍 **2. Mission Planning & Rover Navigation Use**  
    - Generate safe zone maps  
    - Assist rovers in **path planning** using detections
    """)

    st.subheader("📚 How to Use This App")
    st.markdown("""
    - Go to **Detect** → Upload a lunar image and click **Detect Now**  
    - Adjust detection confidence and filter classes  
    - View/download detection results  
    - See reports and insights in **Report** tab
    """)

# ---- Detect ----
elif nav == "Detect":
    st.title("🚁 Detect Lunar Features")
    uploaded_file = st.file_uploader("📂 Upload a Lunar Image", type=["jpg", "jpeg", "png"])
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    if model and hasattr(model, "names"):
        class_list = list(model.names.values())
        selected_classes = st.sidebar.multiselect("Classes to detect", class_list, default=class_list)
    else:
        selected_classes = []

    if uploaded_file:
        image = PILImage.open(uploaded_file).convert("RGB")
        resize_slider = st.slider("Resize image display %", 10, 100, 100)
        new_size = (int(image.width * resize_slider / 100), int(image.height * resize_slider / 100))
        st.image(image.resize(new_size), caption="📷 Uploaded Image", use_container_width=True)

        if st.button("🚀 Detect Now"):
            with st.spinner("Detecting..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)

                result_img, result_pil = detect_objects(image, selected_classes, conf_threshold)
                st.success("✅ Detection Completed")
                st.image(result_img, caption="🧠 Detected Features", use_container_width=True)

                buf = BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button("📅 Download Detected Image", buf.getvalue(), file_name="lunar_detection.png", mime="image/png")


# ---- Report ----


elif nav == "Report":
    st.title("📊 Detection Report")
    st.markdown("---")

    st.subheader("🧠 Model Performance Summary")
    st.markdown("""
    Our detection model is based on **YOLOv8n (nano)**, trained on high-resolution lunar surface images to identify **craters** and **boulders**.

    **Final Evaluation Metrics:**
    - 🔍 mAP@0.5: **0.8596**
    - 📊 mAP@0.5:0.95: **0.6526**
    - 🎯 Precision: **0.8142**
    - 📈 Recall: **0.7635**
    """)

    st.subheader("🧪 Model Training Insights")
    st.markdown("""
    - Trained using **Adam optimizer** with 10 epochs and batch size 8  
    - **Image augmentations** (flipping, brightness) were key to generalization  
    - Used YOLOv8’s **pretrained weights** for transfer learning  
    - Dataset imbalance affected **boulder prediction confidence**
    """)

    st.subheader("🧠 Model Decision Logic")
    st.markdown("""
    The model detects objects by learning distinct spatial and textural patterns:
    
    - **Craters**: Circular depressions with shadow edges  
    - **Boulders**: Small, high-contrast, sharply bounded regions often with cast shadows  
    """)

    st.subheader("📷 Visualization Strategy")
    st.markdown("""
    - Blue boxes: **Craters**  
    - Red boxes: **Boulders**  
    - Saved both `.txt` label files and overlaid detection images  
    """)

    st.subheader("🌍 Real-World Applications")
    st.markdown("""
    - 🔧 **Rover navigation**: Identifying safe or risky terrain  
    - 🪨 **Geological studies**: Automated crater counts for surface aging  
    - 🛰️ **Mission planning**: Filtering flat zones for landing or resource extraction  
    """)

    st.subheader("🚀 Future Improvements")
    st.markdown("""
    - Fine-tuning on **YOLOv8m/l** for better accuracy  
    - Incorporating **elevation/radar metadata** for boulder detection  
    - Adding **Grad-CAM or attention maps** for explainable AI  
    - Improving UI with image comparison & exportable reports  
    """)

    st.success("🔍 Thank you for exploring our Detection Report!")
    st.markdown("*— Team Chaand Sitaare*")

    st.markdown("---")
    st.subheader("📄 Final Report Document")

    # Read PDF as base64
    with open("Lunar Vision-SOI Problem Final Report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # PDF viewer
  

    # Download button
    st.download_button(
        label="📥 Download Full Report",
        data=base64.b64decode(base64_pdf),
        file_name="Lunar Vision-SOI Problem Final Report.pdf",
        mime="application/pdf"
    )

