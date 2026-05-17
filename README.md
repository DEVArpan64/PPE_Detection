# PPE Detection System using YOLO

> A Deep Learning-based Personal Protective Equipment (PPE) Detection System developed for real-time industrial safety compliance monitoring using multiple YOLO architectures and Streamlit deployment.

---

## 📌 Project Overview

Industrial environments require strict adherence to PPE regulations such as helmets, masks, gloves, safety vests, and eye protection. Manual monitoring is inefficient and prone to human error. This project automates PPE compliance detection using **Computer Vision** and **Deep Learning**.

The system detects:
- ✅ PPE Present Classes
- ❌ PPE Violation Classes
- 👷 Workers & Site Objects

---

# 🧠 Models Compared

The project evaluates and compares **5 YOLO architectures** under identical training conditions:

| Model | Description |
|---|---|
| **YOLOv9s** | Lightweight YOLOv9 variant |
| **YOLOv9c** | High-performance YOLOv9 architecture |
| **YOLOv10n** | Nano architecture optimized for speed |
| **YOLOv10s** | Balanced speed-accuracy model |
| **YOLO11s** | Latest YOLO11 architecture |

### 📊 Evaluation Metrics
Models were compared using:
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Inference Speed

After comparison, **YOLOv9c** was selected as the best-performing model due to its superior accuracy and recall in safety-critical scenarios.

---

# ⚙️ Fine-Tuning & Optimization

The selected **YOLOv9c** model underwent:
- Additional fine-tuning epochs
- Advanced data augmentation
- Confidence-threshold optimization
- Hyperparameter tuning

This improved:
- Detection accuracy
- PPE violation recall
- Generalization performance
- Real-time inference reliability

---

# 🗂️ Dataset

### Dataset Used
- **Construction Site Safety Dataset (Roboflow)**

### Dataset Features
- 25 Classes
- YOLO Annotation Format
- Real-world industrial/construction scenarios
- Multiple lighting and viewpoint conditions

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- Streamlit
- CUDA GPU Training
- ONNX

---

# 🚀 Streamlit Deployment (`app_sand.py`)

A production-ready **Streamlit application** was developed for real-time PPE monitoring.

## Features
- 📷 Real-time image inference
- 🎥 Real-time video inference
- 📦 Bounding box visualization
- 📈 Detection confidence display
- 🎚️ Adjustable confidence threshold
- ⬇️ Annotated output download
- 📋 Detection logs
- 🖼️ Side-by-side result visualization

---

# 📤 ONNX Export

The trained **YOLOv9c** model was exported to **ONNX format** to enable compatibility for future edge deployment on:
- NVIDIA Jetson Nano
- NVIDIA Jetson Orin Nano
- Edge AI deployment pipelines

---

# 📈 Performance Highlights

- ✅ Best Performing Model: **YOLOv9c**
- ✅ High mAP and Recall Scores
- ✅ Real-time Detection Support
- ✅ Fine-tuned for Industrial Safety Monitoring

---

# 📁 Project Structure

```bash
├── models/
├── datasets/
├── runs/
├── app_sand.py
├── train.py
├── evaluate.py
├── requirements.txt
├── best.pt
├── best.onnx
└── README.md
```

---

# 🔮 Future Improvements

- Live CCTV/RTSP stream integration
- Alert notification system
- Compliance analytics dashboard
- TensorRT optimization
- Multi-camera support
- Edge deployment on Jetson devices

---


# 🏭 Applications

- Industrial Safety Monitoring
- Construction Site Compliance
- Smart Surveillance Systems
- Factory Floor Monitoring
- Automated PPE Auditing
