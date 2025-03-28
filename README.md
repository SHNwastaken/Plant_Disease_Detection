# 🌿 Plant Disease Detection System

## Overview
An advanced AI-powered system for automated plant leaf disease detection using computer vision and deep learning technologies.

## 🚀 Key Features
- Real-time leaf detection using YOLO object detection
- Comprehensive disease classification across 38 different plant conditions
- Supports multiple plant types including Apple, Tomato, Corn, Grape, and more
- Streamlit-based interactive web interface
- ESP32-CAM integration for remote image capture

## 🛠 Technology Stack
- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- Streamlit
- ESP32-CAM

## 📦 Models
- Leaf Detection Model: Custom YOLO model (`leaf_detection_model.pt`)
- Disease Classification Model: CNN Neural Network (`plant_disease_model.pth`)
  - Trained on 38 different plant disease classes
  - High accuracy multi-class classification

## 🖥 Components
1. `disease_detection_v2.py`: Core disease classification model
2. `final_gui.py`: Streamlit web application
3. `integrated_detection.py`: Combined leaf detection and disease classification
4. `leaf_detection_v1.py`: Initial leaf detection prototype

## 🔍 Detection Workflow
1. Capture image via ESP32-CAM
2. Detect leaf regions using YOLO
3. Crop and classify each leaf's disease condition
4. Display results with confidence scores
5. Optional CSV logging of detections

## 🌈 User Interface
- Start/Stop live streaming
- Adjustable confidence threshold
- Real-time disease detection
- Results visualization and statistics

## 📊 Supported Plant Types
- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

## 🔧 Setup Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required libraries:
  - torch
  - ultralytics
  - streamlit
  - opencv-python
  - numpy

## 📦 Installation
```bash
git clone https://github.com/SHNwastaken/Plant_Disease_Detection.git
cd plant-disease-detection
pip install -r requirements.txt
```

## 🚀 Getting Started
1. Configure ESP32-CAM URL
2. Launch Streamlit app:
   ```bash
   streamlit run final_gui.py
   ```
