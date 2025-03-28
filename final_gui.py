import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
from disease_detection_v2 import CNN_NeuralNet
import requests

# Load models
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    leaf_model = YOLO('leaf_detection_model.pt')
    disease_model = CNN_NeuralNet()
    disease_model.load_state_dict(torch.load('plant_disease_model.pth', map_location=device))
    disease_model.to(device)  # Move model to the appropriate device
    disease_model.eval()
    return leaf_model, disease_model, device

# Function to classify disease
def classify_disease(cropped_leaf, disease_model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    pil_image = Image.fromarray(cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_image).unsqueeze(0).to(device)  # Use the device variable
    with torch.no_grad():
        outputs = disease_model(tensor)
        probs = F.softmax(outputs, dim=1)
    top_prob, top_idx = torch.max(probs, 1)
    return CLASS_NAMES[top_idx.item()], top_prob.item()

# Streamlit app
def main():
    st.title("Plant Disease Detection")
    st.sidebar.title("Settings")

    # Input for ESP32-CAM URL
    esp32_cam_url = st.sidebar.text_input("Enter ESP32-CAM URL", "http://192.168.108.41/capture")

    # Load models
    leaf_model, disease_model, device = load_models()

    # Start/Stop stream
    if st.sidebar.button('Start Stream'):
        st.session_state['stream'] = True
    if st.sidebar.button('Stop Stream'):
        st.session_state['stream'] = False

    # Display live feed
    if st.session_state.get('stream', False):
        stframe = st.empty()
        while st.session_state['stream']:
            response = requests.get(esp32_cam_url)
            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                results = leaf_model(image)
                for result in results:
                    boxes = result.boxes.xyxy
                    confidences = result.boxes.conf
                    for box, confidence in zip(boxes, confidences):
                        if float(confidence) > 0.50:
                            x1, y1, x2, y2 = map(int, box)
                            cropped_leaf = image[y1:y2, x1:x2]
                            if cropped_leaf.size > 0:
                                disease_class, disease_conf = classify_disease(cropped_leaf, disease_model,device)
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(image, f"Leaf: {confidence:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(image, f"Disease: {disease_class.split('___')[-1]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                stframe.image(image, channels="BGR")
            time.sleep(0.1)

    # Button to view CSV
    if st.sidebar.button('View CSV'):
        try:
            df = pd.read_csv('results/detection_results.csv')
            st.write(df)
        except FileNotFoundError:
            st.error("CSV file not found. Please run the detection first.")

if __name__ == "__main__":
    CLASS_NAMES = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
        'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]
    main()