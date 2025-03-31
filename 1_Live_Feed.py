# --- START OF FILE 1_Live_Feed.py ---

import streamlit as st
import cv2
import numpy as np
# import pandas as pd # No longer needed here
import time
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
# Make sure this import works or adjust as needed
try:
    from disease_detection_v2 import CNN_NeuralNet
except ImportError:
    st.error("Could not import CNN_NeuralNet from disease_detection_v2.py. Make sure the file exists and is correct.")
    st.stop()
import requests
import os
from datetime import datetime
import csv
# import json # No longer needed here
# import threading # No longer needed

# --- Constants ---
# (CLASS_NAMES, CSV_FILENAME, DEFAULT_CAM_URL, DEFAULT_GPS_URL, GPS_FETCH_INTERVAL, LOG_ERROR_INTERVAL remain same)
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
CSV_FILENAME = 'results/detection_results.csv'
DEFAULT_CAM_URL = "http://192.168.108.41/capture"
# --- MODIFIED: Rename default URL variable for clarity ---
DEFAULT_SENSOR_URL = "http://192.168.76.189" # Base URL for sensor data endpoint
# GPS_FETCH_INTERVAL = 1.5 # No longer needed for separate thread timing
LOG_ERROR_INTERVAL = 10.0 # Log errors less frequently if needed
# --- Define sensor endpoint ---
SENSOR_DATA_ENDPOINT = "/gps" # Assuming the combined data is still at /gps

# --- Model Loading ---
# (load_models function remains the same)
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        leaf_model_path = 'leaf_detection_model.pt'; disease_model_path = 'plant_disease_model.pth'
        if not os.path.exists(leaf_model_path): st.error(f"Leaf model not found: {leaf_model_path}"); return None, None, None
        if not os.path.exists(disease_model_path): st.error(f"Disease model not found: {disease_model_path}"); return None, None, None
        leaf_model = YOLO(leaf_model_path)
        disease_model = CNN_NeuralNet()
        disease_model.load_state_dict(torch.load(disease_model_path, map_location=device))
        disease_model.to(device); disease_model.eval()
        st.success("Models loaded successfully.")
        return leaf_model, disease_model, device
    except Exception as e: st.error(f"Error loading models: {e}"); return None, None, None

# --- Disease Classification ---
# (classify_disease function remains the same)
def classify_disease(cropped_leaf, disease_model, device):
    if disease_model is None: return "Model N/A", 0.0
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    try:
        if not isinstance(cropped_leaf, np.ndarray) or cropped_leaf.size == 0: return "Invalid Input", 0.0
        pil_image = Image.fromarray(cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = disease_model(tensor)
            if outputs is None or not isinstance(outputs, torch.Tensor): return "Model Error", 0.0
            probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, 1)
        if 0 <= top_idx.item() < len(CLASS_NAMES): return CLASS_NAMES[top_idx.item()], top_prob.item()
        else: return "Unknown Index", 0.0
    except Exception as e: return "Classification Error", 0.0

# --- CSV Saving ---
# --- REVERTED: Match original CSV format ---
def save_to_csv(result, filename=CSV_FILENAME):
    # Reverted fieldnames to match the provided CSV example
    fieldnames = ['timestamp', 'plant_type', 'condition', 'confidence',
                  'latitude', 'longitude', 'gps_valid'] # Removed extra fields
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        is_new_file = not os.path.exists(filename) or os.path.getsize(filename) == 0
        with open(filename, 'a', newline='', encoding='utf-8') as file:
            # Use extrasaction='ignore' still useful if result dict has extra keys temporarily
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
            if is_new_file: writer.writeheader()
            writer.writerow(result)
    except Exception as e: st.error(f"Error writing to CSV: {e}") # Show error in main app

# --- Image Fetching ---
# (fetch_image function remains the same)
def fetch_image(url):
    if not url: return None
    try:
        response = requests.get(url, timeout=5); response.raise_for_status()
        img = cv2.imdecode(np.asarray(bytearray(response.content), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None
        return img
    except Exception: return None # Simplified error handling for brevity

# --- MODIFIED: Combined Sensor Data Fetching (like fedup.py) ---
def fetch_sensor_data(base_url):
    """Fetches combined sensor data (GPS, DHT) from the ESP32 server."""
    if not base_url:
        # st.warning("Sensor data URL not provided.") # Avoid flooding UI
        return None
    try:
        sensor_url = f"{base_url.strip('/')}{SENSOR_DATA_ENDPOINT}" # Construct full URL
        response = requests.get(sensor_url, timeout=2)
        response.raise_for_status()
        data = response.json()
        # Basic validation (can be expanded)
        if isinstance(data, dict):
            return data
        else:
            st.warning(f"Received non-dict JSON from sensor endpoint: {data}")
            return None
    except requests.exceptions.Timeout:
        # Fail silently on timeout during streaming to avoid flooding UI
        # Consider logging this differently if needed
        return None
    except requests.exceptions.RequestException as e:
        # Log other request errors less frequently
        current_time = time.time()
        if current_time - st.session_state.get('last_sensor_error_log_time', 0) > LOG_ERROR_INTERVAL:
            st.warning(f"Could not get sensor data from {sensor_url}: {e}")
            st.session_state['last_sensor_error_log_time'] = current_time
        return None
    except ValueError: # Includes JSONDecodeError
        current_time = time.time()
        if current_time - st.session_state.get('last_sensor_error_log_time', 0) > LOG_ERROR_INTERVAL:
            try:
                error_text = response.text[:100] # Get first 100 chars
            except:
                error_text = "(Could not read response text)"
            st.warning(f"Received invalid JSON from sensor endpoint ({sensor_url}): {error_text}...")
            st.session_state['last_sensor_error_log_time'] = current_time
        return None

# --- MODIFIED: Sensor Display (Sidebar) - Accepts placeholders ---
def display_sensor_info(sensor_data, placeholders):
    # Use the passed-in placeholders
    gps_status_placeholder = placeholders['gps_status']
    gps_metrics_placeholder = placeholders['gps_metrics']
    gps_details_placeholder = placeholders['gps_details']
    dht_status_placeholder = placeholders['dht_status']
    dht_metrics_placeholder = placeholders['dht_metrics']

    # --- GPS Section ---
    with gps_status_placeholder.container():
        st.subheader("🛰️ GPS Status") # Subheader inside placeholder
        if sensor_data and isinstance(sensor_data, dict) and sensor_data.get("valid_gps", False):
            lat = sensor_data.get('latitude', 0.0)
            lon = sensor_data.get('longitude', 0.0)
            with gps_metrics_placeholder.container():
                 st.metric("Latitude", f"{lat:.6f}" if isinstance(lat, float) else str(lat))
                 st.metric("Longitude", f"{lon:.6f}" if isinstance(lon, float) else str(lon))
            with gps_details_placeholder.container():
                 st.caption(f"Satellites: {sensor_data.get('satellites', 'N/A')} | "
                            f"Altitude: {sensor_data.get('altitude_m', 'N/A')} m | "
                            f"HDOP: {sensor_data.get('hdop', 'N/A')} | "
                            f"Timestamp: {sensor_data.get('timestamp_utc', 'N/A')}")
                 st.divider() # Divider after GPS details
        else:
            with gps_metrics_placeholder.container():
                st.warning("Waiting for valid GPS fix...")
            with gps_details_placeholder.container():
                st.caption("No valid GPS data received.")
                st.divider() # Divider after GPS details even when invalid

    # --- Environment Section ---
    with dht_status_placeholder.container():
        st.subheader("🌡️ Environment Status") # Subheader inside placeholder
        if sensor_data and isinstance(sensor_data, dict) and sensor_data.get("valid_dht", False):
            temp = sensor_data.get('temperature_c', 'N/A')
            humidity = sensor_data.get('humidity_percent', 'N/A')
            with dht_metrics_placeholder.container():
                st.metric("Temperature", f"{temp}°C" if isinstance(temp, (int, float)) else str(temp))
                st.metric("Humidity", f"{humidity}%" if isinstance(humidity, (int, float)) else str(humidity))
        else:
            with dht_metrics_placeholder.container():
                st.warning("Waiting for valid Temp/Humidity data...")


# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="Live Feed & Detection")
    st.title("🌿 Plant Disease Detection - Live Feed")
    st.sidebar.title("⚙️ Settings & Status")

    # Initialize session state variables
    if 'stream' not in st.session_state: st.session_state['stream'] = False
    # --- MODIFIED: Use a single state variable for combined sensor data ---
    if 'last_sensor_data' not in st.session_state:
        st.session_state['last_sensor_data'] = {"valid_gps": False, "valid_dht": False} # Initial state
    if 'last_cam_error_log_time' not in st.session_state: st.session_state['last_cam_error_log_time'] = 0
    # --- MODIFIED: Add state for sensor error logging ---
    if 'last_sensor_error_log_time' not in st.session_state: st.session_state['last_sensor_error_log_time'] = 0
    # --- REMOVED: Thread-related state variables ---
    # if 'gps_thread_object' not in st.session_state: st.session_state['gps_thread_object'] = None
    # if 'gps_stop_event' not in st.session_state: st.session_state['gps_stop_event'] = None

    # --- Sidebar Inputs ---
    st.sidebar.subheader("Device URLs")
    esp32_cam_url = st.sidebar.text_input("ESP32-CAM URL", DEFAULT_CAM_URL, key="cam_url")
    # --- MODIFIED: Input for base sensor URL ---
    sensor_module_base_url = st.sidebar.text_input("Sensor Module Base URL (IP/Hostname)", DEFAULT_SENSOR_URL, key="sensor_url")
    st.sidebar.subheader("Detection Settings")
    confidence_threshold = st.sidebar.slider("Min Leaf Confidence", 0.0, 1.0, 0.4, 0.05, key="conf_thresh")

    # --- Model Loading ---
    leaf_model, disease_model, device = load_models()
    if leaf_model is None or disease_model is None:
        st.error("Models failed to load. Cannot start stream.")
        st.stop()

    # --- Stream Control ---
    st.sidebar.subheader("Stream Control")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button('Start Stream', key='start_button', disabled=st.session_state.get('stream', False)):
            if not esp32_cam_url: st.sidebar.error("Please enter the ESP32-CAM URL.")
            # --- MODIFIED: No check needed for sensor URL, fetch handles empty ---
            else:
                 st.session_state['stream'] = True
                 st.session_state['last_cam_error_log_time'] = 0
                 st.session_state['last_sensor_error_log_time'] = 0 # Reset sensor log time
                 # --- MODIFIED: Reset sensor data state ---
                 st.session_state['last_sensor_data'] = {"valid_gps": False, "valid_dht": False}
                 # --- REMOVED: Thread starting logic ---
                 st.success("Stream started...")
                 st.rerun() # Rerun once to start the loop immediately

    with col2:
        if st.button('Stop Stream', key='stop_button', disabled=not st.session_state.get('stream', False)):
            st.session_state['stream'] = False
            # --- REMOVED: Thread stopping logic ---
            # --- MODIFIED: Reset sensor data state on stop ---
            st.session_state['last_sensor_data'] = {"valid_gps": False, "valid_dht": False}
            st.info("Stream stopped.")
            st.rerun() # Rerun once to clear the frame and show stopped message

    # --- Sensor Data Display Area (Sidebar) ---
    # --- MODIFIED: Create placeholders ONCE before the loop ---
    st.sidebar.markdown("---") # Separator before sensor info
    sensor_placeholders = {
        'gps_status': st.sidebar.empty(),
        'gps_metrics': st.sidebar.empty(),
        'gps_details': st.sidebar.empty(),
        # 'dht_divider': st.sidebar.empty(), # Divider is now inside dht_status placeholder
        'dht_status': st.sidebar.empty(),
        'dht_metrics': st.sidebar.empty()
    }
    # --- MODIFIED: Display initial/stopped state using the created placeholders ---
    display_sensor_info(st.session_state.get('last_sensor_data'), sensor_placeholders)
    st.sidebar.markdown("---") # Separator after sensor info
    st.sidebar.info("View detection history and analysis on the 'Results Analysis' page.")


    # --- Main Area: Live Feed Display ---
    stframe = st.empty()

    if not st.session_state.get('stream', False):
         stframe.info("Stream stopped. Click 'Start Stream' in the sidebar.")
    else:
        # --- Streaming Loop ---
        while st.session_state.get('stream', False): # Check state directly
            start_time = time.time() # For throttling

            # --- MODIFIED: Fetch sensor data sequentially ---
            current_sensor_data = fetch_sensor_data(sensor_module_base_url)
            if current_sensor_data:
                st.session_state['last_sensor_data'] = current_sensor_data # Update session state

            # --- MODIFIED: Update sensor display inside the loop using the SAME placeholders ---
            display_sensor_info(st.session_state['last_sensor_data'], sensor_placeholders)

            # Fetch Image
            image = fetch_image(esp32_cam_url)

            if image is not None:
                annotated_image = image.copy()
                try:
                    # --- MODIFIED: Pass confidence directly to model if supported, else filter later ---
                    # Assuming YOLOv8+ where conf can be passed directly
                    results = leaf_model(annotated_image, verbose=False, conf=confidence_threshold)

                    # --- MODIFIED: Process results safely ---
                    if isinstance(results, list) and results:
                         result_obj = results[0] # Process first result object
                         # Check if boxes attribute exists and is not None
                         if hasattr(result_obj, 'boxes') and result_obj.boxes is not None:
                            # Move data to CPU and numpy for processing
                            boxes = result_obj.boxes.xyxy.cpu().numpy()
                            confidences = result_obj.boxes.conf.cpu().numpy()

                            for box, confidence in zip(boxes, confidences):
                                # Confidence check already done by model if conf= was passed
                                # If not, add: if confidence < confidence_threshold: continue

                                x1, y1, x2, y2 = map(int, box)
                                h, w, _ = annotated_image.shape
                                # Clamp box coordinates to image dimensions
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                                # Ensure valid box dimensions after clamping
                                if x1 >= x2 or y1 >= y2: continue

                                cropped_leaf = annotated_image[y1:y2, x1:x2]

                                if cropped_leaf.size > 0:
                                    disease_class, disease_conf = classify_disease(cropped_leaf, disease_model, device)
                                    parts = disease_class.split('___', 1)
                                    plant_type = parts[0]
                                    condition = parts[1] if len(parts) > 1 else disease_class

                                    # --- REVERTED: Get latest sensor data and format for original CSV ---
                                    sensor_info = st.session_state.get('last_sensor_data', {}) or {}
                                    current_detection = {
                                        'timestamp': datetime.now().isoformat(),
                                        'plant_type': plant_type, 'condition': condition,
                                        'confidence': f"{disease_conf:.4f}",
                                        # Add sensor data, using .get for safety
                                        'latitude': sensor_info.get('latitude'),
                                        'longitude': sensor_info.get('longitude'),
                                        # 'altitude_m': sensor_info.get('altitude_m'), # Removed
                                        # 'hdop': sensor_info.get('hdop'), # Removed
                                        # 'satellites': sensor_info.get('satellites'), # Removed
                                        'gps_valid': sensor_info.get('valid_gps', False), # Keep gps_valid
                                        # 'temperature_c': sensor_info.get('temperature_c'), # Removed
                                        # 'humidity_percent': sensor_info.get('humidity_percent') # Removed
                                    }
                                    save_to_csv(current_detection)

                                    # --- Annotation (similar logic, ensure coordinates are valid) ---
                                    label_leaf = f"L: {confidence:.2f}"
                                    label_disease = f"{plant_type[:5]}: {condition} ({disease_conf:.2f})" # Shorten plant type if needed

                                    # Calculate text size for background rectangle
                                    (w_l, h_l), base_l = cv2.getTextSize(label_leaf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    (w_d, h_d), base_d = cv2.getTextSize(label_disease, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    max_text_w = max(w_l, w_d)
                                    total_text_h = h_l + base_l + h_d + base_d + 4 # Add padding

                                    # Determine position (above or below box)
                                    text_y_start_above = y1 - 8
                                    text_y_start_below = y2 + total_text_h + 8

                                    bg_x1 = x1
                                    bg_x2 = x1 + max_text_w + 8

                                    # Prefer placing text above the box if space allows
                                    if text_y_start_above > total_text_h:
                                        bg_y1 = text_y_start_above - total_text_h
                                        bg_y2 = text_y_start_above + base_d # Adjust based on baseline
                                        text1_y = text_y_start_above - h_d - base_d - 2 # Position first line
                                        text2_y = text_y_start_above # Position second line
                                    else: # Place below
                                        bg_y1 = y2 + 4
                                        bg_y2 = y2 + total_text_h + 4
                                        text1_y = y2 + h_l + base_l + 4 # Position first line
                                        text2_y = y2 + total_text_h # Position second line

                                    # Draw background rectangle and text
                                    cv2.rectangle(annotated_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0,100,0), -1) # Dark green background
                                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                                    cv2.putText(annotated_image, label_leaf, (x1 + 4, text1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text
                                    cv2.putText(annotated_image, label_disease, (x1 + 4, text2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1) # Light green text

                except Exception as e:
                    # Log error less frequently to avoid flooding UI
                    current_time = time.time()
                    if current_time - st.session_state.get('last_proc_error_log_time', 0) > LOG_ERROR_INTERVAL:
                         st.error(f"Error during detection/annotation: {e}")
                         st.session_state['last_proc_error_log_time'] = current_time

                # Display the annotated image
                stframe.image(annotated_image, channels="BGR", use_container_width=True)
            else:
                # Handle image fetch failure
                current_time = time.time()
                if current_time - st.session_state.get('last_cam_error_log_time', 0) > LOG_ERROR_INTERVAL:
                    stframe.warning(f"Failed to fetch image from camera ({esp32_cam_url}). Retrying...")
                    st.session_state['last_cam_error_log_time'] = current_time
                # Optional: Add a small sleep even on failure to prevent busy-waiting
                time.sleep(0.2)

            # --- MODIFIED: Throttling without st.rerun() ---
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.5 - elapsed) # Aim for ~2 FPS
            time.sleep(sleep_time)

            # --- REMOVED: st.rerun() ---
            # if st.session_state.get('stream', False):
            #      st.rerun()

        # --- MODIFIED: Display stopped message after loop exits ---
        stframe.info("Stream stopped. Click 'Start Stream' in the sidebar.")


if __name__ == "__main__":
    results_dir = os.path.dirname(CSV_FILENAME)
    if results_dir and not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except OSError as e:
            st.error(f"Could not create results directory '{results_dir}': {e}")
    main()

# --- END OF FILE 1_Live_Feed.py ---