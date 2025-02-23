import streamlit as st
import cv2
import numpy as np
import requests
import time
from ultralytics import YOLO
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name()}")

# Load the plant leaf detection model
model = YOLO('best.pt')  # Add .to('cuda') to move model to GPU
model.to('cuda' if torch.cuda.is_available() else 'cpu')
# Set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# ESP32-CAM URL
esp32_cam_url = "http://192.168.120.41/capture"

def fetch_image_from_esp32():
    """Fetch an image from the ESP32-CAM server."""
    try:
        response = requests.get(esp32_cam_url, timeout=3)  # Add timeout
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        else:
            st.error(f"Failed to fetch image. Status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Error: Cannot connect to ESP32-CAM. Please check the URL and connection.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching image: {e}")
        return None

def run_object_detection(image):
    """Run YOLO object detection on the image."""
    # Resize image to smaller size for faster processing
    resized_image = cv2.resize(image, (640, 480))
    results = model(resized_image, stream=True)  # Enable streaming mode
    return next(results)  # Get first result since we're processing single images

def display_results(image, results):
    """Display the image with bounding boxes and labels."""
    for result in results:
        boxes = result.boxes.xyxy  # Get bounding boxes
        labels = result.boxes.cls  # Get class labels
        confidences = result.boxes.conf  # Get confidence scores

        for box, label, confidence in zip(boxes, labels, confidences):
            x1, y1, x2, y2 = map(int, box)
            label_name = result.names[int(label)]  # Changed from model.names to result.names
            confidence_score = float(confidence)

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label_name} {confidence_score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def main():
    st.title("dumbshit")
    st.sidebar.title("Controls")

    # Initialize session state for stream activity
    if 'stream_active' not in st.session_state:
        st.session_state.stream_active = False

    # Button to start/stop the stream
    if st.session_state.stream_active:
        button_label = "Stop Stream"
    else:
        button_label = "Start Stream"

    if st.sidebar.button(button_label):
        st.session_state.stream_active = not st.session_state.stream_active

    if st.session_state.stream_active:
        st.sidebar.write("Stream is active")
        image_placeholder = st.empty()

        max_retries = 3
        retry_count = 0

        while st.session_state.stream_active:
            # Fetch image from ESP32-CAM
            image = fetch_image_from_esp32()
            if image is None:
                retry_count += 1
                st.sidebar.write(f"Retry attempt {retry_count}/{max_retries}...")

                if retry_count >= max_retries:
                    st.sidebar.error("Maximum retries reached. Stopping stream.")
                    st.session_state.stream_active = False
                    break

                time.sleep(1)  # Wait 1 second before retrying
                continue

            # Reset retry count on successful image fetch
            retry_count = 0

            # Run object detection
            results = run_object_detection(image)

            # Display results
            processed_image = display_results(image, results)
            image_placeholder.image(processed_image, channels="BGR", use_column_width=True)

            # Small delay to prevent high CPU usage
            time.sleep(0.1)

    else:
        st.sidebar.write("Stream is stopped")

if __name__ == "__main__":
    main()