from ultralytics import YOLO
import cv2
import requests
import numpy as np
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
            print(f"Failed to fetch image. Status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to ESP32-CAM. Please check the URL and connection.")
        raise SystemExit(1)  # Exit the program with error code 1
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
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

    cv2.imshow("Plant Leaf Detection", image)

def main():
    max_retries = 3
    retry_count = 0
    
    try:
        while True:
            # Fetch image from ESP32-CAM
            image = fetch_image_from_esp32()
            if image is None:
                retry_count += 1
                print(f"Retry attempt {retry_count}/{max_retries}...")
                
                if retry_count >= max_retries:
                    print("Maximum retries reached. Exiting program.")
                    break
                    
                cv2.waitKey(1000)  # Wait 1 second before retrying
                continue
            
            # Reset retry count on successful image fetch
            retry_count = 0

            # Run object detection - will take as long as needed
            results = run_object_detection(image)

            # Display results - will wait until processing is complete
            display_results(image, results)

            # Check for 'q' key press after each frame is processed and displayed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping the application...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Program terminated.")

if __name__ == "__main__":
    main()
