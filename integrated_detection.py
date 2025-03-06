import torch
import torch.nn.functional as F
import cv2
import requests
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
import time

# Import the CNN model from disease_detection_v2
from disease_detection_v2 import CNN_NeuralNet

# ----------------------------
# 1. Setup and Configuration
# ----------------------------
# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# ESP32-CAM URL
esp32_cam_url = "http://192.168.108.41/capture"

# Class names for disease detection
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

# ----------------------------
# 2. Load Models
# ----------------------------
def load_models():
    # Load YOLO leaf detection model
    leaf_model = YOLO('leaf_detection_model.pt')
    leaf_model.to(device)
    leaf_model.overrides['conf'] = 0.25
    leaf_model.overrides['iou'] = 0.45
    leaf_model.overrides['max_det'] = 1000
    
    # Load disease classification model
    disease_model = CNN_NeuralNet()
    disease_model.load_state_dict(torch.load('plant_disease_model.pth', map_location=device))
    disease_model.to(device)
    disease_model.eval()
    
    print("Both models loaded successfully")
    return leaf_model, disease_model

# ----------------------------
# 3. Image Processing Functions
# ----------------------------
def fetch_image_from_esp32():
    try:
        response = requests.get(esp32_cam_url, timeout=3)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to ESP32-CAM. Please check the URL and connection.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None

def detect_leaves(image, leaf_model):
    # Resize image for faster processing
    resized_image = cv2.resize(image, (640, 480))
    results = leaf_model(resized_image, stream=True)
    return next(results)

def classify_disease(cropped_leaf, disease_model):
    # Transform the cropped leaf for the disease model
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        outputs = disease_model(tensor)
        probs = F.softmax(outputs, dim=1)
    
    top_prob, top_idx = torch.max(probs, 1)
    predicted_class = CLASS_NAMES[top_idx.item()]
    confidence = top_prob.item()
    
    return predicted_class, confidence

# ----------------------------
# 4. Main Processing Loop
# ----------------------------
def main():
    # Load models
    leaf_model, disease_model = load_models()
    
    max_retries = 3
    retry_count = 0
    processed_boxes = {}  # Dictionary to track processed boxes and their timestamps
    classification_cooldown = .5  # seconds between classifications for the same region
    
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
                    
                time.sleep(1)  # Wait 1 second before retrying
                continue
            
            # Reset retry count on successful image fetch
            retry_count = 0
            
            # Step 1: Detect leaves in the image
            results = detect_leaves(image, leaf_model)
            
            # Create a copy of the image for display
            display_img = image.copy()
            
            current_time = time.time()
            # Clean up old entries from processed_boxes
            processed_boxes = {k: v for k, v in processed_boxes.items() 
                              if current_time - v < classification_cooldown}
            
            # Process each detected leaf
            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                
                for box, confidence in zip(boxes, confidences):
                    if float(confidence) > 0.50:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Create a box identifier (could be improved with tracking algorithms)
                        box_id = f"{x1}_{y1}_{x2}_{y2}"
                        
                        # Crop the leaf from the original image
                        cropped_leaf = image[y1:y2, x1:x2]
                        
                        # Check if we have a valid crop and if this box hasn't been processed recently
                        if cropped_leaf.size > 0 and box_id not in processed_boxes:
                            # Step 2: Classify disease on the cropped leaf
                            disease_class, disease_conf = classify_disease(cropped_leaf, disease_model)
                            processed_boxes[box_id] = current_time
                            
                            # Print disease detection to terminal
                            plant_type = disease_class.split('___')[0]
                            condition = disease_class.split('___')[-1]
                            print(f"[DETECTION] Plant: {plant_type} | Condition: {condition} | Confidence: {disease_conf:.4f}")
                            
                            # Draw bounding box and labels
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_img, f"Leaf: {confidence:.2f}", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(display_img, f"Disease: {disease_class.split('___')[-1]}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Just draw the leaf detection
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_img, f"Leaf: {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the results
            cv2.imshow("Plant Disease Detection", display_img)
            
            # Check for 'q' key press to exit
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