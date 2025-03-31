# Plant Disease Detection System 🌱🔬

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO-blueviolet.svg)](https://ultralytics.com/)

This project implements a real-time plant disease detection system using computer vision. It leverages a two-stage approach:

1.  **Leaf Detection:** A YOLO (You Only Look Once) model (`leaf_detection_model.pt`) identifies the location of plant leaves within an input image or video stream.
2.  **Disease Classification:** A custom Convolutional Neural Network (CNN) (`plant_disease_model.pth`) analyzes the detected leaf regions to classify the specific plant disease (or determine if the leaf is healthy) from a set of 38 possible classes.

The system is designed to work with input from an ESP32-CAM module and optionally fetch sensor data (GPS, Temperature, Humidity) from another ESP32 module, providing a practical solution for monitoring plant health directly in the field or greenhouse. It offers an interactive web-based GUI built with Streamlit (`1_Live_Feed.py`, `2_Results_Analysis.py`) and a command-line interface with OpenCV visualization (`integrated_detection.py`).

## ✨ Features

*   **Real-time Leaf Detection:** Utilizes a pre-trained YOLO model (`leaf_detection_model.pt`) for efficient localization of plant leaves.
*   **Multi-Class Disease Classification:** Employs a custom CNN (`plant_disease_model.pth`) trained to identify 38 different plant disease categories across various plant species (Apples, Corn, Grapes, Tomatoes, etc.).
*   **ESP32-CAM Integration:** Directly fetches image streams from an ESP32-CAM module over Wi-Fi for live analysis.
*   **Optional Sensor Integration:** Fetches GPS, temperature, and humidity data from a separate ESP32 sensor module (if configured).
*   **Multi-Page Streamlit Web Interface:** Provides an easy-to-use web GUI:
    *   **`1_Live_Feed.py`:**
        *   Starts/Stops the live detection stream from the ESP32-CAM.
        *   Visualizes the live feed with bounding boxes and disease predictions.
        *   Displays real-time sensor data (GPS, Temp/Humidity) in the sidebar.
        *   Logs detection events (timestamp, plant type, condition, confidence, GPS) to a CSV file.
        *   Allows configuration of ESP32 URLs and leaf detection confidence threshold.
    *   **`2_Results_Analysis.py`:**
        *   Reads and displays the full detection log from the CSV file.
        *   Shows summary statistics (total detections, counts by plant/condition, etc.).
        *   Visualizes detection trends over time and condition breakdowns.
        *   Plots detection locations on an interactive map using GPS coordinates.
        *   Allows downloading the detection log.
*   **OpenCV Interface (`integrated_detection.py`):** Offers a command-line alternative for running the integrated detection pipeline and visualizing results in a standard OpenCV window (Note: may not include sensor integration or logging as implemented in the Streamlit app).
*   **Standalone Leaf Detection (`leaf_detection_v1.py`):** Script dedicated solely to testing and visualizing the YOLO leaf detection model using an ESP32-CAM feed.
*   **Standalone Disease Prediction (`disease_detection_v2.py`):** Contains the CNN model definition and basic functionality to predict disease from a single image file.
*   **Result Logging:** Saves detection events (timestamp, plant type, condition, confidence, latitude, longitude, gps\_valid) to `results/detection_results.csv` via the `1_Live_Feed.py` Streamlit page.
*   **GPU Acceleration:** Supports CUDA acceleration via PyTorch and Ultralytics if a compatible NVIDIA GPU is available, significantly speeding up inference.

## ⚙️ Technology Stack

*   **Programming Language:** Python 3.8+
*   **Deep Learning Framework:** PyTorch
*   **Object Detection:** Ultralytics YOLO (using `leaf_detection_model.pt`)
*   **Web Framework (GUI):** Streamlit
*   **Computer Vision:** OpenCV (cv2)
*   **Image Handling:** PIL (Pillow)
*   **Numerical Computation:** NumPy
*   **Data Handling & Analysis:** Pandas (used heavily in `2_Results_Analysis.py`)
*   **HTTP Requests:** Requests (for fetching images/data from ESP32 modules)
*   **Hardware (Optional):**
    *   NVIDIA GPU with CUDA and cuDNN for faster processing.
    *   ESP32-CAM module for live image input.
    *   ESP32 module with GPS/DHT sensors for sensor data input.

## 🏛️ System Architecture

The core workflow involves these steps:

1.  **Image & Sensor Acquisition (in `1_Live_Feed.py`):**
    *   An image frame is fetched from the ESP32-CAM URL.
    *   Sensor data (GPS, Temp/Humidity) is fetched from the sensor module URL.
2.  **Leaf Detection (YOLO):** The captured image is passed to the `leaf_detection_model.pt`. The model outputs bounding boxes coordinates and confidence scores for detected leaves.
3.  **Leaf Cropping:** For each detected leaf exceeding a confidence threshold, the corresponding region (bounding box) is cropped from the original image.
4.  **Disease Classification (CNN):** The cropped leaf image is preprocessed (resized, converted to tensor) and fed into the `plant_disease_model.pth`.
5.  **Prediction Output:** The CNN outputs probabilities for each of the 38 disease classes. The class with the highest probability is selected as the prediction.
6.  **Visualization, Logging & Analysis:**
    *   **Live Feed (`1_Live_Feed.py`):**
        *   Bounding boxes, detected leaf confidence, and predicted disease labels are drawn onto the original image.
        *   The annotated image is displayed in the Streamlit interface.
        *   Live sensor data is shown in the sidebar.
        *   Detection results (timestamp, plant, condition, confidence, GPS data) are logged to `results/detection_results.csv`.
    *   **Results Analysis (`2_Results_Analysis.py`):**
        *   Reads the `results/detection_results.csv` file.
        *   Displays the data log, statistics, charts, and a map based on the logged information.
    *   **OpenCV (`integrated_detection.py`):** Displays annotated images in an OpenCV window.

## 🔧 Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher
    *   `pip` (Python package installer)
    *   Git (for cloning the repository)
    *   **(Optional but Recommended)** NVIDIA GPU with CUDA and cuDNN drivers installed for significant performance improvement.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/SHNwastaken/Plant_Disease_Detection.git
    cd Plant_Disease_Detection
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    *   Ensure the `requirements.txt` file provided in the repository is up-to-date. It should contain libraries like: `torch`, `torchvision`, `ultralytics`, `opencv-python-headless`, `streamlit`, `pandas`, `requests`, `Pillow`, `numpy`.
    *   Install the requirements:
        ```bash
        pip install -r requirements.txt
        ```
        *(Note: Installing PyTorch might require specific commands depending on your OS and CUDA version. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command if you encounter issues.)*

5.  **Ensure Model Files are Present:**
    *   The `leaf_detection_model.pt` (YOLO) and `plant_disease_model.pth` (CNN) files should be in the root directory.

6.  **Prepare Results Directory:**
    *   The `1_Live_Feed.py` script will attempt to create a `results` directory if it doesn't exist, to store `detection_results.csv`. Ensure you have write permissions in the project directory.

7.  **Configure ESP32 URLs:**
    *   Find the IP addresses of your ESP32-CAM module and your ESP32 sensor module (if used) on your local network.
    *   **For the Streamlit App (`1_Live_Feed.py`):** You will enter the URLs directly into the text input fields in the Streamlit sidebar when you run the application. Default URLs are provided in the script but can be overridden in the UI.
        *   ESP32-CAM URL Example: `http://192.168.1.100/capture`
        *   Sensor Module Base URL Example: `http://192.168.1.101` (The script will append `/gps` or the relevant endpoint).
    *   **For Command-Line Scripts (`integrated_detection.py`, `leaf_detection_v1.py`):** You may need to manually edit the `esp32_cam_url` variable within these Python files if you intend to use them.

## ▶️ Usage

Make sure your virtual environment is activated (`source venv/bin/activate` or `.\venv\Scripts\activate`) and you are in the project's root directory. Ensure the ESP32 modules are powered on and connected to the same network if you intend to use them.

1.  **Run the Streamlit Web Interface (Recommended):**
    *   Provides the most user-friendly experience with live feed, sensor data, logging, and results analysis.
    *   Navigate to the project directory in your terminal.
    *   Run the following command:
        ```bash
        streamlit run 1_Live_Feed.py
        ```
    *   Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
    *   The application will start on the "Live Feed" page (`1_Live_Feed.py`).
    *   Use the sidebar controls to enter the ESP32 URLs, adjust confidence, and start/stop the stream.
    *   Navigate to the "Results Analysis" page (`2_Results_Analysis.py`) using the Streamlit sidebar navigation to view logged data, statistics, and the map.

2.  **Run the Integrated Detection with OpenCV:**
    *   Useful for testing or if a web interface is not needed. Displays results in a desktop window. (Note: Check the script for specific features like sensor integration/logging).
    ```bash
    python integrated_detection.py
    ```
    *   An OpenCV window will appear displaying the live feed with detections.
    *   Press `q` in the OpenCV window to quit.
    *   Detection details might be printed to the console.

3.  **Run Standalone Leaf Detection (YOLO Only):**
    *   For testing the `leaf_detection_model.pt` specifically with the ESP32-CAM feed.
    ```bash
    python leaf_detection_v1.py
    ```
    *   An OpenCV window will show the live feed with only leaf bounding boxes.
    *   Press `q` in the OpenCV window to quit.

4.  **Run Basic Disease Prediction on Static Images:**
    *   Uses the `disease_detection_v2.py` script for classifying *existing* image files.
    *   Modify the `test_images` list inside the script if needed.
    ```bash
    python disease_detection_v2.py
    ```
    *   Predictions for the specified images will be printed to the console.

## 🧠 Models

1.  **Leaf Detection Model (`leaf_detection_model.pt`):**
    *   **Type:** YOLO (Ultralytics format) Object Detection Model.
    *   **Purpose:** Detects the location (bounding boxes) of plant leaves in an image.
    *   **Input:** Image (BGR format expected by OpenCV/Ultralytics).
    *   **Output:** List of detected bounding boxes (coordinates), confidence scores, and class labels (presumably just 'leaf').

2.  **Disease Classification Model (`plant_disease_model.pth`):**
    *   **Type:** Custom Convolutional Neural Network (defined in `disease_detection_v2.py`). Uses Conv2D layers, BatchNorm, ReLU, Max Pooling, and Residual connections.
    *   **Purpose:** Classifies the disease present on a *cropped* image of a single leaf.
    *   **Input:** 3-channel RGB image tensor of size (256, 256).
    *   **Output:** Logits/probabilities for 38 distinct classes.
    *   **Classes:** The model distinguishes between the following 38 plant/disease combinations:
        ```
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
        ```

## 📊 CSV Output (`results/detection_results.csv`)

When using the Streamlit application (`1_Live_Feed.py`), detection results are appended to this file located in the `results` subdirectory.

*   **`timestamp`**: ISO format timestamp when the detection occurred.
*   **`plant_type`**: The type of plant detected (e.g., 'Apple', 'Tomato'). Extracted from the predicted class name.
*   **`condition`**: The detected condition (disease name or 'healthy'). Extracted from the predicted class name.
*   **`confidence`**: The confidence score (probability, 0.0 to 1.0) from the CNN model for the predicted class.
*   **`latitude`**: Latitude coordinate from the sensor module (if available and valid).
*   **`longitude`**: Longitude coordinate from the sensor module (if available and valid).
*   **`gps_valid`**: Boolean (`True`/`False`) indicating if the GPS data for this detection was considered valid by the sensor module.

*(Note: The CSV file might also contain other sensor columns like temperature, humidity, altitude, etc., depending on the exact implementation in `1_Live_Feed.py`'s `save_to_csv` function, but the columns listed above are the primary ones used for analysis in `2_Results_Analysis.py`)*

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please ensure your code follows standard Python practices and includes appropriate documentation or comments.

## 🙏 Acknowledgements

*   Plant Disease Dataset: [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
*   Ultralytics for YOLO models.
*   The Streamlit team for the web framework.
*   The PyTorch team for the deep learning library.
