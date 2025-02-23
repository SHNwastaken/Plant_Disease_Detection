import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 28 * 28)  
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        
        return x

# Initialize the model

model = PlantDiseaseCNN(3)
def test_single_image(model_path, image_path, class_labels=['Healthy', 'Powdery', 'Rust']):
    """
    Test a trained PyTorch model on a single image
    
    Args:
        model_path (str): Path to the saved model file (.pth)
        image_path (str): Path to the image file
        class_labels (list): Optional list of class labels
    """
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device)) #Load model to the correct device
    model.to(device)
    model.eval()
    
    # Define the same transforms used during training
    transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize images to 224x224
    transforms.RandomHorizontalFlip(),      # Random horizontal flip for augmentation
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    # Print results
    print(f"Predicted class: {class_labels[predicted_class] if class_labels else predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # If class labels are provided, show top-3 predictions
    if class_labels:
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        print("\nTop 3 predictions:")
        for i in range(3):
            class_idx = top3_indices[0][i].item()
            prob = top3_prob[0][i].item() * 100
            print(f"{class_labels[class_idx]}: {prob:.2f}%")
test_single_image("best_plant_disease_model.pth", "healthy.jpg")