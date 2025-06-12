import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the same model architecture
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# === CONFIGURATION ===
IMAGE_PATH = "PublicTest_87012441.jpg"  # Change this to your image filename
MODEL_PATH = "emotion_model.pth"
CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Change if needed
IMG_SIZE = 48

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load and preprocess image
image = Image.open(IMAGE_PATH)
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Load model
model = EmotionCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Predict
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Predicted Emotion: {CLASSES[predicted_class]}")

