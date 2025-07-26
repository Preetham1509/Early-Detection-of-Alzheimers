import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# -------------------------------
# Function to unzip the dataset
# -------------------------------
def unzip_dataset(zip_path, extract_to):
    """
    Extracts a zip file containing the dataset.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory where data should be extracted.
    """
    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[INFO] Extraction complete.")

# -----------------------------------------
# Function to prepare data loaders
# -----------------------------------------
def prepare_data(data_dir, img_size=(150, 150), batch_size=32):
    """
    Prepares training and validation data loaders.

    Args:
        data_dir (str): Root directory of dataset.
        img_size (tuple): Target image size for resizing.
        batch_size (int): Number of samples per batch.

    Returns:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        class_names (list): List of class names.
    """
    print("[INFO] Preparing data...")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    # Load dataset from directory
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split into training and validation sets (80-20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get class labels
    class_names = train_dataset.dataset.classes
    return train_loader, val_loader, class_names

# -----------------------------------------
# CNN Model for Image Classification
# -----------------------------------------
class CNNModel(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.
    """
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 37 * 37, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------------------
# Training function
# -----------------------------------------
def train_model(model, train_loader, val_loader, epochs=10):
    """
    Trains the CNN model on the dataset.

    Args:
        model (nn.Module): CNN model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of training epochs.
    """
    print("[INFO] Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Save model to disk
    torch.save(model.state_dict(), "alzheimers_model.pth")
    print("[INFO] Model saved as 'alzheimers_model.pth'")

# -----------------------------------------
# Prediction function for single image
# -----------------------------------------
def predict_image(model, image_path, class_names, img_size=(150, 150)):
    """
    Predicts the class of a single input image.

    Args:
        model (nn.Module): Trained CNN model.
        image_path (str): Path to the image to predict.
        class_names (list): List of class labels.
        img_size (tuple): Size to which the image should be resized.

    Returns:
        predicted_class (str): Predicted class label.
        confidence (float): Confidence of the prediction.
    """
    print(f"[INFO] Predicting image: '{image_path}'")
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()
        predicted_class = class_names[predicted_index]

        print(f"🧠 Predicted Stage: {predicted_class} ({confidence * 100:.2f}% confidence)")
        return predicted_class, confidence

# -----------------------------------------
# Main script to run the pipeline
# -----------------------------------------
if __name__ == '__main__':
    zip_path = "C:/Users/Acer/Downloads/archive (2).zip"          # Path to ZIP file
    extract_path = 'alzheimers_dataset'                           # Directory to extract to
    image_to_predict = input("Enter the path to the image you want to classify: ")                              # Image for prediction

    # Step 1: Unzip dataset
    unzip_dataset(zip_path, extract_path)

    # Step 2: Prepare data loaders
    data_path = os.path.join(extract_path, "AugmentedAlzheimerDataset")
    train_loader, val_loader, class_names = prepare_data(data_path)

    # Step 3: Initialize and train model
    model = CNNModel(num_classes=len(class_names))
    train_model(model, train_loader, val_loader)

    # Step 4: Load model and make prediction
    model.load_state_dict(torch.load('alzheimers_model.pth'))
    predict_image(model, image_to_predict, class_names)
