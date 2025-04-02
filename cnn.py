from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
import csv


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder("squares/train/cropped", transform=transform)
test_ds = datasets.ImageFolder("squares/val/cropped", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

#define model Architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#Model training stage
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

torch.save(model.state_dict(), "CNN_model.pth")

# Evaluate accuracy
def evaluate(model, dataloader, class_names):
    model.eval()
    preds, labels = [], []
    incorrect_samples = []

    with torch.no_grad():
        for images, targets in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            preds.extend(predicted.numpy())
            labels.extend(targets.numpy())

            # Track incorrect predictions
            for i in range(len(targets)):
                if predicted[i] != targets[i]:
                    incorrect_samples.append({
                        'index': len(labels) - len(targets) + i,
                        'predicted': class_names[predicted[i]],
                        'actual': class_names[targets[i]]
                    })

    accuracy = sum(p == t for p, t in zip(preds, labels)) / len(labels)
    
    #Use SK-Learn evaluation packages
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    return incorrect_samples

incorrect_samples = evaluate(model, test_loader)

#save incorrect images for review later
output_csv = "misclassified_samples.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "actual_class", "predicted_class"])  # Header

    for sample in incorrect_samples:
        idx = sample["index"]
        filename, label = test_ds.samples[idx]
        actual_class = test_ds.classes[label]
        predicted_class = sample["predicted"]
        writer.writerow([filename, actual_class, predicted_class])

