import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json, os
from datetime import datetime

print("=" * 50)
print("   ENTRAINEMENT DU CNN")
print("=" * 50)

# 1. Préparer les données
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.ImageFolder("data/train", transform=transform)
test_data  = datasets.ImageFolder("data/test",  transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=16, shuffle=False)

classes = train_data.classes
print(f"Classes : {classes}")
print(f"Train   : {len(train_data)} images")
print(f"Test    : {len(test_data)} images")

# 2. Définir le modèle CNN
class DocumentCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = DocumentCNN(num_classes=len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Entraînement
print("\nEntrainement en cours...")
history = []

for epoch in range(10):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1:2d}/10 | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.1f}%")
    history.append({"epoch": epoch+1, "loss": round(total_loss/len(train_loader), 4), "accuracy": round(acc, 1)})

# 4. Evaluation sur le test
print("\nEvaluation sur le test...")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"Accuracy finale sur test : {test_acc:.1f}%")

# 5. Sauvegarder
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/document_cnn.pt")
with open("models/classes.json", "w") as f:
    json.dump(classes, f)
with open("models/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nModele sauvegarde dans models/document_cnn.pt")
print("=" * 50)
print(f"   Accuracy finale : {test_acc:.1f}%")
print("=" * 50)