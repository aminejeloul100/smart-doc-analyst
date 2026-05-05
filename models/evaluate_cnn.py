import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

# Charger les classes
with open("models/classes.json") as f:
    classes = json.load(f)

# Modèle
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
            nn.Linear(256, len(classes))
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = DocumentCNN(num_classes=len(classes))
model.load_state_dict(torch.load("models/document_cnn.pt"))
model.eval()

# Données test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_data = datasets.ImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16)

# Matrice de confusion
matrix = [[0]*len(classes) for _ in range(len(classes))]

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(1)
        for true, pred in zip(labels, preds):
            matrix[true][pred] += 1

# Affichage
print("\nMATRICE DE CONFUSION")
print("-" * 45)
print(f"{'':12}", end="")
for c in classes:
    print(f"{c:12}", end="")
print()
for i, row in enumerate(matrix):
    print(f"{classes[i]:12}", end="")
    for val in row:
        print(f"{val:12}", end="")
    print()

print("\nRESULTATS PAR CLASSE")
print("-" * 45)
for i, c in enumerate(classes):
    correct = matrix[i][i]
    total = sum(matrix[i])
    print(f"{c:12} : {correct}/{total} correct ({100*correct//total}%)")

print("\nModele pret pour etre utilise dans les agents !")