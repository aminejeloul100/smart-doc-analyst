import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

with open("models/classes.json") as f:
    CLASSES = json.load(f)

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

_model = DocumentCNN(num_classes=len(CLASSES))
_model.load_state_dict(torch.load("models/document_cnn.pt"))
_model.eval()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def classify_document(image_path: str) -> dict:
    """Classifie un document et retourne sa categorie et confiance."""
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = _transform(img).unsqueeze(0)
        with torch.no_grad():
            output = _model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = probs.argmax().item()
        return {
            "categorie": CLASSES[pred],
            "confiance": round(probs[pred].item() * 100, 2),
            "toutes_categories": {
                CLASSES[i]: round(probs[i].item() * 100, 2)
                for i in range(len(CLASSES))
            }
        }
    except Exception as e:
        return {"erreur": str(e)}