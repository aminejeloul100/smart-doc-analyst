# Smart Document Analyst 🤖

Système multi-agents d'analyse automatique de documents.

## Description
Ce projet utilise un CNN (PyTorch) et un LLM (Ollama/Gemma3) pour classifier et résumer automatiquement des documents.

## Architecture
- **Agent 1** : Classificateur (CNN PyTorch entraîné)
- **Agent 2** : Résumeur (Gemma3 via Ollama)
- **Orchestrateur** : Coordonne les agents + validation humaine

## Installation

```bash
git clone https://github.com/aminejeloul100/smart-doc-analyst.git
cd smart-doc-analyst
python -m venv venv
venv\Scripts\activate
pip install torch torchvision pillow ollama crewai
ollama pull gemma3:1b
```

## Utilisation

```bash
# Interface graphique
python interface.py

# Ligne de commande
python main.py
```

## Entraîner le modèle

```bash
python data\download_dataset.py
python models\train_cnn.py
```

## Résultats
- Accuracy CNN : **100%** sur 4 catégories
- Catégories : facture, lettre, formulaire, article
- Dataset : 400 images synthétiques

## Technologies
- Python 3.11
- PyTorch 2.11
- CrewAI
- Ollama (Gemma3:1b)
- Tkinter