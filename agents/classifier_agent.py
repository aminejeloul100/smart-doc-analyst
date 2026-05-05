import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.cnn_tool import classify_document
import ollama
import json
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agents.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

def classifier_agent(image_path: str) -> dict:
    """Agent 1 : classifie un document avec le CNN."""

    print("\n[AGENT CLASSIFICATEUR] Analyse du document...")
    logging.info(f"Classifier agent - image: {image_path}")

    # Etape 1 : CNN classifie l'image
    result = classify_document(image_path)

    if "erreur" in result:
        logging.error(f"Erreur CNN: {result['erreur']}")
        return {"erreur": result["erreur"]}

    categorie = result["categorie"]
    confiance = result["confiance"]

    print(f"  -> Categorie detectee : {categorie}")
    print(f"  -> Confiance          : {confiance}%")

    # Etape 2 : LLM explique le resultat
    prompt = f"""Tu es un assistant qui analyse des documents.
Le CNN a classifie ce document comme : {categorie} (confiance: {confiance}%)
Scores: {json.dumps(result['toutes_categories'])}

En 2 phrases courtes, explique ce que signifie cette classification."""

    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}]
    )
    explication = response["message"]["content"]

    logging.info(f"Classification: {categorie} ({confiance}%)")

    return {
        "categorie": categorie,
        "confiance": confiance,
        "explication": explication,
        "scores_complets": result["toutes_categories"]
    }