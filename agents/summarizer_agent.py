import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ollama
import logging

def summarizer_agent(image_path: str, categorie: str) -> dict:
    """Agent 2 : extrait et resume le contenu du document."""

    print("\n[AGENT RESUMEUR] Extraction du contenu...")
    logging.info(f"Summarizer agent - categorie: {categorie}")

    prompt = f"""Tu es un expert en analyse de documents.
On t'a donne un document de type : {categorie}
Chemin du document : {image_path}

Basé sur le type de document, génère :
1. Un résumé professionnel (3 phrases)
2. Les points clés à retenir (3 points)
3. Une action recommandée

Reponds en francais, de facon structuree."""

    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}]
    )

    contenu = response["message"]["content"]
    print(f"  -> Resume genere avec succes")
    logging.info(f"Resume genere pour: {categorie}")

    return {
        "categorie": categorie,
        "analyse": contenu
    }