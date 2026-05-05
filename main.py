import sys
import os
import json
import logging
from datetime import datetime
from agents.classifier_agent import classifier_agent
from agents.summarizer_agent import summarizer_agent

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agents.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

def orchestrateur(image_path: str):
    """Orchestrateur principal du systeme multi-agents."""

    print("\n" + "="*55)
    print("   SMART DOCUMENT ANALYST - Systeme Multi-Agents")
    print("="*55)
    print(f"Document : {image_path}")
    print("="*55)

    # AGENT 1 : Classification
    resultat_classification = classifier_agent(image_path)

    if "erreur" in resultat_classification:
        print(f"\nERREUR : {resultat_classification['erreur']}")
        return

    categorie = resultat_classification["categorie"]
    confiance = resultat_classification["confiance"]

    # HUMAN IN THE LOOP
    print("\n" + "-"*55)
    print("   VALIDATION HUMAINE REQUISE")
    print("-"*55)
    print(f"Le CNN a classifie ce document comme : {categorie}")
    print(f"Confiance : {confiance}%")
    print(f"\nExplication : {resultat_classification['explication']}")
    print("\nScores par categorie :")
    for cat, score in resultat_classification["scores_complets"].items():
        print(f"  {cat:15} : {score}%")

    print("\nConfirmez-vous cette classification ?")
    print("  [1] Oui, continuer")
    print("  [2] Non, corriger")
    print("  [3] Annuler")

    choix = input("\nVotre choix (1/2/3) : ").strip()

    if choix == "3":
        print("\nAnalyse annulee par l'utilisateur.")
        logging.info("Analyse annulee par utilisateur")
        return

    if choix == "2":
        print("\nCategories disponibles : article, facture, formulaire, lettre")
        categorie = input("Entrez la bonne categorie : ").strip()
        print(f"Categorie corrigee : {categorie}")
        logging.info(f"Categorie corrigee par humain: {categorie}")

    logging.info(f"Validation humaine: OK - categorie={categorie}")

    # AGENT 2 : Resume
    resultat_resume = summarizer_agent(image_path, categorie)

    # RAPPORT FINAL
    print("\n" + "="*55)
    print("   RAPPORT FINAL")
    print("="*55)
    print(f"Type de document : {categorie}")
    print(f"Confiance CNN    : {confiance}%")
    print(f"\nAnalyse :\n{resultat_resume['analyse']}")

    # Sauvegarder le rapport
    rapport = {
        "timestamp": datetime.now().isoformat(),
        "document": image_path,
        "categorie": categorie,
        "confiance": confiance,
        "analyse": resultat_resume["analyse"]
    }
    rapport_path = f"logs/rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(rapport_path, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    print(f"\nRapport sauvegarde : {rapport_path}")
    print("="*55)

if __name__ == "__main__":
    # Tester avec une image du dataset
    image_test = "data/test/facture/doc_0.png"
    orchestrateur(image_test)