import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.classifier_agent import classifier_agent
from agents.summarizer_agent import summarizer_agent

class SmartDocApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Document Analyst")
        self.root.geometry("700x600")
        self.root.configure(bg="#1e1e2e")

        # Titre
        tk.Label(root, text="Smart Document Analyst",
                 font=("Arial", 20, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=20)

        # Bouton choisir document
        tk.Button(root, text="Choisir un document",
                  command=self.choisir_document,
                  font=("Arial", 12), bg="#89b4fa", fg="#1e1e2e",
                  padx=20, pady=10, cursor="hand2").pack(pady=10)

        # Label fichier choisi
        self.label_fichier = tk.Label(root, text="Aucun document choisi",
                                       font=("Arial", 10), bg="#1e1e2e", fg="#a6adc8")
        self.label_fichier.pack()

        # Bouton analyser
        self.btn_analyser = tk.Button(root, text="Analyser",
                                       command=self.analyser,
                                       font=("Arial", 12, "bold"),
                                       bg="#a6e3a1", fg="#1e1e2e",
                                       padx=20, pady=10,
                                       cursor="hand2", state="disabled")
        self.btn_analyser.pack(pady=10)

        # Zone résultat
        tk.Label(root, text="Résultat :", font=("Arial", 11, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(anchor="w", padx=20)

        self.zone_resultat = scrolledtext.ScrolledText(
            root, font=("Courier", 10),
            bg="#313244", fg="#cdd6f4",
            height=20, wrap=tk.WORD
        )
        self.zone_resultat.pack(fill="both", expand=True, padx=20, pady=10)

        self.image_path = None

    def choisir_document(self):
        path = filedialog.askopenfilename(
            title="Choisir un document",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("Tous", "*.*")]
        )
        if path:
            self.image_path = path
            nom = os.path.basename(path)
            self.label_fichier.config(text=f"Document : {nom}", fg="#a6e3a1")
            self.btn_analyser.config(state="normal")
            self.zone_resultat.delete(1.0, tk.END)
            self.zone_resultat.insert(tk.END, f"Document prêt : {nom}\n\nClique sur Analyser !")

    def analyser(self):
        self.btn_analyser.config(state="disabled", text="Analyse en cours...")
        self.zone_resultat.delete(1.0, tk.END)
        self.zone_resultat.insert(tk.END, "Analyse en cours, patiente...\n")
        threading.Thread(target=self.lancer_analyse, daemon=True).start()

    def lancer_analyse(self):
        try:
            self.log("Agent 1 : Classification du document...")
            result = classifier_agent(self.image_path)

            if "erreur" in result:
                self.log(f"ERREUR : {result['erreur']}")
                return

            categorie = result["categorie"]
            confiance = result["confiance"]

            self.log(f"Catégorie : {categorie}")
            self.log(f"Confiance : {confiance}%")
            self.log(f"\nExplication : {result['explication']}")
            self.log("\nScores par catégorie :")
            for cat, score in result["scores_complets"].items():
                self.log(f"  {cat:15} : {score}%")

            # Validation humaine via popup
            confirme = self.root.after(0, lambda: self.validation_humaine(categorie, confiance, result))

        except Exception as e:
            self.log(f"ERREUR : {str(e)}")
            self.root.after(0, lambda: self.btn_analyser.config(state="normal", text="Analyser"))

    def validation_humaine(self, categorie, confiance, result):
        reponse = messagebox.askyesno(
            "Validation Humaine",
            f"Le CNN a classifié ce document comme :\n\n"
            f"Catégorie : {categorie}\n"
            f"Confiance : {confiance}%\n\n"
            f"Confirmez-vous cette classification ?"
        )

        if reponse:
            self.log("\nValidation humaine : CONFIRMÉ")
            threading.Thread(target=self.generer_resume, args=(categorie,), daemon=True).start()
        else:
            categories = ["article", "facture", "formulaire", "lettre"]
            self.log("\nValidation humaine : REFUSÉ")
            self.choisir_categorie(categories)

    def choisir_categorie(self, categories):
        win = tk.Toplevel(self.root)
        win.title("Corriger la catégorie")
        win.geometry("300x200")
        win.configure(bg="#1e1e2e")

        tk.Label(win, text="Choisir la bonne catégorie :",
                 bg="#1e1e2e", fg="#cdd6f4", font=("Arial", 11)).pack(pady=10)

        var = tk.StringVar(value=categories[0])
        for cat in categories:
            tk.Radiobutton(win, text=cat, variable=var, value=cat,
                          bg="#1e1e2e", fg="#cdd6f4",
                          selectcolor="#313244").pack()

        def confirmer():
            categorie_corrigee = var.get()
            win.destroy()
            self.log(f"Catégorie corrigée : {categorie_corrigee}")
            threading.Thread(target=self.generer_resume,
                           args=(categorie_corrigee,), daemon=True).start()

        tk.Button(win, text="Confirmer", command=confirmer,
                 bg="#89b4fa", fg="#1e1e2e").pack(pady=10)

    def generer_resume(self, categorie):
        self.log("\nAgent 2 : Génération du résumé...")
        result = summarizer_agent(self.image_path, categorie)
        self.log(f"\nRAPPORT FINAL\n{'='*40}")
        self.log(result["analyse"])
        self.root.after(0, lambda: self.btn_analyser.config(
            state="normal", text="Analyser"))

    def log(self, texte):
        self.root.after(0, lambda: self.zone_resultat.insert(tk.END, texte + "\n"))
        self.root.after(0, lambda: self.zone_resultat.see(tk.END))


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartDocApp(root)
    root.mainloop()