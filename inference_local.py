"""
inference_local.py — Projet 2 : Analyse de sentiments (DistilBERT fine-tuné)
Inference CPU, conçue pour tourner sur un PC modeste (Intel i3, 8 Go RAM).

Utilisation :
    python inference_local.py --text "This movie was absolutely amazing!"

Prérequis :
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers

Le dossier du modèle fine-tuné (téléchargé depuis Colab, ex: "modele_sentiment_distilbert")
doit se trouver dans le même répertoire que ce script.
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "modele_sentiment_distilbert"
LABELS = {0: "négatif", 1: "positif"}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()  # mode inference (désactive dropout, etc.)
    return tokenizer, model


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():  # pas de calcul de gradients -> économise RAM et CPU
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_class = int(torch.argmax(probs))
    confidence = float(probs[predicted_class])

    return LABELS[predicted_class], confidence, {LABELS[i]: float(p) for i, p in enumerate(probs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyser le sentiment d'un texte.")
    parser.add_argument("--text", required=True, help="Texte à analyser")
    args = parser.parse_args()

    tokenizer, model = load_model()
    label, confidence, all_scores = predict(args.text, tokenizer, model)

    print(f"\nTexte : {args.text}")
    print(f"Sentiment prédit : {label}")
    print(f"Confiance : {confidence * 100:.2f}%\n")
    print("Détail des scores :")
    for classe, score in all_scores.items():
        print(f"  {classe:10s} : {score * 100:5.2f}%")
