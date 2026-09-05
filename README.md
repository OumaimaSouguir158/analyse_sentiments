# analyse_sentiments

# Projet 2 — Analyse de sentiments par Fine-Tuning de DistilBERT (NLP)

## 🎯 Objectif
Classifier un texte (avis, commentaire, tweet) comme **positif** ou **négatif** en fine-tunant **DistilBERT**, une version allégée de BERT (~40 % plus petite, ~60 % plus rapide, tout en conservant ~97 % de ses performances).

## 📁 Structure du projet
```
projet2_analyse_sentiments/
├── README.md                # ce fichier
├── rapport_projet.md        # rapport académique complet
├── requirements.txt         # dépendances
├── train_colab.ipynb        # notebook d'entraînement (Google Colab)
└── inference_local.py       # script d'inference locale (CPU, sur ton PC)
```

## 📦 Dataset
**IMDB Movie Reviews** (50 000 avis de films, positif/négatif), chargé directement via la bibliothèque `datasets` de Hugging Face — pas de téléchargement manuel nécessaire.

## 🚀 Étape 1 — Fine-tuning sur Google Colab
1. Ouvre `train_colab.ipynb` dans Google Colab.
2. Active le GPU : `Exécution > Modifier le type d'exécution > GPU (T4)`.
3. Exécute les cellules dans l'ordre :
   - chargement du dataset IMDB ;
   - tokenization avec le tokenizer DistilBERT ;
   - chargement du modèle pré-entraîné `distilbert-base-uncased` ;
   - fine-tuning avec `Trainer` (Hugging Face) ;
   - évaluation (accuracy, F1-score, matrice de confusion) ;
   - **sauvegarde du modèle fine-tuné** avec `save_pretrained()`.
4. Compresse le dossier du modèle sauvegardé et télécharge-le sur ton PC (dans le même dossier que `inference_local.py`).

## 💻 Étape 2 — Inference locale (ton PC i3 / 8 Go RAM)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python inference_local.py --text "This movie was absolutely amazing!"
```
DistilBERT (~260 Mo en mémoire, CPU) tourne correctement pour de l'inference ponctuelle sur un i3 avec 8 Go de RAM — prévoir quelques secondes par prédiction (pas de traitement en temps réel massif).

## ⚠️ Notes matérielles
- **Ne jamais fine-tuner en local** : même DistilBERT est trop lent à entraîner sur CPU pour un dataset de 50k exemples — cette étape reste réservée à Colab.
- Si l'inference locale te semble encore trop lente/gourmande en RAM, une option supplémentaire (mentionnée dans le rapport) consiste à exporter le modèle en **ONNX** avec quantification dynamique pour réduire encore la charge CPU/RAM.
- Pas besoin de stocker le dataset IMDB en local : il n'est utilisé que pendant l'entraînement sur Colab.

## 📊 Résultats
Voir `rapport_projet.md`, section "Résultats et évaluation" — à compléter avec tes propres chiffres après exécution du notebook.
