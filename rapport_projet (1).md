# Rapport de projet — Analyse de sentiments par Fine-Tuning de DistilBERT

**Module :** Intelligence Artificielle / Traitement Automatique du Langage Naturel (NLP)
**Type de projet :** Projet individuel — portfolio Master IA / Ingénieur IA
**Environnement d'exécution :** Fine-tuning sur Google Colab (GPU T4) — Inference sur PC local (Intel i3, 8 Go RAM, SSD)

---

## 1. Introduction et contexte

L'analyse de sentiments (ou *sentiment analysis*) est l'une des tâches fondamentales du traitement automatique du langage naturel (NLP). Elle consiste à déterminer automatiquement la polarité (positive/négative, voire neutre) d'un texte : avis client, tweet, commentaire. Ses applications sont nombreuses : veille de réputation de marque, modération de contenu, analyse de retours utilisateurs à grande échelle.

Les modèles de type **Transformer** (BERT et ses variantes) ont considérablement amélioré l'état de l'art en NLP depuis 2018. Cependant, ces modèles sont coûteux en mémoire et en calcul, ce qui pose un défi lorsqu'on souhaite les utiliser sur un poste de travail aux ressources limitées.

## 2. Problématique et objectifs

**Problématique :** Comment tirer parti de la puissance des modèles Transformer pour l'analyse de sentiments tout en respectant une contrainte matérielle forte (i3, 8 Go RAM, pas de GPU) ?

**Objectifs :**
- Fine-tuner un modèle Transformer pré-entraîné sur une tâche de classification binaire de sentiments.
- Choisir une architecture volontairement allégée (**DistilBERT**) plutôt qu'un modèle BERT complet, pour limiter l'empreinte mémoire à l'inference.
- Déporter l'entraînement (coûteux) sur Google Colab, et ne conserver en local que l'inference (peu coûteuse).
- Évaluer une piste d'optimisation supplémentaire (export ONNX + quantification) pour encore réduire la charge sur un CPU modeste.

## 3. État de l'art (synthèse)

BERT (Devlin et al., 2018) a introduit les représentations contextuelles bidirectionnelles pré-entraînées, révolutionnant le NLP. Son coût computationnel élevé (110M+ paramètres) a motivé le développement de versions distillées, dont **DistilBERT** (Sanh et al., 2019), obtenu par *knowledge distillation* : un modèle "étudiant" plus petit est entraîné à reproduire le comportement du modèle "professeur" (BERT), ce qui permet de conserver environ 97 % des performances tout en réduisant la taille de 40 % et en accélérant l'inference de 60 %.

## 4. Méthodologie

### 4.1 Dataset
**IMDB Movie Reviews** — 50 000 avis de films en anglais, étiquetés positif/négatif, chargés via la bibliothèque `datasets` de Hugging Face. Un sous-ensemble (5 000 exemples d'entraînement, 2 000 de test) est utilisé pour limiter le temps d'entraînement sur Colab.

### 4.2 Prétraitement
- Tokenization avec le tokenizer natif de DistilBERT (`distilbert-base-uncased`), troncature à 256 tokens.
- Padding dynamique par batch via `DataCollatorWithPadding` (plus efficace qu'un padding fixe sur tout le dataset).

### 4.3 Architecture et fine-tuning
- Modèle de base : `distilbert-base-uncased` (66M paramètres) avec une tête de classification à 2 sorties (`AutoModelForSequenceClassification`).
- Fine-tuning complet du modèle (pas de gel de couches ici, car DistilBERT est déjà suffisamment léger) via l'API `Trainer` de Hugging Face.
- Hyperparamètres : taux d'apprentissage 2e-5, 3 epochs, batch size 16, weight decay 0.01.
- Métriques suivies : accuracy et F1-score.

### 4.4 Environnement d'exécution
| Phase | Environnement | Justification |
|---|---|---|
| Fine-tuning | Google Colab (GPU T4) | Même DistilBERT reste trop lent à entraîner sur CPU i3 pour un dataset de plusieurs milliers d'exemples |
| Export (optionnel) | Google Colab | Conversion en ONNX + quantification dynamique (int8) pour réduire davantage la charge à l'inference |
| Inference | PC local (i3, 8 Go RAM) | Modèle chargé en mode `eval()`, calcul sans gradient (`torch.no_grad()`), empreinte mémoire limitée à ~260 Mo |

## 5. Implémentation

Le code source est organisé en deux parties :
- `train_colab.ipynb` : notebook exécuté sur Google Colab, couvrant le chargement des données, la tokenization, le fine-tuning, l'évaluation, et la sauvegarde du modèle (avec une étape optionnelle d'export ONNX quantifié).
- `inference_local.py` : script Python autonome utilisant `transformers` et `torch` (CPU uniquement), chargeant le modèle fine-tuné sauvegardé pour analyser le sentiment d'un nouveau texte.

Le choix de désactiver le calcul de gradients (`torch.no_grad()`) pendant l'inference est déterminant pour limiter la consommation de RAM sur la machine locale.

## 6. Résultats et évaluation

*Section à compléter après exécution du notebook sur Google Colab :*

- Accuracy sur le set de test : **[à renseigner]**
- F1-score : **[à renseigner]**
- Matrice de confusion : **[voir sortie du notebook, section 7]**
- Temps d'inference moyen par texte sur CPU i3 : **[à mesurer, généralement 0.5 à 2 secondes selon la longueur du texte]**
- (Si export ONNX effectué) Gain de vitesse et réduction mémoire par rapport à la version PyTorch standard : **[à mesurer]**

> Remarque méthodologique : ces valeurs dépendent des hyperparamètres choisis et de la taille du sous-ensemble d'entraînement ; elles doivent être mesurées lors de l'exécution réelle et reportées ici pour constituer un rapport final complet.

## 7. Difficultés rencontrées et contraintes matérielles

- **RAM limitée (8 Go) :** impose de charger le modèle uniquement en mode inference (`eval()` + `no_grad()`), et de traiter les textes un par un ou en petits batches plutôt qu'en larges lots.
- **Absence de GPU local :** exclut totalement le fine-tuning en local, y compris pour un modèle "léger" comme DistilBERT sur un dataset de plusieurs milliers d'exemples.
- **Choix du sous-ensemble d'entraînement :** réduire le dataset à 5 000/2 000 exemples permet de garder un temps d'entraînement raisonnable sur le GPU gratuit de Colab (session limitée dans le temps), au prix d'une performance légèrement inférieure à un entraînement sur les 50 000 exemples complets.

## 8. Conclusion et perspectives

Ce projet illustre comment exploiter des modèles Transformer de pointe pour le NLP tout en respectant des contraintes matérielles fortes, grâce à une architecture distillée (DistilBERT), une séparation claire entre entraînement cloud et inference locale, et des techniques d'optimisation de l'inference (désactivation des gradients, quantification optionnelle).

**Perspectives d'amélioration :**
- Entraîner sur l'intégralité du dataset IMDB (50k exemples) pour de meilleures performances, si le temps de session Colab le permet.
- Étendre à une classification multi-classes (très négatif / négatif / neutre / positif / très positif).
- Adapter le pipeline à des données en français (ex: modèle `distilbert-base-multilingual-cased` ou `camembert-base`).
- Déployer le modèle derrière une petite API locale (FastAPI) ou une interface Streamlit pour une utilisation interactive.

## 9. Références

- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*
- Documentation Hugging Face Transformers : https://huggingface.co/docs/transformers
- Dataset IMDB : https://huggingface.co/datasets/imdb
