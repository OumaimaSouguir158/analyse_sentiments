"""
train_and_evaluate.py — Projet 1 : Système de recommandation (filtrage collaboratif SVD)

Entièrement exécutable en local sur un PC modeste (i3, 8 Go RAM) :
dataset léger (100k notes) et algorithme SVD peu coûteux.

Utilisation :
    python train_and_evaluate.py
"""

import pickle
from surprise import SVD, Dataset
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

MODEL_PATH = "modele_svd.pkl"


def main():
    print("Chargement du dataset MovieLens 100k (téléchargement automatique si absent)...")
    data = Dataset.load_builtin("ml-100k")

    print("\n=== Validation croisée (5 folds) ===")
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    cv_results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

    print("\nRMSE moyen (validation croisée) :", cv_results["test_rmse"].mean())
    print("MAE moyen (validation croisée) :", cv_results["test_mae"].mean())

    print("\n=== Entraînement final sur train/test split (80/20) ===")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    final_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    final_model.fit(trainset)

    predictions = final_model.test(testset)
    print("RMSE (test set) :", accuracy.rmse(predictions))
    print("MAE (test set) :", accuracy.mae(predictions))

    # Sauvegarde du modèle final (ré-entraîné sur l'ensemble complet pour la production)
    full_trainset = data.build_full_trainset()
    production_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    production_model.fit(full_trainset)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(production_model, f)

    print(f"\nModèle sauvegardé dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()
