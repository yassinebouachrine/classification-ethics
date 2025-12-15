# Classification des Dilemmes Éthiques

**Description courte**
Un pipeline compact pour classifier des énoncés moraux (Éthique vs Non-Éthique) en utilisant le dataset `hendrycks/ethics (commonsense)`.
dataset : https://huggingface.co/datasets/hendrycks/ethics/tree/main/data/commonsense

**But**
Construire un pipeline fiable (prétraitement → TF‑IDF → sélection → réduction → features manuelles → classifieurs) tout en évitant les fuites de données (leakage).

---

## Prérequis

* Python 3.8+
* Packages : `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `joblib`, `matplotlib`, `seaborn`, `datasets`

Installer rapidement :

```bash
pip install numpy pandas scikit-learn xgboost lightgbm joblib matplotlib seaborn datasets
```

## Exécution

```bash
ethic_model.ipynb
```

Sorties : dossier `figures/`, `model_comparison_results.csv`, et le meilleur modèle sauvegardé (`best_ethics_model_*.pkl`).

## Structure du pipeline (résumé)

* **Prétraitement** : nettoyage léger, mise en minuscule, marquage des négations, suppression d'URLs.
* **Text features** : TF‑IDF → SelectKBest (chi²) → TruncatedSVD.
* **Features manuelles** : longueurs, compte de négations, mots moralement chargés, pronoms, modaux, ratio, etc., puis StandardScaler.
* **Fusion** : FeatureUnion (text + manuel) → Classifieur.
* **Validation** : StratifiedKFold (5 folds), métrique principale `f1_weighted`.

## Modèles testés

LogisticRegression, RidgeClassifier, LightGBM, RandomForest, XGBoost, GradientBoosting. Ensembles : Voting (top-3), AdaBoost.

## Anti‑leakage

Les transformeurs textuels (TF‑IDF, SelectKBest) sont fit uniquement à l'intérieur du pipeline / des folds pour éviter toute fuite de données.

## Fonction utile

`predict_ethics(text, return_proba=False)` — applique le prétraitement et renvoie la prédiction.

## Fichiers générés

* `figures/` : graphiques d'EDA et résultats
* `model_comparison_results.csv`
* `best_ethics_model_*.pkl` (+ transformeurs exportés si possible)

## Ce projet réalisé par: 
    - KHACHA Mohamed
    - NAHIMY Abdeljalil
    - BOUACHRINE Yassine
---
