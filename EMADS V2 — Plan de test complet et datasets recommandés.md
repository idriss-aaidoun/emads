# EMADS V2 — Plan de test complet et datasets recommandés

---

## 1. Catégories de tests à effectuer

### 1.1 Tests unitaires par agent (isolés, sans lancer tout le pipeline)

Pour chaque agent, teste `execute()` directement avec un `EMADSState` construit à la
main, sans passer par le Supervisor :

| Agent | Ce qu'il faut vérifier |
|---|---|
| Data Understanding | Détection correcte de target (3 cas : fournie / nom reconnu / fallback), classification vs régression sur des targets synthétiques connues |
| EDA | Génération des plots sans crash, détection d'outliers sur des données avec outliers connus injectés manuellement |
| Preprocessing | Pas de NaN restant après exécution, colonnes constantes bien supprimées, encodage correct selon cardinalité |
| Model Selection | Le test de Wilcoxon se déclenche bien quand deux modèles ont des scores très proches (à forcer artificiellement) |
| Hyperparameter | Le retry se déclenche bien quand l'amélioration est sous le seuil ; le budget de 90s n'est jamais dépassé |
| Evaluation | IC 95% cohérent avec cv_mean/cv_std connus ; ROC AUC correct sur cas binaire ET multiclasse |
| Explainability | Permutation Importance + Spearman se calculent sans erreur ; l'arbitrage LLM se déclenche bien si le score d'accord est artificiellement bas |
| Meta-Evaluator | Le verdict change bien selon le seuil de confiance ; aucune ré-exécution automatique n'est jamais déclenchée |
| Reporting | Le PDF se génère sans crash même si certains champs du state sont vides/None |

### 1.2 Tests d'intégration bout-en-bout (pipeline complet)

- **Cas nominal** : dataset propre, pipeline complet, vérifier que `report_path` existe et que le PDF est ouvrable.
- **Cas classification binaire** vs **classification multiclasse** vs **régression** : vérifier que les bonnes métriques apparaissent dans chaque cas (`accuracy` vs `mae`/`r2`), et qu'aucun champ incompatible n'apparaît (ex: pas de `roc_auc` en régression).

### 1.3 Cas limites de données (les plus importants pour un projet censé généraliser)

| Cas | Ce qui doit se passer |
|---|---|
| Dataset avec beaucoup de valeurs manquantes (>50% sur certaines colonnes) | Colonnes droppées proprement, pas de crash |
| Dataset avec une colonne catégorielle à très haute cardinalité (ex: ID unique par ligne) | Détecté comme quality issue, droppée en preprocessing |
| Classe très minoritaire (déséquilibre extrême, ex: 98%/2%) | StratifiedKFold doit réduire le nombre de folds automatiquement sans planter |
| Dataset très petit (<20 lignes) | Le pipeline doit soit tourner avec des folds réduits, soit échouer avec un message clair — pas de crash silencieux |
| Colonne target avec une seule valeur unique | Doit être détecté et lever une erreur claire (pas de ML possible) |
| Toutes les colonnes numériques (aucune catégorielle) | Le Preprocessing Agent doit sauter proprement l'étape d'encodage |
| Toutes les colonnes catégorielles (aucune numérique) | Pas de crash sur le scaling (aucune colonne numérique à standardiser) |
| Doublons massifs dans le dataset | Bien détectés et supprimés, log cohérent |
| Dataset avec des colonnes de dates/texte libre non structuré | Vérifier comment le système les traite actuellement (probablement mal — bon test pour découvrir une limite réelle à documenter) |

### 1.4 Tests spécifiques aux nouveaux mécanismes statistiques/arbitrage (V2)

- **Forcer un cas d'égalité statistique** : construire artificiellement deux modèles avec des scores CV très proches → vérifier que `p_value > 0.05` déclenche bien l'appel LLM, et que `llm_arbitration_log` contient l'entrée attendue.
- **Forcer une confiance de target basse** : dataset sans target explicite ni nom reconnu → vérifier l'appel LLM et son intégration si la suggestion est valide.
- **Forcer un désaccord SHAP/Permutation** : c'est le plus dur à provoquer artificiellement — un bon candidat est un modèle avec beaucoup de features corrélées entre elles (la colinéarité est une cause connue de désaccord entre méthodes d'importance).
- **Vérifier le fallback LLM hors-ligne** : coupe volontairement `GROQ_API_KEY` (ou mets une clé invalide) et vérifie que chaque agent produit un message de fallback clair (`[Offline mode]...`) au lieu de crasher.

### 1.5 Tests de performance/timing

- Mesurer le temps total du pipeline sur un dataset de taille moyenne (quelques milliers de lignes, ~20 colonnes) — doit rester sous les 2 minutes que tu t'étais fixées pour l'UI.
- Vérifier spécifiquement que `HyperparameterAgent` (avec son retry potentiel) ne dépasse jamais 90 secondes, même dans le pire cas (force le retry à se déclencher systématiquement en abaissant temporairement `MIN_IMPROVEMENT_THRESHOLD`).

### 1.6 Tests d'UI (manuels, via Streamlit)

- Upload d'un CSV, vérifier l'aperçu et la sélection de target.
- Vérifier que la barre de progression avance bien étape par étape (via `.stream()`).
- Vérifier le bouton de téléchargement PDF (fichier non corrompu, s'ouvre correctement).
- Re-upload d'un nouveau dataset après un premier run terminé — vérifier que `st.session_state` ne mélange pas les anciens et nouveaux résultats.

---

## 2. Datasets publics recommandés — par type de problème

L'objectif ici est de vérifier que ton pipeline **généralise** vraiment (l'exigence explicite de ton encadrante), pas seulement qu'il marche sur Titanic.

### Classification binaire
- **Titanic** (déjà utilisé) — baseline de référence
- **Breast Cancer Wisconsin** — disponible directement dans scikit-learn (`sklearn.datasets.load_breast_cancer()`), aucune colonne catégorielle, bon test de "tout numérique"
- **Pima Indians Diabetes** — intéressant car les valeurs manquantes sont encodées en `0` plutôt qu'en `NaN` (piège réaliste pour tester la détection de qualité de données)

### Classification multiclasse
- **Iris** — `sklearn.datasets.load_iris()`, très petit (150 lignes), bon test du cas limite "dataset petit + StratifiedKFold"
- **Wine** — `sklearn.datasets.load_wine()`, 3 classes, toutes features numériques
- **Palmer Penguins** — a des valeurs manquantes ET des colonnes catégorielles, bon test combiné

### Régression
- **California Housing** — `sklearn.datasets.fetch_california_housing()`, remplace l'ancien Boston Housing (retiré de scikit-learn pour des raisons éthiques documentées — à éviter si tu le vois encore cité quelque part)
- **Bike Sharing Demand** (UCI/Kaggle) — bon test avec des colonnes de type "compte"/quasi-catégorielles
- **Diamonds** (souvent disponible via seaborn : `sns.load_dataset('diamonds')`) — mélange de catégorielles ordinales et numériques

### Cas difficiles / stress-tests
- **Credit Card Fraud Detection** (Kaggle) — déséquilibre de classe extrême (~0.17% de fraude), excellent test de robustesse pour `StratifiedKFold` et la gestion du déséquilibre
- **Adult Income / Census Income** (UCI) — haute cardinalité sur certaines colonnes catégorielles (`native-country`), bon test de la logique One-Hot vs Label Encoding
- **Ames Housing** (Kaggle) — beaucoup de colonnes (~80), mélange fort de types, bon test de robustesse générale et de temps d'exécution

---

## 3. Où trouver ces datasets

- **scikit-learn intégré** (le plus simple, zéro téléchargement) : `sklearn.datasets.load_*` / `fetch_*` — pour Iris, Wine, Breast Cancer, California Housing.
  ```python
  from sklearn.datasets import load_breast_cancer
  data = load_breast_cancer(as_frame=True)
  df = data.frame  # DataFrame prêt à uploader dans EMADS après export CSV
  df.to_csv("breast_cancer.csv", index=False)
  ```
- **UCI Machine Learning Repository** (archive.ics.uci.edu) — la référence académique historique, pour Adult Income, Bike Sharing, Pima Diabetes.
- **Kaggle Datasets** (kaggle.com/datasets) — pour Credit Card Fraud, Ames Housing, Palmer Penguins ; nécessite un compte gratuit.
- **OpenML** (openml.org) — agrège des centaines de datasets avec métadonnées standardisées, pratique pour scripter le téléchargement automatique de plusieurs datasets d'un coup si tu veux faire l'évaluation multi-datasets systématique de la V3.

---

## 4. Suggestion de méthode de test

Plutôt que de tester manuellement dataset par dataset dans l'UI, je te recommande
d'écrire un script de test batch (réutilisable pour l'axe "évaluation multi-datasets"
de la V3 dont on a parlé) :

```python
# tests/test_multi_dataset_smoke.py
import pandas as pd
from app.core.state.emads_state import create_initial_state
from app.core.supervisor.supervisor_agent import SupervisorAgent

DATASETS = [
    ("data/test_datasets/titanic.csv", "Survived"),
    ("data/test_datasets/breast_cancer.csv", "target"),
    ("data/test_datasets/iris.csv", "target"),
    ("data/test_datasets/california_housing.csv", "MedHouseVal"),
    ("data/test_datasets/pima_diabetes.csv", "Outcome"),
    # ajoute-en au fur et à mesure
]

for path, target in DATASETS:
    print(f"--- Testing {path} ---")
    state = create_initial_state(dataset_path=path, dataset_name=path)
    state["target_column"] = target
    try:
        result = SupervisorAgent().run_pipeline(state)
        print(f"OK — model: {result.get('selected_model_name')}, "
              f"metrics: {result.get('metrics')}")
    except Exception as e:
        print(f"FAILED: {e}")
```

Ça te donne, en une exécution, une vue d'ensemble de tous les datasets qui passent ou
échouent — beaucoup plus rapide que de tester un par un dans Streamlit, et directement
réutilisable pour l'évaluation empirique de la V3.