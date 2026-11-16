# Food Recipe Provenance Classification

**Classifying Italian and American recipes using ingredient-level TF-IDF features, supervised feature engineering, and ensemble learning.**

An individual final project for a Machine Learning course at Bocconi University. I developed and compared classification models, engineered cuisine-specific features, and combined complementary models in a weighted soft-voting ensemble.

## Problem

The goal is to predict whether a recipe is **American (`1`)** or **Italian (`2`)** from its ingredient profile.

| Dataset | Recipes | Features |
| --- | ---: | --- |
| Training set | 3,200 labelled | 40 ingredient TF-IDF scores |
| Held-out test set | 1,734 unlabelled | 40 ingredient TF-IDF scores |

Each feature reflects the importance of an ingredient within a recipe, represented using **term frequency–inverse document frequency (TF-IDF)**. Performance is measured by **misclassification rate (0–1 loss; lower is better)** using stratified 10-fold cross-validation.

## Approach

### 1. Leakage-safe feature engineering

I implemented a custom scikit-learn transformer, `CuisineProfileAugmenter`, to add four supervised features to the original 40:

- Cosine similarity to the average American recipe profile.
- Cosine similarity to the average Italian recipe profile.
- Difference between the two similarity scores.
- An ingredient-level log-ratio score indicating relative association with Italian versus American recipes.

The transformer is fitted **inside each cross-validation training fold** through scikit-learn `Pipeline`, so the validation labels are never used to construct these features.

### 2. Model selection and tuning

I compared several model families, then focused on nonlinear approaches. The final workflow tunes three base classifiers with `GridSearchCV` and stratified 10-fold cross-validation:

- **RBF Support Vector Machine (SVM)**
- **Random Forest**
- **ExtraTrees**

The SVM pipeline includes imputation, cuisine-profile augmentation, and standard scaling; the tree-based pipelines omit scaling.

### 3. Weighted soft-voting ensemble

The final model combines **five classifiers**: one RBF SVM, two Random Forests, and two ExtraTrees models. The tree models use different random seeds to introduce diversity, and their class-probability predictions are aggregated using weighted soft voting.

A cross-validated search over candidate weight vectors selected **`[7, 0.5, 0.5, 2, 2]`** in the original project.

## Results

The following are the **cross-validation results recorded during the project**, expressed as misclassification rates:

| Model | 10-fold CV loss |
| --- | ---: |
| Initial RBF SVM baseline | 0.0963 |
| Tuned RBF SVM | 0.0791 |
| Tuned Random Forest | 0.0900 |
| Tuned ExtraTrees | 0.0775 |
| **Final weighted ensemble** | **0.0722** |

The final recorded CV loss corresponds to approximately **92.78% cross-validation accuracy**. It is a cross-validation estimate, **not** a measured accuracy on the unlabelled held-out test set.

## Ingredient interpretation

I examined what distinguishes the two cuisines using two complementary analyses:

1. **Cuisine-profile log-ratios** to identify ingredients more strongly associated with one class.
2. **ExtraTrees feature importance** to identify the ingredients most useful for classification.

One interesting finding is that **pasta can appear more American-associated under the TF-IDF representation**, despite its obvious culinary association with Italy. This illustrates how inverse document frequency can downweight ingredients that are ubiquitous within a cuisine. The interpretation concerns the **dataset representation**, not cuisine in general.
