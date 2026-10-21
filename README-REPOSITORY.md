# DSB-Unit-03 — Repository overview and completeness tracker

This educational repository contains Data Science Bootcamp learning materials for machine learning and advanced modeling. The numbered folders are learning units covering probability, regression, classification, model evaluation, time series, ensemble methods, clustering, dimensionality reduction, and recommender systems. Materials include Jupyter notebooks, classroom examples, practice exercises, datasets, and supporting media.

The existing [Readme.MD](Readme.MD) is the detailed data science glossary. This second README provides the repository inventory and tracks slide-deck completeness.

## Slide-deck requirement

**Every numbered learning-unit folder should contain its corresponding slide deck as a PDF document.** Notebooks, embedded slides, images, and links to external slides do not satisfy this requirement. Supporting folders (`assets/`, `data/`, and `labs/`) are listed separately below.

**Inventory checked: 2026-10-21. Status: 9 of 25 learning units have a PDF slide deck; 16 are missing.** The nine existing unit folders through unit 10 now contain their slide decks. Ten PDFs were copied from the instructor repository’s `99 - Presentations/` folder: nine unit decks plus the supplementary `05a - residuals.pdf` in unit 05. Original filenames are preserved. Units after 10 remain pending in this repository.

This is a manual tracker, not an automatically refreshed inventory. After adding a PDF, confirm that it is the correct unit slide deck, replace `Missing` with `Present`, add its relative link in the last column, and update the date and totals above. Store PDFs directly in the relevant unit folder; preserving the original slide-deck filename is recommended.

## Learning-unit inventory

Contents below include files in nested practice and media folders. Notebook filenames are listed so classroom and practice variants can be distinguished.

| Learning unit | Current contents | PDF slide deck | Slide-deck file |
| --- | --- | --- | --- |
| [01 - Probability](01%20-%20Probability) | 1 notebook: `02-probability.ipynb` | Present | [01 - Probability.pdf](01%20-%20Probability/01%20-%20Probability.pdf) |
| [02 - Linear Regression I](02%20-%20Linear%20Regression%20I) | 5 notebooks: `02-simple-linear-regression-live.ipynb`; `02-simple-linear-regression.ipynb`; `03-multiple-linear-regression-live.ipynb`; `03-multiple-linear-regression.ipynb`; `practice/linear_regression-review-lab-starter.ipynb` | Present | [02 - ML-Linear-Regression.pdf](02%20-%20Linear%20Regression%20I/02%20-%20ML-Linear-Regression.pdf) |
| [04 - Regression Metrics](04%20-%20Regression%20Metrics) | 2 notebooks: `02-regression-metrics-live.ipynb`; `02-regression-metrics.ipynb` | Present | [04 - ML-Regression-Metrics.pdf](04%20-%20Regression%20Metrics/04%20-%20ML-Regression-Metrics.pdf) |
| [05 - Model Generalizability](05%20-%20Model%20Generalizability) | 2 notebooks: `02-train-test-split-cross-validation-live-classroom.ipynb`; `02-train-test-split-cross-validation.ipynb` | Present | [05 - ML-Model-Generalizability-Bias-Variance.pdf](05%20-%20Model%20Generalizability/05%20-%20ML-Model-Generalizability-Bias-Variance.pdf); [05a - residuals.pdf (supplement)](05%20-%20Model%20Generalizability/05a%20-%20residuals.pdf) |
| [06 - Model Workflow](06%20-%20Model%20Workflow) | 2 notebooks: `02-model-workflow-live-classroom.ipynb`; `02-model-workflow.ipynb` | Present | [06 - ML-Model-Workflow.pdf](06%20-%20Model%20Workflow/06%20-%20ML-Model-Workflow.pdf) |
| [07 - k-Nearest Neighbors](07%20-%20k-Nearest%20Neighbors) | 2 notebooks: `02-knn-live.ipynb`; `02-knn.ipynb` | Present | [07 - ML-k-Nearest-Neighbors.pdf](07%20-%20k-Nearest%20Neighbors/07%20-%20ML-k-Nearest-Neighbors.pdf) |
| [08 - Logistic Regression](08%20-%20Logistic%20Regression) | 3 notebooks: `02-logistic-regression-live.ipynb`; `02-logistic-regression.ipynb`; `practice/logistic-regression-practice.ipynb` | Present | [08 - ML-Logistic-Regression.pdf](08%20-%20Logistic%20Regression/08%20-%20ML-Logistic-Regression.pdf) |
| [09 - Classification Metrics](09%20-%20Classification%20Metrics) | 4 notebooks: `02-classification-metrics-live.ipynb`; `02-classification-metrics.ipynb`; `03-roc-auc-live.ipynb`; `03-roc-auc.ipynb` | Present | [09 - ML-Classification-Metrics.pdf](09%20-%20Classification%20Metrics/09%20-%20ML-Classification-Metrics.pdf) |
| [10 - Imbalanced Data](10%20-%20Imbalanced%20Data) | 1 notebook: `02-imbalanced-data.ipynb` | Present | [10 - Imbalanced Classes.pdf](10%20-%20Imbalanced%20Data/10%20-%20Imbalanced%20Classes.pdf) |
| [11 - Regularization](11%20-%20Regularization) | 1 notebook: `02-regularization.ipynb` | Missing | — |
| [12 - Grid Search and Pipelines](12%20-%20Grid%20Search%20and%20Pipelines) | 2 notebooks: `02-gridsearch-pipelines.ipynb`; `03-gridsearch-pipelines-practice.ipynb` | Missing | — |
| [13 - Model Interpretation](13%20-%20Model%20Interpretation) | 1 notebook: `02-model-explainability.ipynb` | Missing | — |
| [14 - Framing a Data Science Problem](14%20-%20Framing%20a%20Data%20Science%20Problem) | 1 notebook: `calculating-roi-starter-code.ipynb` | Missing | — |
| [16 - Time Series I - EDA](16%20-%20Time%20Series%20I%20-%20EDA) | 2 notebooks: `graph-gen.ipynb`; `ts-i-eda-starter.ipynb` | Missing | — |
| [17 - Time Series II - Modeling](17%20-%20Time%20Series%20II%20-%20Modeling) | 1 notebook: `ts-ii-modeling-starter.ipynb` | Missing | — |
| [18 - Time Series III - FB Prophet](18%20-%20Time%20Series%20III%20-%20FB%20Prophet) | 1 notebook: `ts-iii-model-demos.ipynb` | Missing | — |
| [20 - Decision Trees with CART](20%20-%20Decision%20Trees%20with%20CART) | 2 notebooks: `02-decision-trees.ipynb`; `visualization-carts.ipynb` | Missing | — |
| [21 - Bootstrapping Bagging and Random Forests](21%20-%20Bootstrapping%20Bagging%20and%20Random%20Forests) | 1 notebook: `02-ensembles-bagging-random-forest.ipynb` | Missing | — |
| [22 - Boosting](22%20-%20Boosting) | 2 notebooks: `02-ensembles-boosting.ipynb`; `adaboost-from-scratch.ipynb` | Missing | — |
| [23 - Support Vector Machines (SVMs)](23%20-%20Support%20Vector%20Machines%20%28SVMs%29) | 1 notebook: `starter-code.ipynb` | Missing | — |
| [24 - Gradient Descent](24%20-%20Gradient%20Descent) | 1 notebook: `02-gradient-descent.ipynb` | Missing | — |
| [25 - Intro to Clustering with K-Means](25%20-%20Intro%20to%20Clustering%20with%20K-Means) | 4 notebooks: `02-kmeans-NOSLIDES.ipynb`; `02-kmeans.ipynb`; `practice/cluster_evaluation-lab.ipynb`; `practice/kmeans_clustering-lab.ipynb`; `practice/README.md` | Missing | — |
| [26 - DBSCAN and Other Clustering Techniques](26%20-%20DBSCAN%20and%20Other%20Clustering%20Techniques) | 2 notebooks: `practice/DBSCAN_kmeans_and_hierarchical-lab.ipynb`; `practice/practice_dbscan-lab.ipynb`; `practice/README.md`; 8 practice CSV datasets in `practice/data/` | Missing | — |
| [27 - Principal Component Analysis (PCA)](27%20-%20Principal%20Component%20Analysis%20%28PCA%29) | 2 notebooks: `doug-generator.ipynb`; `pca-blank.ipynb`; 9 images in `images/` and `make-doug/` | Missing | — |
| [28 - Recommender Systems](28%20-%20Recommender%20Systems) | 1 notebook: `02-recommender-systems.ipynb` | Missing | — |

Unit numbers **03, 15, and 19 have no corresponding folders** in this checkout. They are excluded from the 25-unit total; verify against the course syllabus whether additional units should be added.

## Supporting folders

| Folder | Contents | PDF status |
| --- | --- | --- |
| [assets/](assets/) | 71 supporting files: 69 images, an HTML glossary, and an audio summary. | No PDFs; shared supporting materials. |
| [data/](data/) | 47 shared data/documentation files: 40 CSVs, 5 text files (including dataset READMEs), and 2 `.data` files. | No PDFs; shared datasets. |
| [labs/](labs/) | Three applied labs, detailed below, with instructions, notebooks, datasets, images, and site configuration. | No PDFs; exercise materials. |

| Lab | Current contents | PDF present? |
| --- | --- | --- |
| [01 - Regression-Sacramento](labs/01%20-%20Regression-Sacramento) | `README.md`; `lab-regression-sacramento-YOURNAME-HERE.ipynb`; `lab-regression-sacramento.ipynb`; `data/`, `assets/`, `_layouts/`, and `_config.yml`. | No |
| [02 - Classification-LeftHanded](labs/02%20-%20Classification-LeftHanded) | `README.md`; `lab-classification-left-handedness-YOURNAME-HERE.ipynb`; `data/`, `assets/`, `_layouts/`, and `_config.yml`. | No |
| [03 - Supervised-Model-Comparison](labs/03%20-%20Supervised-Model-Comparison) | `README.md`; `lab-supervised-learning-model-comparison.ipynb`; `data/`, `assets/`, `_layouts/`, and `_config.yml`. | No |

## Root files

- [Readme.MD](Readme.MD): machine learning and advanced modeling glossary, examples, and revision resources.
- [README-REPOSITORY.md](README-REPOSITORY.md): this repository overview and manual completeness tracker.
- [index.html](index.html) and `.nojekyll`: repository website entry point and GitHub Pages configuration.
- [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), and [main.py](main.py): local Python project configuration, dependency lockfile, and script. These three files were present locally but untracked at the time of this inventory.
- `.gitignore` and `.gitattributes`: version-control configuration. Local environment and Git metadata folders are not learning units.
