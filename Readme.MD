<a id="top"></a>

# Data Science Glossary: Machine Learning and Advanced Modeling

For the repository overview, unit contents, and PDF slide-deck completeness tracker, see [README-REPOSITORY.md](README-REPOSITORY.md). Every numbered learning-unit folder should contain its corresponding slide deck as a PDF.

Key terms, functions and methods from the machine learning unit, running from probability to recommender systems. This glossary picks up where the [Unit 2 glossary](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD) (sections 1–11: Python, pandas and EDA) left off, so numbering starts at section 12, and terms already defined there are linked rather than repeated. Each of the 548 entries says what the term means, where it appears in the course, and (where relevant) shows working code.

**How to read the examples.** Examples follow the course notebooks and assume the standard imports below. Extra libraries are imported inside the entries that need them; `statsmodels`, `imbalanced-learn`, `xgboost`, `arch` and `shap` may need installing with `pip install`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
```

Most notebooks read a CSV from the course `data/` folder, and the entries use the same variable names: `elec` (`elecdemand.csv`), `bike` (`bikeshare.csv`), `admissions` (`grad_admissions.csv`), `wine` (`winequality_merged.csv`), `titanic` (`titanic_clean.csv`), `aapl` (`AAPL.csv`) and so on. Scaled data is named `X_train_sc` or `Z_train`. To try an example without the course files, swap in a built-in dataset such as `load_breast_cancer()`, `load_iris()`, `load_digits()`, `make_classification()` or `make_blobs()` from `sklearn.datasets`, or `sm.datasets.macrodata` for time series.

**Library versions.** Code is written to run on current releases (scikit-learn 1.8, pandas 3, statsmodels 0.15). Libraries change between versions: parameters are renamed, defaults change and old functions are removed, so tutorials and older notebooks sometimes use syntax that no longer works. Where that affects something you'll use, the entry says what changed, and [Library version changes to know](#library-version-changes-to-know) collects them in one table. When code fails or warns, check the documentation for the version you have installed.

**For your capstone.** Every section opens with an **In your capstone** line saying when to reach for it, and section 38, the [Capstone playbook](#s38-capstone-playbook), walks the data science lifecycle stage by stage with tested code templates and a checklist of common mistakes.

<a id="audio-summary"></a>

**AI audio summary.** An AI-generated audio overview of the glossary's contents and key learnings, useful for revision on the go. It's a summary, not a substitute: where anything differs, the written entries are the reference.

<audio controls preload="none" style="width:100%;max-width:560px">
  <source src="assets/AIAudioSummary.m4a" type="audio/mp4">
</audio>

[Listen to or download the audio summary (AIAudioSummary.m4a)](assets/AIAudioSummary.m4a). GitHub doesn't play audio inside a README, so on github.com this link opens the file's page (use **View raw** or the download button); it plays directly in the [HTML glossary](https://charleyhananiaga.github.io/DSB-Unit-03/).

## Contents

12. [Probability](#s12-probability)
13. [Machine learning foundations and linear regression](#s13-machine-learning-foundations-and-linear-regression)
14. [Regression metrics](#s14-regression-metrics)
15. [Generalization, bias–variance and cross-validation](#s15-generalization-bias-variance-and-cross-validation)
16. [k-Nearest Neighbors](#s16-k-nearest-neighbors)
17. [Logistic regression](#s17-logistic-regression)
18. [Classification metrics](#s18-classification-metrics)
19. [Imbalanced classes](#s19-imbalanced-classes)
20. [Regularization](#s20-regularization)
21. [Hyperparameter tuning: grid search and pipelines](#s21-hyperparameter-tuning-grid-search-and-pipelines)
22. [Model explainability](#s22-model-explainability)
23. [Framing a data science problem and ROI](#s23-framing-a-data-science-problem-and-roi)
24. [Leveraging GenAI as a data scientist](#s24-leveraging-genai-as-a-data-scientist)
25. [Time series: data handling and stationarity](#s25-time-series-data-handling-and-stationarity)
26. [Time series: forecasting models](#s26-time-series-forecasting-models)
27. [Simulation](#s27-simulation)
28. [Decision trees](#s28-decision-trees)
29. [Ensembles: bagging and random forests](#s29-ensembles-bagging-and-random-forests)
30. [Ensembles: boosting](#s30-ensembles-boosting)
31. [Support vector machines](#s31-support-vector-machines)
32. [Gradient descent and optimization](#s32-gradient-descent-and-optimization)
33. [Unsupervised learning and k-means clustering](#s33-unsupervised-learning-and-k-means-clustering)
34. [DBSCAN and hierarchical clustering](#s34-dbscan-and-hierarchical-clustering)
35. [Principal component analysis (PCA)](#s35-principal-component-analysis-pca)
36. [Recommender systems](#s36-recommender-systems)
37. [Bayesian statistics](#s37-bayesian-statistics)
38. [Capstone playbook](#s38-capstone-playbook)

[Audio summary](#audio-summary) · [Concept map](#concept-map) · [A–Z index](#a-z-index)

<a id="concept-map"></a>

## Concept map

The whole unit on one page: 27 modules, grouped into themes, with the key concepts of each (294 in all). Every concept is a link.

**Using the clickable map.** GitHub displays SVG files as plain pictures, so the links inside the map only work when it's opened in a browser. On GitHub, use the **linked outline below**, which works right here. For the interactive map, use the GitHub Pages site: the [clickable mind map](https://charleyhananiaga.github.io/DSB-Unit-03/assets/ds-ml-glossary-mindmap.svg) and the [HTML glossary](https://charleyhananiaga.github.io/DSB-Unit-03/), which adds search, the audio summary and the clickable map embedded in the page. (To switch the site on: repository **Settings → Pages → Deploy from a branch → main / (root)**.) Offline, download the repository and open `assets/ds-ml-glossary-mindmap.svg` or `assets/ds-ml-glossary.html` in a browser.

[![Concept map of the machine learning unit](assets/ds-ml-glossary-mindmap.png)](https://charleyhananiaga.github.io/DSB-Unit-03/assets/ds-ml-glossary-mindmap.svg)

The same map as a linked outline:

- **[12. Probability](#s12-probability)**
  - _Sets:_ [Sample space](#sample-space), [Event](#event), [Intersection ∩](#intersection-a-b), [Union ∪](#union-a-b), [Complement](#complement-a)
  - _Rules:_ [Complement rule](#complement-rule), [Multiplication rule](#multiplication-rule-intersections), [Addition rule](#addition-rule-unions), [Independence](#statistical-independence), [Conditional probability](#conditional-probability)
  - _Simulation:_ [Monte Carlo simulation](#simulation-monte-carlo), [Law of Large Numbers](#law-of-large-numbers-lln), [Gambler's fallacy](#gambler-s-fallacy), [np.random.choice()](#np-random-choice)
- **[13. Linear regression](#s13-machine-learning-foundations-and-linear-regression)**
  - _ML basics:_ [Supervised learning](#supervised-learning), [Unsupervised learning](#unsupervised-learning), [Regression vs classification](#regression-vs-classification-tasks), [Predictor & response](#predictor-and-response-and-their-many-names)
  - _OLS model:_ [OLS](#ordinary-least-squares-ols), [Intercept (β₀)](#intercept), [Coefficient (β₁)](#slope-coefficient), [Interpreting coefficients](#interpreting-coefficients)
  - _scikit-learn:_ [LinearRegression](#linearregression), [Six-step workflow](#the-modeling-workflow-in-scikit-learn), [X is 2D, y is 1D](#x-is-2d-y-is-1d)
  - _Assumptions:_ [LINEM assumptions](#linem-assumptions), [Residual plot](#residual-plot), [VIF](#variance-inflation-factor-vif), [Log-transform y](#log-transforming-the-target)
  - _Extensions:_ [Dummy variables](#dummy-variables-in-regression), [statsmodels OLS](#statsmodels-ols), [PolynomialFeatures](#polynomialfeatures)
- **[14. Regression metrics](#s14-regression-metrics)**
  - _Errors:_ [SSE](#sum-of-squared-errors-sse), [MSE](#mean-squared-error-mse), [RMSE](#root-mean-squared-error-rmse), [MAE](#mean-absolute-error-mae)
  - _Fit:_ [R²](#r2-coefficient-of-determination), [Adjusted R²](#adjusted-r2), [Null model](#null-model-baseline-for-regression)
- **[15. Generalization](#s15-generalization-bias-variance-and-cross-validation)**
  - _Fit:_ [Underfitting](#underfitting), [Overfitting](#overfitting), [Bias](#bias), [Variance](#variance-model-variance), [Bias–variance trade-off](#bias-variance-trade-off), [Sampling bias](#sampling-bias)
  - _Holdout:_ [Train-test split](#train-test-split), [stratify / shuffle](#test-size-stratify-and-shuffle), [Train vs test scores](#reading-train-vs-test-scores)
  - _Cross-validation:_ [k-fold cross-validation](#k-fold-cross-validation), [KFold](#kfold), [cross_val_score()](#cross-val-score), [Data leakage](#data-leakage-in-model-evaluation)
- **[16. k-Nearest Neighbors](#s16-k-nearest-neighbors)**
  - _Algorithm:_ [kNN algorithm](#k-nearest-neighbors-knn), [Distance metrics](#distance-metrics-euclidean-manhattan-minkowski), [Choosing k](#choosing-k)
  - _Practice:_ [Scaling](#scaling-for-distance-based-models), [KNeighborsClassifier](#kneighborsclassifier)
  - _Baselines:_ [Accuracy](#accuracy), [Null accuracy](#null-accuracy), [DummyClassifier](#dummyclassifier)
- **[17. Logistic regression](#s17-logistic-regression)**
  - _Model:_ [Sigmoid function](#logistic-sigmoid-function), [Logit link](#link-function-and-the-logit), [Perfect separation](#perfect-separation)
  - _Odds:_ [Odds](#odds), [Log-odds](#log-odds), [Odds ratio](#odds-ratio), [Interpreting coefficients](#interpreting-logistic-regression-coefficients)
  - _scikit-learn:_ [.predict_proba()](#predict-proba), [solver and max_iter](#solver-and-max-iter), [Default regularization](#regularization-is-on-by-default)
  - _Multiclass:_ [Multinomial](#multinomial-logistic-regression), [One-vs-rest (OvR)](#one-vs-rest-ovr)
- **[18. Classification metrics](#s18-classification-metrics)**
  - _Confusion matrix:_ [Confusion matrix](#confusion-matrix), [TP / FP / TN / FN](#true-false-positives-and-negatives), [Type I & II errors](#type-i-and-type-ii-errors)
  - _Metrics:_ [Recall](#recall-sensitivity-true-positive-rate), [Precision](#precision), [Specificity](#specificity-true-negative-rate), [F1 score](#f1-score)
  - _Thresholds:_ [Threshold](#classification-threshold), [ROC curve](#roc-curve), [ROC AUC](#auc-roc-auc), [Precision–recall curve](#precision-recall-curve)
- **[19. Imbalanced classes](#s19-imbalanced-classes)**
  - _Problem:_ [Imbalanced classes](#imbalanced-classes), [Balanced accuracy](#balanced-accuracy)
  - _Resampling:_ [Oversampling](#oversampling), [RandomOverSampler](#randomoversampler), [SMOTE](#smote), [ADASYN](#adasyn)
  - _Other fixes:_ [Class weights](#class-weights), [Resampling trade-off](#resampling-is-a-trade-off), [Lower the threshold](#changing-the-threshold-for-imbalanced-data), [imblearn pipeline](#imbalanced-learn-pipeline)
- **[20. Regularization](#s20-regularization)**
  - _Penalties:_ [Ridge (L2)](#ridge-regression-l2), [Lasso (L1)](#lasso-regression-l1), [ElasticNet](#elasticnet)
  - _Tuning:_ [alpha](#alpha-regularization-strength), [RidgeCV and LassoCV](#ridgecv-and-lassocv), [C](#c-inverse-regularization-strength)
  - _Practice:_ [Standardize first](#standardize-before-regularizing), [LogisticRegressionCV](#logisticregressioncv)
- **[21. Grid search & pipelines](#s21-hyperparameter-tuning-grid-search-and-pipelines)**
  - _Tuning:_ [Hyperparameter](#hyperparameter), [GridSearchCV](#gridsearchcv), [RandomizedSearchCV](#randomizedsearchcv)
  - _Results:_ [best*params*](#best-params-best-score-and-best-estimator), [cv*results*](#cv-results)
  - _Pipelines:_ [Pipeline](#pipelines-in-tuning), [step\_\_param names](#step-parameter-naming), [Pipeline + GridSearchCV](#pipeline-gridsearchcv)
- **[22. Explainability](#s22-model-explainability)**
  - _Why:_ [Explainability](#model-explainability), [White-box model](#white-box-model), [Black-box model](#black-box-model), [Global vs local](#global-vs-local-explanations)
  - _Tools:_ [plot_tree](#visualizing-a-decision-tree-plot-tree), [Feature importance](#feature-importance), [Permutation importance](#permutation-importance), [Partial dependence](#partial-dependence-plot-pdp), [SHAP](#shap-shapley-additive-explanations)
- **[23. Framing & ROI](#s23-framing-a-data-science-problem-and-roi)**
  - _Framing:_ [Problem framing](#problem-framing), [Data science lifecycle](#data-science-lifecycle), [Five considerations](#five-key-considerations-for-framing), [Problem statement](#problem-statement), [SMART goals](#smart-goals)
  - _People:_ [Stakeholder](#stakeholder), [Influence/interest grid](#influence-interest-grid), [Sponsor & KPI](#project-sponsor-and-kpi)
  - _Success:_ [Metrics & hypotheses](#success-metrics-and-hypotheses), [Base rate & false alarms](#base-rate-and-false-alarms), [Problem type](#framing-the-problem-type-early)
  - _Value:_ [ROI](#return-on-investment-roi), [Risk-adjusted ROI](#risk-adjusted-roi), [Payback & NPV](#payback-period-and-npv)
  - _Ethics:_ [Ethics, legal, strategy](#ethical-legal-and-strategic-implications), [Data minimization](#data-minimization-and-anonymization), [Build, buy or partner](#build-buy-or-partner)
- **[24. GenAI](#s24-leveraging-genai-as-a-data-scientist)**
  - _How LLMs work:_ [LLM](#large-language-model-llm), [Token](#token), [Embedding](#embedding), [Transformer & attention](#transformer-and-attention), [Temperature](#temperature-and-sampling)
  - _Training & limits:_ [How LLMs are trained](#how-llms-are-trained), [Context window](#context-window), [Hallucination](#hallucination)
  - _Using GenAI:_ [Where GenAI helps](#where-genai-helps-in-the-data-science-lifecycle), [Generation vs analysis](#generation-mode-vs-analysis-mode), [Calling an LLM from Python](#calling-an-llm-from-python), [Data privacy](#data-privacy-with-genai-tools)
  - _Checking:_ [How output goes wrong](#how-genai-output-goes-wrong), [Testing AI code](#testing-ai-generated-code), [Validating classifications](#validating-genai-classifications)
- **[25. Time series data](#s25-time-series-data-handling-and-stationarity)**
  - _pandas:_ [Datetime index](#datetime-index), [.resample()](#resample-for-time-series), [Rolling statistics](#rolling-statistics), [.shift() lags](#lag-backshift-and-shift), [.diff()](#differencing-and-diff), [Seasonal differencing](#seasonal-differencing)
  - _Diagnostics:_ [Stationarity](#stationarity), [Autocorrelation](#autocorrelation), [ACF plot](#acf-plot), [PACF plot](#pacf-plot)
- **[26. Forecasting](#s26-time-series-forecasting-models)**
  - _Decomposition:_ [Classical decomposition](#classical-decomposition), [Trend](#trend), [Seasonality](#seasonality), [Naive forecasts](#naive-and-seasonal-naive-forecasts)
  - _ARIMA:_ [AR](#autoregression-ar), [MA](#moving-average-model-ma), [ARIMA](#arima-p-d-q), [SARIMA](#sarima), [ADF test](#augmented-dickey-fuller-adf-test), [Ljung–Box test](#ljung-box-test), [AIC](#aic-akaike-information-criterion)
  - _Extensions:_ [GLS](#generalized-least-squares-gls), [GARCH](#garch-p-q), [VAR](#vector-autoregression-var)
- **[27. Simulation](#s27-simulation)**
  - _Concepts:_ [Simulation](#simulation), [Key components](#key-components-of-a-simulation), [Four pillars](#four-pillars-of-simulation), [Utilization](#utilization)
  - _Approaches:_ [Discrete event (DES)](#discrete-event-simulation-des), [Agent-based (ABM)](#agent-based-modelling-abm), [Monte Carlo simulation](#monte-carlo-simulation), [System dynamics](#system-dynamics-sd)
  - _Python:_ [Queue simulation](#coffee-shop-queue-simulation), [np.random.default_rng()](#np-random-default-rng), [SimPy](#simpy)
  - _Trust:_ [Verification & validation](#verification-and-validation), [Sensitivity analysis](#sensitivity-analysis)
- **[28. Decision trees](#s28-decision-trees)**
  - _Structure:_ [Nodes & leaves](#root-node-internal-node-leaf-and-branch), [CART algorithm](#cart-algorithm), [Gini impurity](#gini-impurity)
  - _Control:_ [Stopping criteria](#stopping-criteria), [ccp_alpha pruning](#ccp-alpha-cost-complexity-pruning), [DecisionTreeClassifier](#decisiontreeclassifier-and-decisiontreeregressor)
- **[29. Bagging & random forests](#s29-ensembles-bagging-and-random-forests)**
  - _Bootstrap:_ [Bootstrap sample](#bootstrap-sample), [Bagging](#bagging-bootstrap-aggregating), [BaggingClassifier](#baggingclassifier)
  - _Evaluate:_ [Out-of-bag error](#out-of-bag-oob-observations-and-error), [Feature importance in ensembles](#feature-importance-in-ensembles)
  - _Forests:_ [Random forest](#random-forest), [max_features](#max-features), [Stacking](#stacking)
- **[30. Boosting](#s30-ensembles-boosting)**
  - _Idea:_ [Boosting](#boosting), [Decision stump](#decision-stump), [Bagging vs boosting](#bagging-vs-boosting)
  - _Algorithms:_ [AdaBoost](#adaboost), [Gradient boosting](#gradient-boosting), [XGBoost](#xgboost)
  - _scikit-learn:_ [AdaBoostClassifier](#adaboostclassifier), [learning_rate](#learning-rate-and-n-estimators)
- **[31. Support vector machines](#s31-support-vector-machines)**
  - _Margins:_ [Separating hyperplane](#separating-hyperplane), [Maximal margin](#maximal-margin-classifier), [Support vectors](#support-vectors), [Soft margin](#soft-margin-classifier)
  - _Kernels:_ [Kernel trick](#kernel-trick), [Linear / poly / RBF](#linear-polynomial-and-rbf-kernels), [C](#c-svm)
  - _scikit-learn:_ [SVC and LinearSVC](#svc-and-linearsvc)
- **[32. Gradient descent](#s32-gradient-descent-and-optimization)**
  - _Basics:_ [Loss function](#loss-function), [Derivative](#derivative), [Gradient descent](#gradient-descent)
  - _Step size:_ [Learning rate α](#learning-rate-step-size), [Convergence](#convergence-and-tolerance)
  - _Pitfalls:_ [Local vs global minimum](#local-vs-global-minimum), [Convex function](#convex-function), [Feature scaling](#feature-scaling-and-gradient-descent), [Newton's method](#newton-s-method), [L-BFGS](#quasi-newton-methods-bfgs-l-bfgs)
- **[33. k-means clustering](#s33-unsupervised-learning-and-k-means-clustering)**
  - _Clustering:_ [Clustering](#clustering), [Uses of clustering](#uses-of-clustering), [k-means algorithm](#k-means), [Centroid](#centroid), [Birch & Affinity Prop.](#birch-and-affinity-propagation)
  - _scikit-learn:_ [KMeans](#kmeans), [labels* and cluster_centers*](#labels-and-cluster-centers), [Labels are arbitrary](#cluster-labels-are-arbitrary)
  - _Evaluation:_ [Inertia](#inertia-within-cluster-sum-of-squares), [Elbow method](#elbow-method), [Silhouette score](#silhouette-score)
- **[34. DBSCAN & hierarchical](#s34-dbscan-and-hierarchical-clustering)**
  - _DBSCAN:_ [DBSCAN](#dbscan), [eps & min_samples](#eps-and-min-samples), [Core / border / noise](#core-border-and-noise-points), [k-distance plot](#k-distance-plot), [HDBSCAN](#hdbscan)
  - _Hierarchical:_ [Agglomerative clustering](#agglomerative-clustering), [Linkage](#linkage), [Dendrogram](#dendrogram)
  - _Metrics:_ [External cluster metrics](#external-cluster-metrics), [ARI](#adjusted-rand-index-ari)
- **[35. PCA](#s35-principal-component-analysis-pca)**
  - _Idea:_ [Dimensionality reduction](#dimensionality-reduction), [Principal component](#principal-component)
  - _scikit-learn:_ [PCA()](#pca-in-scikit-learn), [Explained variance](#explained-variance-and-explained-variance-ratio), [Scree plot](#scree-plot), [Loadings](#loadings-components)
  - _Maths:_ [SVD](#singular-value-decomposition-svd), [Reconstruction](#dimensionality-reduced-reconstruction)
- **[36. Recommenders](#s36-recommender-systems)**
  - _Types:_ [Collaborative filtering](#collaborative-filtering), [Content-based & hybrid](#content-based-knowledge-based-and-hybrid-recommenders)
  - _Build:_ [Ratings matrix](#utility-ratings-matrix), [Sparse matrix](#sparse-matrix), [Cosine similarity](#cosine-similarity), [pairwise_distances()](#pairwise-distances)
- **[37. Bayesian statistics](#s37-bayesian-statistics)**
  - _Ideas:_ [Bayesian statistics](#bayesian-statistics), [Frequentist vs Bayesian](#frequentist-vs-bayesian), [Bayes' theorem](#bayes-theorem), [Base rate fallacy](#base-rate-fallacy)
  - _Distributions:_ [Prior, likelihood, posterior](#prior-likelihood-and-posterior), [PMF & PDF](#probability-mass-and-density-functions), [Conjugate prior](#conjugate-prior-beta-binomial), [Credible interval](#credible-interval)
  - _Computing:_ [Grid approximation](#grid-approximation), [MCMC](#markov-chain-monte-carlo-mcmc), [PyMC](#pymc), [A/B testing](#bayesian-a-b-testing)
- **[38. Capstone playbook](#s38-capstone-playbook)**
  - _Lifecycle:_ [Using the glossary](#using-this-glossary-for-your-capstone), [1 Frame](#stage-1-frame-the-question), [2 Data](#stage-2-get-and-understand-the-data), [3 Validate](#stage-3-choose-the-split-and-validation-strategy), [4 Baselines](#stage-4-build-baselines)
  - _Build:_ [5 Candidate models](#stage-5-choose-candidate-models), [Approach by question](#choosing-an-approach-by-question-type), [Model cheat sheet](#model-selection-cheat-sheet), [6 Pipeline & tune](#stage-6-build-a-pipeline-and-tune)
  - _Deliver:_ [7 Evaluate](#stage-7-evaluate-honestly), [8 Explain](#stage-8-interpret-and-explain), [9 Communicate](#stage-9-communicate-the-value), [10 Deploy](#stage-10-deploy-monitor-and-next-steps)
  - _Templates:_ [Classification](#capstone-template-classification), [Regression](#capstone-template-regression), [Time series](#capstone-template-time-series-forecasting), [Clustering](#capstone-template-clustering)
  - _Habits:_ [Mistakes checklist](#common-capstone-mistakes-checklist), [Library changes](#library-version-changes-to-know), [Reproducibility](#reproducibility-habits)

---

<a id="s12-probability"></a>

## 12. Probability

_The language of uncertainty that every model in this unit is built on: experiments, events, set operations, the three probability rules, and estimating probabilities by simulation when the maths gets hard. Examples come from the dice, coin and urn problems in the probability notebook._

**In your capstone:** Use simulation to sanity-check any probability your analysis relies on, and think about base rates before trusting a classifier's alarms.

<a id="probability"></a>

### Probability

_Concept_

If you repeated an experiment infinitely many times, the **probability** of an event is the proportion of those repeats in which the event happens. It is always a number between 0 and 1: a fair coin lands heads with probability 0.5. Written formally, P(A) = lim (number of times A occurs ÷ n) as n → ∞. We can't run anything infinitely often, so in practice we either calculate probabilities by hand or estimate them by running many simulated trials.

Two interpretations are worth knowing. The definition above is the **frequentist** view (long-run frequency), which underpins sampling, A/B testing and Monte Carlo simulation. The **Bayesian** view treats probability as a degree of belief that is updated as evidence arrives. Probability theory itself began in the 16th century with the analysis of dice and card games.

See also: [Simulation (Monte Carlo)](#simulation-monte-carlo), [Law of Large Numbers (LLN)](#law-of-large-numbers-lln), [Frequentist vs Bayesian](#frequentist-vs-bayesian)

<a id="probability-notation-p-a"></a>

### Probability notation P(A)

_Concept_

`P(A)` reads "the probability that event A occurs". `P(A | B)` reads "the probability of A given that B has already happened". `P(A ∩ B)` is the probability that both happen and `P(A ∪ B)` the probability that at least one happens.

<a id="experiment"></a>

### Experiment

_Concept_

Any clearly defined process with an outcome: flipping a coin twice, rolling a die, drawing a marble. ("Lab coat optional.") Not to be confused with a designed experiment from Unit 2, though the idea is related.

See also: [Experiment (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#experiment)

<a id="outcome"></a>

### Outcome

_Concept_

A single possible result of an experiment, such as rolling a 4 or flipping HT.

<a id="sample-space"></a>

### Sample space

_Concept_

The set of **all** possible outcomes of an experiment, usually written S (or Ω). Flip a coin twice: S = {HH, HT, TH, TT}. Roll one die: S = {1, 2, 3, 4, 5, 6}.

**Order matters** in sequential experiments: HT (heads then tails) and TH are different outcomes, so two coin flips have 2² = 4 outcomes, not 3. Listing them separately prevents undercounting. With S = {HH, HT, TH, TT}: P(both heads) = 1/4 and P(at least one head) = P({HH, HT, TH}) = 3/4.

<a id="event"></a>

### Event

_Concept_

Any collection of outcomes, i.e. a subset of the sample space. For one die roll: A = {1}, B = {an even number} = {2, 4, 6}, C = {a prime number} = {2, 3, 5}.

> **Watch out:** 1 is neither prime nor composite, so the primes on a standard die are 2, 3 and 5. Notice that B and C share the outcome 2: that overlap is an intersection.

<a id="set-and-element"></a>

### Set and element

_Concept, Python_

A **set** is an unordered collection of distinct objects; each object in it is an **element**. Sets ignore order and duplicates: {1, 2} = {2, 1}, and `set([1, 1, 2])` is `{1, 2}`. `{Chloe, 17, 😂}` is a set with three elements. Python's built-in `set` behaves the same way and supports the set operations below.

```python
A = {2, 4, 6, 8}
B = {2, 3, 5, 7}
A & B          # {2}                    intersection
A | B          # {2, 3, 4, 5, 6, 7, 8}  union
S = set(range(1, 9))
S - A          # {1, 3, 5, 7}           complement of A within S
S - (A | B)    # {1}
```

Python sets also give very fast membership tests (`x in my_set`), which is why converting a long list to a set before repeated lookups is a common speed-up.

See also: [Set (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#set)

<a id="intersection-a-b"></a>

### Intersection (A ∩ B)

_Concept_

The elements in **both** A and B ("A and B"). For an 8-sided die with A = {2, 4, 6, 8} and B = {2, 3, 5, 7}, A ∩ B = {2}.

Memory hint: **∩ looks like an A, for AND**. In SQL this is `INTERSECT`; in pandas it's combining boolean masks with `&`.

<a id="union-a-b"></a>

### Union (A ∪ B)

_Concept_

The elements in A **or** B, or both. With the sets above, A ∪ B = {2, 3, 4, 5, 6, 7, 8}.

In probability, "or" is **inclusive** (A, B or both) unless stated otherwise. In SQL this is `UNION`; in pandas, `|`.

<a id="complement-a"></a>

### Complement (Aᶜ)

_Concept_

Everything in the sample space that is **not** in A. With S = {1, …, 8} and A = {2, 4, 6, 8}, Aᶜ = {1, 3, 5, 7}, and (A ∪ B)ᶜ = {1}.

In SQL this is `NOT IN` / `EXCEPT`; in pandas, `~mask`.

<a id="de-morgan-s-laws"></a>

### De Morgan's laws

_Concept_

Two rules for complements of unions and intersections: **(A ∪ B)ᶜ = Aᶜ ∩ Bᶜ** ("not (A or B)" = "not A and not B") and **(A ∩ B)ᶜ = Aᶜ ∪ Bᶜ**. With the 8-sided die, (A ∪ B)ᶜ = {1} = {1, 3, 5, 7} ∩ {1, 4, 6, 8}. The same rule applies to boolean filters in pandas: `~(a | b)` is the same as `~a & ~b`.

```python
S, A, B = set(range(1, 9)), {2, 4, 6, 8}, {2, 3, 5, 7}
S - (A | B) == (S - A) & (S - B)      # True
```

<a id="disjoint-mutually-exclusive-events"></a>

### Disjoint (mutually exclusive) events

_Concept_

Events that can't happen together, so P(A ∩ B) = 0. "Both marbles are blue" and "both marbles are red" are disjoint. For disjoint events, the union rule simplifies to P(A ∪ B) = P(A) + P(B).

A single coin flip landing heads **and** tails is the simplest example. In the urn problem, "both blue" and "both red" can't both happen, so P(same colour) = 11/46 + 11/46 = **11/23 ≈ 0.478** exactly.

<a id="key-probability-facts-kolmogorov-s-axioms"></a>

### Key probability facts (Kolmogorov's axioms)

_Concept_

Three facts that always hold, known as **Kolmogorov's axioms**: **P(S) = 1** (something in the sample space happens), **P(∅) = 0** (the probability of the empty set, an impossible event, is zero), and **0 ≤ P(A) ≤ 1** for any event A. If a calculation gives you a probability of 1.3 or −0.2, something has gone wrong. In ML, a model "probability" outside [0, 1] usually means you are looking at an unnormalized raw score (such as a logit, before the sigmoid or softmax is applied) or a bug.

See also: [Logistic (sigmoid) function](#logistic-sigmoid-function)

<a id="conditional-probability"></a>

### Conditional probability

_Concept_

The probability of A **given** that B has already happened, written P(A | B). Drawing marbles without replacement is the classic case: once a blue marble is out, the chance the next one is blue drops from 12/24 to 11/23.

Read the vertical bar **|** as "given that".

See also: [Bayes' theorem](#bayes-theorem)

<a id="statistical-independence"></a>

### Statistical independence

_Concept_

Two events are **independent** if learning that one happened gives you no information about the other: **P(A | B) = P(A)**. Equivalent statements are P(B | A) = P(B) and **P(A ∩ B) = P(A) · P(B)**. If P(A | B) ≠ P(A), the events are **dependent**.

The course's knowledge check sorts five pairs:

- **Independent:** a coin flip and a die roll; you hitting snooze and a stranger in Paris hitting snooze.
- **Dependent (behaviour):** pizza for lunch makes pizza for dinner less likely.
- **Dependent (confounding variable):** shark attacks and ice-cream sales rise together because warm weather drives both, not because one causes the other.
- **Dependent (autocorrelation):** today's and tomorrow's temperature, a time series.

Independence comes up again and again: it's the **I** in the LINE assumptions, Naive Bayes classifiers assume features are independent given the class, and time series data violate it by design.

See also: [Independence (I)](#independence-i), [Time series](#time-series), [Confounding variable (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#confounding-variable), [Correlation does not imply causation (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#correlation-does-not-imply-causation)

<a id="complement-rule"></a>

### Complement rule

_Concept_

Rule 1: **P(Aᶜ) = 1 − P(A)**. Often the easiest route to a hard probability: P(at least one six in four rolls) = 1 − P(no sixes) = 1 − (5/6)⁴.

Why it holds: A and Aᶜ split the sample space into two non-overlapping pieces that together make S, so P(A) + P(Aᶜ) = P(S) = 1. Calculating "at least one…" directly often needs many cases; "1 − P(none)" needs one.

<a id="multiplication-rule-intersections"></a>

### Multiplication rule (intersections)

_Concept_

Rule 2: **P(A ∩ B) = P(A) · P(B | A)**. With an urn of 12 red and 12 blue marbles, drawing twice without replacement, P(both blue) = (12/24) × (11/23) = 11/46 ≈ 0.239. If A and B are independent this reduces to P(A) · P(B).

The urn calculation, step by step: P(B₁) = 12/24. After one blue ball is removed, 11 blue and 12 red remain (23 in total), so P(B₂ | B₁) = 11/23. Both the numerator and the denominator drop because we sample **without replacement**. By symmetry, P(both red) is also 11/46.

<a id="addition-rule-unions"></a>

### Addition rule (unions)

_Concept_

Rule 3: **P(A ∪ B) = P(A) + P(B) − P(A ∩ B)**. You subtract the intersection because adding P(A) and P(B) counts the overlap twice. P(both marbles the same colour) = P(both blue) + P(both red) − 0 = 0.239 + 0.239 = 0.478, since the two events are disjoint.

Picture it as a Venn diagram: shading circle A and then circle B colours the overlap twice, so you subtract it once to get the true area of A ∪ B.

<a id="sampling-with-and-without-replacement"></a>

### Sampling with and without replacement

_Concept, NumPy_

**With replacement**, each item goes back before the next draw, so draws are independent and the same item can be picked twice (a bootstrap sample works this way). **Without replacement**, each item can be picked only once, so later draws depend on earlier ones (the urn problems).

```python
urn = np.array([np.repeat("red", 12), np.repeat("blue", 12)]).ravel()
np.random.choice(urn, size=2, replace=False)   # two different marbles
np.random.choice(urn, size=2, replace=True)    # could be the same marble twice
```

See also: [Bootstrap sample](#bootstrap-sample)

<a id="simulation-monte-carlo"></a>

### Simulation (Monte Carlo)

_Concept, Python_

Estimating a probability by running an experiment many times in code and counting how often the event happens: P(A) ≈ (times A occurs) ÷ n. When problems get too hard to do by hand, "why do the maths?" Every problem in the probability notebook follows the same pattern: a counter, a loop, an `if`, and a division.

```python
dice = [1, 2, 3, 4, 5, 6]

def odd_roll(n):
    count = 0
    for i in range(n):
        if np.random.choice(dice) % 2 != 0:   # is the roll odd?
            count += 1
    return count / n

odd_roll(1_000_000)    # ≈ 0.5
```

In modern data science, a quick simulation is often faster and less error-prone than hand-calculating complex combinatorics, and it's a good way to **check** an analytical answer. Monte Carlo methods get their own treatment in the simulation section.

See also: [Law of Large Numbers (LLN)](#law-of-large-numbers-lln), [Monte Carlo simulation](#monte-carlo-simulation)

<a id="law-of-large-numbers-lln"></a>

### Law of Large Numbers (LLN)

_Concept_

As you repeat an experiment more and more times, the simulated probability of an event gets closer and closer to the true probability. Formally, for independent, identically distributed (i.i.d.) random variables X₁, …, Xₙ with mean μ, the sample mean (1/n)ΣXᵢ converges to μ as n → ∞. Flip a coin 10 times and you might see 7 heads; simulate 1,000,000 flips and the proportion settles very close to 0.500. That's why `odd_roll(1_000_000)` is closer to 0.5 than `odd_roll(1_000)`, and it's the justification for every simulation in the course.

> **Watch out:** The LLN is **not** the Gambler's fallacy. Coins have no memory: after five heads in a row, the next flip is still 50/50. The proportion converges because a huge number of later flips dilutes the early run, not because the results "balance out".

See also: [Gambler's fallacy](#gambler-s-fallacy)

<a id="gambler-s-fallacy"></a>

### Gambler's fallacy

_Concept_

The mistaken belief that independent random outcomes "balance out" in the short run: "red has come up six times, so black is due". Each spin or flip is independent, so past results don't change the next probability. It's a common misreading of the Law of Large Numbers.

<a id="np-random-choice"></a>

### np.random.choice()

_NumPy_

Picks random items from a list or array. `size` sets how many, `replace` sets whether an item can be picked twice, and `p` sets the probability of each item, which lets you simulate an unfair coin.

```python
np.random.choice(dice)                                   # one die roll
np.random.choice(["heads", "tails"], p=[0.2, 0.8])       # unfair coin, 20% heads
np.random.choice(urn, size=2, replace=False)              # two draws, no replacement
```

See also: [Random number generators (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#random-number-generators)

<a id="np-random-randint-and-np-random-uniform"></a>

### np.random.randint() and np.random.uniform()

_NumPy_

`np.random.randint(low, high)` returns a random integer from `low` up to but **not including** `high`: `randint(1, 7)` is a die roll and `randint(0, 2)` a coin flip. `np.random.uniform(low, high)` returns a random decimal, used in the "break a stick in two places" interview problem.

```python
np.random.randint(1, 7)       # 1, 2, 3, 4, 5 or 6
np.random.randint(0, 2)       # 0 or 1
np.random.uniform(0, 1)       # e.g. 0.3745...
```

> **Watch out:** The upper bound is exclusive. `randint(1, 6)` never rolls a 6.

<a id="while-loop"></a>

### while loop

_Python_

Repeats a block **as long as** a condition is true, which suits "keep flipping until you get heads". Use `while True:` with `break` when the stopping condition is easier to check inside the loop.

```python
rolls = 1
while np.random.randint(0, 2) == 0:   # keep flipping while we get tails
    rolls += 1
```

> **Watch out:** If the condition can never become false (say, a probability of heads of 0), the loop runs forever. Guard against impossible inputs first.

See also: [for loop (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#for-loop), [continue and break (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#continue-and-break)

---

<a id="s13-machine-learning-foundations-and-linear-regression"></a>

## 13. Machine learning foundations and linear regression

_What machine learning is, the two big families of models, and the first model in the toolkit: ordinary least squares linear regression. Covers fitting, interpreting and checking a model in scikit-learn and statsmodels, using the electricity demand (`elecdemand.csv`) and bikeshare (`bikeshare.csv`) notebooks._

**In your capstone:** A linear regression is often your first and most interpretable model for a numeric target. Check the LINEM assumptions before interpreting coefficients, and always state effects with their units.

<a id="machine-learning"></a>

### Machine learning

_Concept_

Letting a computer use data to learn the relationship between some **predictors** (x-variables) and a **response** (y-variable). The slides call it "a fancy way of saying statistical modeling". Examples: predicting a house's sale price from size, location and rooms, or whether a customer will cancel their subscription next month from their recent activity.

"Learning" has a precise meaning: the model has adjustable numbers called **parameters** (for linear regression, the intercept and slopes), and learning means choosing values for them that make the model fit the data well.

A useful test for "is it ML?": **was the behaviour learned from data, or programmed as rules?** From the course discussions:

- An online house-price estimate (an automated valuation model trained on past sales): **ML**, specifically supervised regression.
- A factory robot following pre-programmed motion paths: **automation**, not ML (though defect-spotting cameras on the same line might use ML).
- A mortgage calculator applying a fixed income multiple: **rules-based**, not ML (though the bank's real lending decision may use an ML credit score).

See also: [AI, machine learning and deep learning](#ai-machine-learning-and-deep-learning)

<a id="ai-machine-learning-and-deep-learning"></a>

### AI, machine learning and deep learning

_Concept_

Three nested terms. **Artificial intelligence (AI)** is any system that performs tasks we associate with intelligence. **Machine learning** is the part of AI where systems learn from data. **Deep learning** is the part of ML that uses large neural networks. A single "neuron" in a neural network is essentially a linear regression (a weighted sum of inputs plus an intercept) passed through a non-linear function.

The xkcd joke in the deck has a true core: most ML really is linear algebra, with data stored as matrices that models multiply by learned numbers.

<a id="statistical-modeling"></a>

### Statistical modeling

_Concept_

Combining data with statistical theory to model the real-world relationship between predictors and a response: "a more down-to-earth way of saying machine learning". Statisticians and ML engineers often describe the same thing in different words, which is why so many terms below have synonyms.

Statistician Leo Breiman described "two cultures" (2001): one focused on **inference** (understanding how x relates to y and testing hypotheses) and one on **prediction** (getting accurate y values, even from black-box models). Linear regression sits in both camps. "All models are wrong, but some are useful" (George Box): a model is a simplified representation of reality that never captures everything but can capture enough to be useful.

<a id="predictor-and-response-and-their-many-names"></a>

### Predictor and response (and their many names)

_Concept_

Because ML grew out of several fields, the same thing has many names. The **predictor** is also called an independent variable, feature, covariate or input (e.g. square footage). The **response** is also called a dependent variable, target, label or output (e.g. house price). Conventionally, predictors go in `X` and the response in `y`.

Tip for spotting which is which: the dependent variable is the thing **being predicted or affected**. "The effect of X on Y" and "predicting Y using X" give it away. "Independent variable" is a slightly misleading name: predictors are often correlated with each other; it means "the variable we vary", not "statistically independent".

See also: [Feature (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#feature-independent-variable), [Target (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#target-dependent-variable)

<a id="types-of-variables"></a>

### Types of variables

_Concept_

The type of each variable decides how a model can use it, and the type of the **target** decides the problem type (numeric → regression, categorical → classification).

- **Numerical, continuous:** any value in a range (height, weight, temperature). Scales may round, but the quantity is continuous.
- **Numerical, discrete:** countable with gaps (bedrooms: you can't have 2.5).
- **Categorical, nominal:** labels with no order (eye colour, city).
- **Categorical, ordinal:** labels with an order but uneven gaps (poor < fair < good < excellent).

Linear models do arithmetic, so categorical features must be converted to numbers first (one-hot encoding). **Month** is the tricky case: ordinal, and also **cyclical** (December sits next to January). Coding it 1–12 tells a linear model December is "12 times January", which is why the bikeshare notebook treats season and hour as categories.

See also: [Nominal (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#nominal), [Ordinal (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#ordinal), [Dummy variables in regression](#dummy-variables-in-regression)

<a id="supervised-learning"></a>

### Supervised learning

_Concept_

Learning from data where the answer is known: "given X, can we predict Y?" Every row has a label, and the model learns to map features to that label. Supervised learning splits into **regression** and **classification**.

See also: [Unsupervised learning](#unsupervised-learning), [Regression vs classification tasks](#regression-vs-classification-tasks)

<a id="unsupervised-learning"></a>

### Unsupervised learning

_Concept_

Learning from data with **no** y-variable: "what does X look like, really?" Used to find groups (clustering) or to compress many variables into a few (dimensionality reduction). There's no right answer to score against, so results need human interpretation.

See also: [Clustering](#clustering), [Principal component analysis (PCA)](#principal-component-analysis-pca)

<a id="reinforcement-learning"></a>

### Reinforcement learning

_Concept_

A third family where an **agent** takes actions, receives rewards or penalties, and learns by trial and error to do better next time (game-playing AIs such as AlphaGo, robotics). It's a well-established field, also used in training today's large language models, but it's outside the scope of this course.

See also: [Semi-supervised learning](#semi-supervised-learning)

<a id="regression-vs-classification-tasks"></a>

### Regression vs classification tasks

_Concept_

The two kinds of supervised learning. **Regression**: the y-variable is numeric (tomorrow's closing price, this year's harvest yield, a home's price). **Classification**: the y-variable is a category (will they buy, will it rain, will they pass). A yes/no target is **binary** classification; three or more classes is **multiclass**.

Some problems can be framed either way: "how many mm of rain tomorrow?" is regression; "will it rain tomorrow?" is classification. "How long until this machine fails?" is regression; "will it fail this week?" is classification. Classification models usually output a **probability** for each class, which a threshold turns into a decision.

See also: [Regression vs classification (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#regression-vs-classification), [Binary and multiclass classification](#binary-and-multiclass-classification)

<a id="ingredients-of-a-supervised-model"></a>

### Ingredients of a supervised model

_Concept_

From the slides, to build a supervised model you need: a **problem statement**, a **data set**, a choice of **independent variables**, a choice of **model**, and a way to **measure how good the model is**. Leaving out the last one is the most common mistake.

Mapped onto the first notebook: problem statement, predict electricity demand in Victoria; data set, half-hourly demand and temperature for 2014; independent variable, temperature; model, OLS linear regression; measurement, MSE, RMSE and R². The steps influence each other: the problem decides what data you need, and the data limits which features you can use.

<a id="linear-regression"></a>

### Linear regression

_Concept, scikit-learn_

A supervised, white-box regression model that assumes y changes in a straight-line way with each x: y = β₀ + β₁x₁ + … + βₚxₚ + ε. Its big selling point is that the **coefficients are interpretable**. Its weakness is the list of assumptions (LINEM) that need to hold before you trust them.

**Linear** means "linear in the parameters": the prediction is a weighted sum. You can include x² as a feature and it's still linear regression, because the betas still just multiply things and get added.

A little history: the least-squares method was published by Legendre (1805) and Gauss (around 1809). The word "regression" comes from Francis Galton in the 1880s, who noticed that children of very tall parents tended to be tall but closer to average: their heights "regressed" towards the mean.

See also: [Ordinary least squares (OLS)](#ordinary-least-squares-ols), [LINEM assumptions](#linem-assumptions)

<a id="ordinary-least-squares-ols"></a>

### Ordinary least squares (OLS)

_Concept_

The standard way to fit a linear regression: choose the β values that **minimize the sum of squared residuals**, hence "least squares". It is one of the very few models with a closed-form solution, so it doesn't need gradient descent.

Why square the residuals rather than just add them?

1. Positive and negative residuals would otherwise cancel out: a terrible line could have residuals summing to zero.
2. Squaring punishes big misses much more than small ones (an error of 10 counts 100; an error of 2 counts 4).
3. The maths is smooth, which gives an exact formula for the best betas.

For simple regression the formulas are β̂₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)² and β̂₀ = ȳ − β̂₁x̄. One consequence: **the OLS line always passes through the point (x̄, ȳ)**.

> **Watch out:** Because squaring magnifies large errors, OLS is sensitive to outliers: one extreme point can pull the line noticeably.

See also: [Matrix form of OLS](#matrix-form-of-ols), [Line of best fit](#line-of-best-fit)

<a id="simple-linear-regression"></a>

### Simple linear regression

_Concept_

Linear regression with **one** predictor: ŷ = β̂₀ + β̂₁x. In the notebook: electricity demand predicted from temperature, giving ŷ = 2.32 + 0.11·temp.

<a id="multiple-linear-regression"></a>

### Multiple linear regression

_Concept_

Linear regression with **more than one** predictor: ŷ = β̂₀ + β̂₁x₁ + … + β̂ₚxₚ. Adds one assumption (no multicollinearity) and changes how you interpret coefficients ("holding all else constant").

<a id="error-term"></a>

### Error term (ε)

_Concept_

The part of y the model can't explain: random noise inherent in any prediction "because nothing will be perfect". It captures variables we didn't measure, measurement error and genuine randomness. The model is y = β₀ + β₁x + ε; observation = model + error.

**Error vs residual:** ε is the gap from the **true, unknown** line; the **residual** e = y − ŷ is the gap from **our fitted** line. Residuals are our observable stand-in for errors, which is why assumption checks use residuals. Sign convention: a **positive** residual means the actual value is above the line (the model **under**estimated). For an OLS model with an intercept, the residuals always sum (and average) to exactly zero.

```python
resids = y - y_pred
resids.mean()        # 0 (up to rounding) for OLS with an intercept
```

See also: [Residual (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#residual)

<a id="intercept"></a>

### Intercept (β₀)

_Concept, scikit-learn_

The expected value of y when every x is zero, i.e. where the line crosses the y-axis. It doesn't always make sense: with y = 12000 + 250x for house price and square footage, "a house with no floor area sells for $12,000" is meaningless. In scikit-learn: `model.intercept_`.

The house example is an **extrapolation**: no house in the data is anywhere near 0 sq ft. Even when its interpretation is meaningless, we still need the intercept to anchor the line. To make it meaningful, **centre** the feature (subtract its mean): the intercept then becomes "the predicted price of an average-sized house".

<a id="slope-coefficient"></a>

### Slope / coefficient (β₁)

_Concept, scikit-learn_

The rate of change between x and y: **for every 1-unit increase in x, we expect y to change by β₁**. In the house example, each extra square foot adds about $250. In scikit-learn: `model.coef_` (an array, one value per feature).

Worked example with y = 12000 + 250x: a 1,000 sq ft house is predicted at 12,000 + 250 × 1,000 = $262,000, and a 1,100 sq ft house at $287,000. The 100 sq ft difference is worth 100 × $250 = $25,000. The sign gives direction, the size gives strength **in the units of x and y** (always state them). The word "expect" matters: individual houses deviate, and this is an **association**, not necessarily a causal effect.

<a id="extrapolation"></a>

### Extrapolation

_Concept_

Predicting for x-values far outside the range of the training data. The fitted line may not hold there: the electricity model was trained on temperatures above 15 °C, so its "demand at 0 °C" (the intercept, 2.32 GW) is a guess, not evidence. Predictions are most reliable inside the range the model has seen.

<a id="hat-notation"></a>

### Hat notation (ŷ, β̂)

_Concept_

A hat means "estimated" or "predicted". **ŷ** ("y-hat") is a prediction; **β̂** is a coefficient estimated from data. A bold **y** means the whole vector of responses, and **n** is the sample size.

Betas **without** hats describe the true relationship in the whole population, which we can never know exactly; β̂ ("beta-hat") is our estimate from one sample, and a different sample would give slightly different estimates. That uncertainty is what p-values and confidence intervals describe. There's no ε in the ŷ equation because a prediction is our best guess of the average; the error is unknown at prediction time.

<a id="line-of-best-fit"></a>

### Line of best fit

_Concept, seaborn_

The straight line that minimizes the sum of squared residuals. seaborn draws it for you, which is a quick visual check before fitting anything.

```python
sns.regplot(x="temp", y="demand", data=elec, ci=None,
            scatter_kws={"s": 1}, line_kws={"color": "red"})
plt.axhline(elec["demand"].mean(), color="orange", lw=3)  # the null model, for comparison
```

`sns.lmplot()` does the same but as a figure-level plot.

Before fitting, read the scatter plot for **direction** (up or down), **form** (straight or curved), **strength** (how tightly the dots cluster, summarized by the correlation r) and **unusual points** (outliers). The variable you want to predict goes on the y-axis.

<a id="interpreting-coefficients"></a>

### Interpreting coefficients

_Concept_

**Simple regression:** "For every 1-unit increase in x, we expect y to increase by β." **Multiple regression:** add "**holding all other variables constant**". **Dummy variables:** the effect is relative to the baseline category. From the bikeshare model: "holding temperature constant, it being Summer means 105 fewer riders per hour relative to Winter"; "for every one-degree increase in temperature, about 14 more riders per hour".

A worked dummy-variable example: if price = 50 + 10·Spring + 20·Summer + 5·Fall (in $k) with Winter as the baseline, predictions are Winter 50, Spring 60, Summer 70, Fall 55: "summer houses sell for $20k more than winter ones, on average, all else equal". The negative Summer coefficient in the bikeshare model surprises students: summer is hot, and once temperature is accounted for, "being summer" adds nothing extra.

> **Watch out:** Always check the units. In the Advertising data, budgets are in thousands of dollars and sales in thousands of units, so a radio coefficient of 0.1885 means each extra $1,000 of radio spend is associated with about **189 more units sold** . It's easy to misread this as "$0.19M more sales"; write the units out in full whenever you interpret a coefficient.

See also: [Dummy variables in regression](#dummy-variables-in-regression), [Coefficients are not importance](#coefficients-are-not-importance)

<a id="coefficients-are-not-importance"></a>

### Coefficients are not importance

_Concept_

A common pitfall: coefficients don't tell you which variable has the biggest impact. Each one is in its feature's own units, so a coefficient of 250 for square feet and 15,000 for bedrooms can't be compared. You can only compare them if all features are on the same scale (for example, after standardizing).

The bikeshare notebook shows this directly: measuring temperature in °F instead of °C shrinks the coefficient by a factor of 1.8, yet the predictions are identical. A coefficient of 0.5 per square foot and 20,000 per bedroom doesn't mean bedrooms matter 40,000 times more; adding a bedroom is simply a much bigger change than adding a square foot. Standardize first, and each coefficient becomes "change in y per one-standard-deviation change in x", which is comparable. With multicollinearity, coefficients can't be trusted for ranking at all.

See also: [Feature importance](#feature-importance)

<a id="linearregression"></a>

### LinearRegression

_scikit-learn_

scikit-learn's OLS model, in `sklearn.linear_model`. It follows the six-step workflow below.

```python
from sklearn.linear_model import LinearRegression

X = elec[["temp"]]          # 2D: a DataFrame
y = elec["demand"]          # 1D: a Series
lr = LinearRegression()
lr.fit(X, y)
lr.intercept_, lr.coef_     # (2.32..., array([0.11...]))
```

`LinearRegression()` adds the intercept automatically (`fit_intercept=True` by default), so you never add a column of 1s yourself. Every scikit-learn model follows the same create → `.fit()` → `.predict()` pattern, so learning it here transfers to dozens of models.

See also: [The modeling workflow in scikit-learn](#the-modeling-workflow-in-scikit-learn)

<a id="the-modeling-workflow-in-scikit-learn"></a>

### The modeling workflow in scikit-learn

_scikit-learn, Concept_

The six core steps used for linear regression and then reused for every model in the course:

1. **Assemble** predictors `X` (a DataFrame) and target `y` (a Series).
2. **Instantiate** the model.
3. **Fit** the model on training data.
4. **Check out and interpret** the coefficients.
5. **Make predictions** with `.predict()`.
6. **Evaluate** the model, ideally against a baseline.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LinearRegression()                       # 2
model.fit(X_train, y_train)                      # 3
dict(zip(X.columns, model.coef_))                # 4
y_pred = model.predict(X_test)                   # 5
metrics.root_mean_squared_error(y_test, y_pred)  # 6
```

See also: [fit / predict (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#fit-predict)

<a id="x-is-2d-y-is-1d"></a>

### X is 2D, y is 1D

_scikit-learn_

scikit-learn expects `X` to be two-dimensional (n rows × p features) even when there is only one feature, and `y` to be one-dimensional (length n). That's why the notebooks write `elec[["temp"]]` (double brackets → DataFrame) but `elec["demand"]` (single brackets → Series).

> **Watch out:** `lr.fit(elec["temp"], y)` raises "Expected 2D array, got 1D array". Use double brackets or `.reshape(-1, 1)`. The same applies when predicting one new value: `lr.predict([[25]])`, not `lr.predict([25])`.

See also: [reshape(-1, 1) (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#reshape-1-1)

<a id="coef-and-intercept"></a>

### .coef* and .intercept*

_scikit-learn_

Fitted attributes (note the trailing underscore: scikit-learn's convention for "learned from data"). Pair coefficients with column names to read them.

```python
pd.Series(model.coef_, index=X.columns).sort_values()
```

> **Watch out:** It's tempting to pair names and coefficients with `set(zip(X.columns, model.coef_))`, but sets have no order, so the output can come back shuffled; `dict(zip(...))` or a `pd.Series` keeps the pairing readable.

<a id="score"></a>

### .score()

_scikit-learn_

A built-in evaluation shortcut on every model. For **regressors** it returns **R²**; for **classifiers** it returns **accuracy**. `lr.score(X_test, y_test)` is the same as `metrics.r2_score(y_test, lr.predict(X_test))`.

> **Watch out:** `model.score()` returns ordinary R², not adjusted R², and it's easy to label it wrongly. Compute adjusted R² yourself or read it from a statsmodels summary.

See also: [R² (coefficient of determination)](#r2-coefficient-of-determination), [Accuracy](#accuracy)

<a id="null-model-baseline-for-regression"></a>

### Null model (baseline) for regression

_Concept_

The simplest possible regression model: always predict the **mean** of y, whatever the input. Every real model should beat it. Its MSE is the variance of y, and its R² is 0 by definition.

```python
y_bar = np.mean(y)
null_mse = np.mean((y - y_bar) ** 2)
```

Its MSE equals the variance of y, and it's exactly the "null SSE" model inside the R² formula, so R² is literally "how much better than the baseline". Context is everything: a model RMSE of 50 is a big improvement if the baseline RMSE is 200, and nearly worthless if it's 52.

```python
baseline_rmse = np.sqrt(np.mean((y - y.mean()) ** 2))
```

See also: [Baseline model (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#baseline-model), [Null accuracy](#null-accuracy)

<a id="linem-assumptions"></a>

### LINEM assumptions

_Concept_

What has to hold for OLS results (especially the coefficients) to be trustworthy:

- **L**inearity: the relationship between x and y is roughly a straight line.
- **I**ndependence: observations don't affect each other.
- **N**ormality: residuals are roughly normally distributed.
- **E**qual variance (homoscedasticity): residuals have about the same spread for every predicted value.
- **M**ulticollinearity absent: predictors aren't strongly correlated with each other (multiple regression only).

Why they matter depends on your goal. For **prediction**, violations usually mean the model is less accurate than it could be. For **inference** (interpreting coefficients, p-values, confidence intervals), violations can make your conclusions misleading. Most checks look at the residuals.

See also: [Linear model assumptions (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#linear-model-assumptions)

<a id="linearity-l"></a>

### Linearity (L)

_Concept, matplotlib_

Checked with a scatter plot of each x against y (or `sns.pairplot`). A gentle curve may be acceptable; a strong curve suggests a transformation or polynomial terms.

```python
plt.scatter(elec["temp"], elec["demand"], s=1)
```

An apparent non-linearity often means a **missing variable**. The course's insurance-costs plot shows separate bands (plausibly smokers vs non-smokers): one straight line through all of them fits nobody, but adding a smoker indicator would fix much of it. For genuine curves, transform y or x, or add polynomial terms.

<a id="independence-i"></a>

### Independence (I)

_Concept_

Usually checked by thinking about how the data were collected rather than with a plot. The most common violation is **time series data**, where each observation depends on the previous ones. (The bikeshare data are hourly, so this assumption is "probably violated".)

Both walkthrough datasets are time series (half-hourly electricity demand, hourly bike rentals), so independence is very likely violated. It's easy to wave this assumption through: think about how the data were collected. The consequence: coefficients are usually still reasonable, but the model is **overconfident** (standard errors too small, p-values too optimistic). Other violations: repeated measurements on the same person, students in the same school, neighbouring houses.

See also: [Durbin–Watson statistic](#durbin-watson-statistic)

See also: [Statistical independence](#statistical-independence), [Generalized least squares (GLS)](#generalized-least-squares-gls)

<a id="durbin-watson-statistic"></a>

### Durbin–Watson statistic

_statsmodels_

A test for autocorrelation in regression residuals, printed in every statsmodels OLS summary. Values range from 0 to 4: **around 2 suggests no autocorrelation**, well below 2 suggests positive autocorrelation (typical of time series), and above 2 negative autocorrelation.

```python
from statsmodels.stats.stattools import durbin_watson
durbin_watson(results.resid)
```

<a id="normality-of-residuals-n"></a>

### Normality of residuals (N)

_Concept, matplotlib_

Plot a histogram of residuals; it should look roughly bell-shaped and centred on zero. Strong skew is a warning sign.

```python
resids = y - y_pred
plt.hist(resids, bins=50)
```

> **Watch out:** A common misconception is that x or y must be normally distributed. They don't: the assumption is about the **residuals**. It matters most for p-values and confidence intervals with small samples; mild non-normality rarely hurts predictions. Severe skew is still a red flag that the model is missing something.

See also: [Q–Q plot](#q-q-plot)

<a id="q-q-plot"></a>

### Q–Q plot

_statsmodels, SciPy_

A quantile–quantile plot compares the residuals' quantiles with those of a normal distribution. If the residuals are normal, the points fall along a straight diagonal line; curves at the ends signal skew or heavy tails. More precise than eyeballing a histogram.

```python
import statsmodels.api as sm
sm.qqplot(resids, line="45", fit=True)

from scipy import stats
stats.probplot(resids, dist="norm", plot=plt)    # SciPy alternative
```

See also: [Normal distribution (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#normal-gaussian-distribution)

<a id="equal-variance-homoscedasticity-e"></a>

### Equal variance / homoscedasticity (E)

_Concept, matplotlib_

Residuals should have roughly the same spread at every predicted value. Check with a residual plot: you want a random, patternless cloud around zero. A **fan shape** (spread growing with ŷ) is **heteroscedasticity**; a curve means the linearity assumption is also failing.

Consequence of heteroscedasticity: the coefficients are still unbiased on average, but their **standard errors are wrong**, so p-values and confidence intervals can't be trusted, and predictions are more uncertain in some regions than others. (Pronounced "HOH-moh-skeh-das-TIS-ih-tee": "homo" = same, "scedastic" = scatter.)

See also: [Residual plot](#residual-plot), [Homoscedasticity (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#homoscedasticity)

<a id="residual-plot"></a>

### Residual plot

_matplotlib_

Predicted values on the x-axis, residuals on the y-axis, with a horizontal line at zero. The single most useful diagnostic for a regression model.

```python
plt.scatter(y_pred, resids, s=1)
plt.axhline(0, color="orange")
```

One plot checks two assumptions: **uneven spread** (a fan or clusters) is an equal-variance problem, while a **curved band** is strictly a linearity problem. Real data rarely looks perfectly random; the standard is "no obvious pattern".

<a id="no-multicollinearity-m"></a>

### No multicollinearity (M)

_Concept, pandas_

Predictors shouldn't be strongly correlated with each other (e.g. number of cars a garage holds vs garage size). Multicollinearity doesn't hurt predictions but "completely destroys model inference": coefficients become unstable and unreliable. Check with a correlation matrix or heatmap and keep only one of a correlated pair.

```python
X.corr()
np.corrcoef(bike.temp, bike.atemp)     # temp and "feels like" temp: almost 1
```

Symptoms: coefficients swing wildly or even **flip sign** with small changes in the data, and their standard errors balloon. Fixes: drop one of a correlated pair, combine them into one feature, or use regularization (ridge regression shares the credit between correlated features).

See also: [Variance inflation factor (VIF)](#variance-inflation-factor-vif)

See also: [Multicollinearity (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#multicollinearity)

<a id="variance-inflation-factor-vif"></a>

### Variance inflation factor (VIF)

_statsmodels_

A per-feature measure of multicollinearity: how much a coefficient's variance is inflated because that feature can be predicted from the others. **1 = no correlation with the other features**; a common rule of thumb treats values **above 5–10 as concerning**. Better than a correlation matrix because it catches one feature being a combination of several others.

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

Xc = sm.add_constant(X)
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
                index=Xc.columns).drop("const")
vif.sort_values(ascending=False)
```

<a id="log-transforming-the-target"></a>

### Log-transforming the target

_Concept, NumPy_

A common fix when residuals are right-skewed, fan-shaped, or the relationship is slightly curved: model log(y) instead of y. The log compresses large values more than small ones, so it pulls in a long right tail, stabilizes spread that grows with y, and turns multiplicative relationships into additive ones. Prices, incomes and costs are classic candidates. Remember to undo it with `np.exp()` before reporting predictions.

```python
y_log = np.log(df["apps"])
lr_log = LinearRegression().fit(X_train, y_train_log)
preds = np.exp(lr_log.predict(X_test))   # back to the original units
```

Interpretation changes too: with log(y), a coefficient β roughly means "a one-unit increase in x is associated with about **100 × β % change** in y" (accurate for small β).

> **Watch out:** Don't confuse "log regression" (modelling log(y)) with **logistic regression** (a classification model). Forgetting to exponentiate means your "predictions" are in log units, and `np.log` fails on zeros: use `np.log1p` / `np.expm1` if y can be 0. Back-transforming gives roughly the median rather than the mean prediction, a small known bias.

See also: [Log transformation (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#log-transformation), [Comparing models on the same scale](#comparing-models-on-the-same-scale)

<a id="design-matrix"></a>

### Design matrix

_Concept_

The data matrix X used to fit a model: each **row** is an observation, each **column** a variable, and (mathematically) a first column of 1s for the intercept. scikit-learn adds the intercept column for you; statsmodels needs `sm.add_constant()`.

Notation: xᵢⱼ is the value in row i, column j (observation i's value of feature j). Shape: n rows × (p + 1) columns. The column of 1s lets the intercept be treated like any other coefficient: prediction = β₀×1 + β₁x₁ + β₂x₂ + …

<a id="matrix-form-of-ols"></a>

### Matrix form of OLS

_Concept_

With a design matrix, OLS is written compactly as **y = Xβ + ε**, and its closed-form solution is **β̂ = (XᵀX)⁻¹Xᵀy**. You won't compute it by hand, but it explains why perfectly collinear columns break the model: XᵀX can't be inverted.

```python
Xd = np.column_stack([np.ones(len(X)), X])           # add the column of 1s
beta_hat = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y       # same as LinearRegression
```

This formula is called the **normal equation**. Dimension check: (n × (p+1)) times ((p+1) × 1) gives (n × 1), which matches y. The inverse (XᵀX)⁻¹ only exists if no column of X can be built from the others, which is exactly what fails in the dummy variable trap.

<a id="dummy-variables-in-regression"></a>

### Dummy variables in regression

_Concept, pandas_

To use a categorical predictor with k levels, create k − 1 dummy (0/1) columns. The dropped level becomes the **baseline category**, and every dummy coefficient is interpreted relative to it. In the bikeshare model, Winter (season code 1) is dropped, so `season_3` means "Summer compared with Winter". Binary features (workday, holiday) can go in directly as a single 0/1 column.

Why not just code the levels 1, 2, 3, 4? That forces the categories into an order with equal spacing and a single slope, so level 4's effect would have to be four times level 1's. One-hot encoding lets each category have its own independent effect.

```python
X = bike[["season", "temp"]]
X = pd.get_dummies(X, columns=["season"], drop_first=True, dtype=int)
```

Choosing the baseline yourself (any baseline gives identical predictions; it only changes interpretation, so pick the most common or most natural comparison):

```python
df["season"] = pd.Categorical(df["season"], categories=["Winter", "Spring", "Summer", "Fall"])
pd.get_dummies(df, columns=["season"], drop_first=True, dtype=int)   # Winter is the baseline

from sklearn.preprocessing import OneHotEncoder
OneHotEncoder(drop=["Winter"])                                      # same idea in scikit-learn
```

> **Watch out:** `pd.get_dummies` drops nothing unless you pass `drop_first=True`, and then it drops the category that **sorts first**: for text labels that's alphabetical, so Fall/Spring/Summer/Winter drops **Fall**, not Winter. In recent pandas it also returns `True`/`False` columns unless you pass `dtype=int`.

See also: [One-hot encoding (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#one-hot-encoding), [Rank deficiency](#rank-deficiency)

<a id="rank-deficiency"></a>

### Rank deficiency

_Concept_

What happens if you keep **all** k dummy columns: they always add up to 1, which is exactly the intercept column. The design matrix is "rank-deficient" (a column is a combination of others), so the model can't be fitted uniquely. Dropping one dummy fixes it. In Unit 2 this was called the dummy variable trap.

**Rank** = the number of genuinely independent columns: five columns (intercept + four seasons) but only four independent ones. There are infinitely many equally good solutions (add 1,000 to every season coefficient and subtract 1,000 from the intercept: identical predictions), so the coefficients have no unique meaning.

> **Watch out:** In practice scikit-learn and statsmodels use a pseudo-inverse and will still return numbers, sometimes without any warning. Nothing crashes, but the individual coefficients are arbitrary, which makes the trap more dangerous.

See also: [Dummy variable trap (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#dummy-variable-trap)

<a id="feature-scale-in-linear-regression"></a>

### Feature scale in linear regression

_Concept_

Changing the units of a feature (Celsius to Fahrenheit) doesn't change OLS predictions; it just rescales the coefficient. The bikeshare model predicts the same riders for 25 °C as the Fahrenheit model does for 77 °F. Scale **does** matter for regularized models, kNN, SVMs, k-means and PCA.

See also: [Standardize before regularizing](#standardize-before-regularizing)

<a id="statsmodels-ols"></a>

### statsmodels OLS

_statsmodels_

An alternative to scikit-learn that gives a full statistical summary: coefficients with standard errors, p-values, confidence intervals, R², adjusted R², AIC/BIC and the Durbin–Watson statistic. Note the argument order (**y first**, the opposite of scikit-learn's `fit(X, y)`) and that you must add the intercept column yourself.

```python
import statsmodels.api as sm

X = sm.add_constant(elec[["temp", "workday"]])
y = elec["demand"]
ols = sm.OLS(y, X).fit()
ols.summary()
```

> **Watch out:** Without `sm.add_constant()`, the model is forced through the origin (no intercept), a classic bug. New data needs the constant too, or `predict` raises a shape error.

<a id="reading-a-statsmodels-results-object"></a>

### Reading a statsmodels results object

_statsmodels_

The fitted results expose everything in the summary as attributes. `.params` gives coefficients, `.rsquared` gives R², `.pvalues` gives p-values, and `.predict()` gives predictions (new data needs a constant column too).

```python
results.params["const"]        # or results.params.iloc[0]
results.rsquared
results.pvalues
results.t_test([0, 1]).pvalue   # test the second coefficient
Xnew = sm.add_constant([20, 21])
results.predict(Xnew)
```

**Reading a p-value:** the probability of seeing an estimate this far from zero if the true coefficient were actually zero. Below 0.05 means "unlikely to be zero". In the review lab, the slope's p-value is about 3.8 × 10⁻¹⁸ (overwhelming evidence of a relationship) while the intercept's is about 0.39 (not distinguishable from zero); the fitted line is Y = −0.0857 + 0.290X with R² ≈ 0.986.

> **Watch out:** `results.params[0]` raises a `KeyError` on pandas 3, because params is labelled ('const', 'X'). Use `results.params["const"]` or `.iloc[0]`.

See also: [p-value (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#p-value)

<a id="polynomial-regression"></a>

### Polynomial regression

_Concept_

Fitting a curve with linear regression by adding powers of x as extra features: x, x², x³, … The model is still "linear" in its coefficients. Higher degrees make the model more flexible ("responsive") and much more prone to overfitting.

In the review lab, a degree-5 polynomial on 20 points lowers the training SSE from about 0.79 to 0.28 and raises R² from 0.986 to 0.995: a closer fit to the training data, but a degree-5 curve on 20 points risks overfitting.

See also: [np.polyfit() and np.polyval()](#np-polyfit-and-np-polyval)

See also: [PolynomialFeatures](#polynomialfeatures), [Model complexity](#model-complexity)

<a id="polynomialfeatures"></a>

### PolynomialFeatures

_scikit-learn_

A transformer that creates powers and **interaction terms** (products such as x₁·x₂) of your features. Used in the review lab for a degree-5 fit and in the regularization notebook to deliberately overfit the wine data.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly.get_feature_names_out(X.columns)   # 'fixed_acidity', 'fixed_acidity^2', 'fixed_acidity density', ...
```

> **Watch out:** The number of columns explodes: 12 features at degree 2 become 90. Combine with regularization.

<a id="interaction-term"></a>

### Interaction term

_Concept_

A feature made by multiplying two others (x₁x₂), letting the effect of one depend on the value of the other. Written in model formulas like `y ~ x1 + x2 + x1x2`. Tree-based models capture interactions automatically.

In the generalization slides' model notation, `y ~ x1 + x2 + x1x2 + x3²`, "~" means "is modelled using", x₁x₂ is an interaction and x₃² a squared term.

---

<a id="s14-regression-metrics"></a>

## 14. Regression metrics

_How to put a number on "how good is this regression model?": distance-based error measures, R², and comparison with the null model. Examples use the advertising data (`Advertising.csv`), where `sales` is predicted from TV, radio and newspaper spend. "All models are wrong, but some are useful."_

**In your capstone:** Report one error metric in real units (RMSE or MAE) and R², always next to the null-model baseline and on test or cross-validated data.

<a id="euclidean-distance"></a>

### Euclidean distance

_Concept, NumPy_

The straight-line distance between two points, from Pythagoras: √Σ(aᵢ − bᵢ)². Treating y and ŷ as two points in n-dimensional space, their distance is a natural measure of error, and it's the starting point for SSE, MSE and RMSE. It reappears as the distance measure in kNN and k-means.

```python
y = np.array([4, 18, 9, 2, 7, 5])
y_hat = np.array([3, 15, 8, 1, 8, 12])
np.sqrt(np.sum((y - y_hat) ** 2))
```

Worked example: the residuals above are 1, 3, 1, 1, −1, −7, so the distance is √62 ≈ 7.87.

> **Watch out:** Convert lists to arrays first: subtracting two Python lists raises a `TypeError`.

See also: [Distance metrics (Euclidean, Manhattan, Minkowski)](#distance-metrics-euclidean-manhattan-minkowski)

<a id="distance-metric-the-four-rules"></a>

### Distance metric (the four rules)

_Concept_

In maths, a **distance metric** obeys four rules: (1) **non-negativity**, d(x, y) ≥ 0; (2) **identity**, d(x, y) = 0 only if x = y (zero error means perfect predictions); (3) **symmetry**, home-to-work equals work-to-home; (4) the **triangle inequality**, going directly from x to y is never longer than detouring via z.

Euclidean distance, RMSE (Euclidean distance ÷ √n) and MAE obey all four. SSE and MSE break the triangle inequality (squaring stretches distances), and R² isn't a distance at all. In everyday ML, "metric" just means "evaluation measure", so all of them are still called metrics.

<a id="sum-of-squared-errors-sse"></a>

### Sum of squared errors (SSE)

_Concept, NumPy_

Add up the squared residuals: SSE = Σ(yᵢ − ŷᵢ)² (also called RSS, the residual sum of squares). This is the quantity OLS minimizes. Worked example: y = [4, 18, 9, 2, 7, 5] and ŷ = [3, 15, 8, 1, 8, 12] give residuals 1, 3, 1, 1, −1, −7, squares 1, 9, 1, 1, 1, 49, and **SSE = 62**. Notice that the single residual of −7 contributes 49 of the 62: squaring makes big errors dominate.

```python
sse = np.sum((y - y_hat) ** 2)
```

> **Watch out:** SSE grows with the number of rows, so only compare SSEs computed on the **same** data. It's in squared units ("dollars squared").

<a id="mean-squared-error-mse"></a>

### Mean squared error (MSE)

_Concept, scikit-learn_

The average squared residual: MSE = (1/n)Σ(yᵢ − ŷᵢ)². Lower is better (as with any metric with "error" in its name). Squaring punishes large errors heavily. It's the traditional metric for estimating the bias–variance trade-off.

```python
np.mean((y - y_hat) ** 2)
metrics.mean_squared_error(y, y_hat)
```

From the SSE example, MSE = 62 ÷ 6 ≈ 10.33. Dividing by n means you can compare training MSE with test MSE even though the sets differ in size. MSE is also **differentiable** (smooth, no sharp corners), which is why it's the most common loss for regression and easy to minimize with calculus or gradient descent. On the training data, no other straight line has a lower MSE than the OLS line.

<a id="root-mean-squared-error-rmse"></a>

### Root mean squared error (RMSE)

_Concept, scikit-learn_

The square root of the MSE. Its advantage is that it's in the **same units as the target**, so an RMSE of 2,000 when predicting house prices means "on average, the model is off by roughly ±$2,000".

```python
np.sqrt(metrics.mean_squared_error(y, y_hat))
metrics.root_mean_squared_error(y, y_hat)      # scikit-learn 1.4+
```

> **Watch out:** `root_mean_squared_error` only exists in scikit-learn 1.4 and later. On older versions use `np.sqrt(mean_squared_error(...))` (or `mean_squared_error(..., squared=False)`, which has since been removed).

Think of it like standard deviation: MSE is an average squared distance; its square root brings you back to the original units. From the SSE example, RMSE = √10.33 ≈ 3.21 (which is also the Euclidean distance 7.87 ÷ √6). On the Advertising data the full model's RMSE is about 1.67 (thousand units) against a baseline RMSE of 5.20.

> **Watch out:** RMSE is not literally "the average amount the model is wrong by"; that's the MAE. RMSE is a **typical** error size that weights large errors more heavily, and it's always ≥ MAE. If RMSE is much bigger than MAE, a few large errors are present. Always judge it relative to the scale of y and the baseline RMSE.

<a id="mean-absolute-error-mae"></a>

### Mean absolute error (MAE)

_Concept, scikit-learn_

The average absolute residual: (1/n)Σ|yᵢ − ŷᵢ|. Also in the target's units, and less sensitive to a few huge errors than RMSE.

```python
metrics.mean_absolute_error(y, y_hat)
```

An error of 10 counts 10 in MAE but 100 in MSE, which is why MAE is less sensitive to outliers. Advertising data: MAE ≈ 1.25 vs RMSE ≈ 1.67. Its absolute value has a sharp corner at zero, which historically made it harder to optimize.

> **Watch out:** Don't confuse MAE with the **mean absolute deviation**, the average distance of values from their own mean. The names are easy to mix up, but MAE measures prediction errors.

<a id="r2-coefficient-of-determination"></a>

### R² (coefficient of determination)

_Concept, scikit-learn_

The **proportion of the variability in y that the model explains**: R² = 1 − SSE / SST. It usually runs from 0 to 1 (1 = perfect). "35.2% of the variability in electricity demand can be explained by temperature." On test data it can be negative if the model is worse than predicting the mean.

```python
1 - np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2)   # by hand
metrics.r2_score(y, y_hat)
lm.score(X, y)
```

> **Watch out:** Training R² always goes up when you add features, even pure noise. Judge a model on test or cross-validated R².

Advertising data: SSE ≈ 556.8 and null SSE ≈ 5,417.1, so R² = 1 − 556.8/5,417.1 ≈ **0.897**. For simple regression, R² equals the square of the correlation r between x and y. Cautions: a high R² doesn't prove the model is correct (assumptions, overfitting), and a low R² isn't always useless (human behaviour is noisy; 0.2 can be valuable). R² is unitless and complements RMSE rather than replacing it. Unlike error metrics, higher is better.

See also: [Adjusted R²](#adjusted-r2), [Evaluation metrics (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#evaluation-metrics-r-mse-rmse-mae-accuracy)

<a id="total-sum-of-squares-sst"></a>

### Total sum of squares (SST)

_Concept_

Σ(yᵢ − ȳ)²: the SSE of the null model that always predicts the mean. R² compares your model's SSE with this. (Also written SSTO; `ssto` in code.)

<a id="adjusted-r2"></a>

### Adjusted R²

_Concept_

A version of R² that penalizes extra features, so it only rises if a new feature improves the fit more than chance would: adj R² = 1 − (1 − R²)(n − 1)/(n − p − 1), with p features. statsmodels reports it in `.summary()` (`results.rsquared_adj`); scikit-learn doesn't have it built in.

```python
r2 = metrics.r2_score(y, y_hat)
n, p = X.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

Why plain R² can't be trusted for comparing feature sets: OLS can always give a new feature a coefficient of 0, so adding a feature never lowers training R², even pure noise. Adjusted R² only rises when a feature helps more than chance would, and it can be negative. Advertising check: R² 0.8972 with n = 200 and p = 3 gives adjusted R² ≈ 0.8956, barely different because p is small relative to n. Comparing on held-out data or with cross-validation is the stronger modern practice.

<a id="information-criteria-aic-bic-mallows-s-cp"></a>

### Information criteria (AIC, BIC, Mallows's Cp)

_Concept, statsmodels_

Metrics that balance goodness of fit against model complexity, like adjusted R². For all of them, **lower is better**, and they only compare models fitted to the same data.

- **AIC** = 2k − 2 ln(L), where k = number of parameters and L = the model's likelihood. **AICc** adds a small-sample correction.
- **BIC** = k ln(n) − 2 ln(L). Its penalty grows with sample size, so it favours simpler models than AIC.
- **Mallows's Cₚ** (pronounced "C-p") compares a smaller OLS model with the full model; values close to p suggest a good model.

statsmodels prints AIC and BIC in every regression summary (`results.aic`, `results.bic`).

> **Watch out:** Mallows's Cₚ is specific to OLS, but AIC and BIC apply to any model fitted by maximum likelihood, including logistic regression and ARIMA (where AIC is the go-to metric).

See also: [AIC (Akaike Information Criterion)](#aic-akaike-information-criterion)

<a id="sklearn-metrics"></a>

### sklearn.metrics

_scikit-learn_

The module holding every scoring function: regression (`mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`), classification (`accuracy_score`, `recall_score`, `precision_score`, `f1_score`, `confusion_matrix`, `roc_auc_score`), and clustering (`silhouette_score` and friends). The course usually imports the whole module: `from sklearn import metrics`.

Every function takes `(y_true, y_pred)` in that order. It doesn't matter for MSE or MAE, but it does for R².

<a id="choosing-a-regression-metric"></a>

### Choosing a regression metric

_Concept_

Use **RMSE** or **MAE** when you need to explain errors in the target's units to a stakeholder (MAE if a few big misses shouldn't dominate). Use **MSE** when comparing models mathematically or as a loss. Use **R²** for "how much of the variation do we explain". Whatever you choose, always compare against the null model and compute it on **test** data.

A good habit is to report **one error metric** (how big the errors are in real terms) and **one variance metric** (how much better than the baseline). The metric a model is trained with and the one you report don't have to match: a model trained with MSE can be reported to the business in MAE. Consistency across the models you compare matters more than which one you pick.

---

<a id="s15-generalization-bias-variance-and-cross-validation"></a>

## 15. Generalization, bias–variance and cross-validation

_Why a model that's perfect on its training data can be useless, how to measure that, and how to choose a model that will work on data it has never seen. Includes the Bias–Variance Tradeoff deck's "fitting sine" exercise and the College applications model workflow._

**In your capstone:** Decide your split and validation strategy before you model, and keep the test set untouched until the final check.

<a id="generalization"></a>

### Generalization

_Concept_

A model **generalizes** well if what it learned from a limited amount of training data works on new data. That's the goal of nearly every model you'll build: aiming for a balance between being underfit and overfit. "Having a good training MSE is unimpressive."

The course's warm-up (adapted from David Spiegelhalter's _The Art of Statistics_) makes this concrete without maths. Predict your chance of surviving 10 years by finding people like you. Match on two features and you find 10,000 people: a **stable but generic** estimate (high bias). Match on many features and you find 2 people: **specific but unstable**, since two slightly different matches could flip the answer between 0% and 100% (high variance). The best answer is in between.

<a id="underfitting"></a>

### Underfitting

_Concept_

The model is too simple to capture the shape of the data (a straight line through a parabola). **High error on training data and on new data.** It has high bias.

**Fixes:** a more flexible model, more or better features (including polynomial terms), less regularization. **More data does not fix high bias**: a straight line stays a straight line however many points you give it.

<a id="overfitting"></a>

### Overfitting

_Concept_

The model memorizes the training data, noise included, so it fails on new values. **Low training error, high error on new data.** It has high variance. Signs: training score much better than test score. "If we overfit our model, we're just modeling noise!"

**Fixes:** a simpler model, fewer features, regularization, more training data. Unlike bias, **more data does help** with variance, because noise averages out. The extreme case: a polynomial of degree n − 1 passes exactly through n points, and a lookup table "memorizes" the training data perfectly.

See also: [Reading train vs test scores](#reading-train-vs-test-scores), [Regularization](#regularization)

<a id="balanced-model"></a>

### Balanced model

_Concept_

The sweet spot between under- and overfitting: medium error on the training data and **low** error on new data. It captures the real pattern without memorizing the noise.

<a id="bias"></a>

### Bias

_Concept_

Error from a model's **inability to capture the true relationship**: plainly, "model badness". A straight line fitted to a sine curve is biased no matter how much data you give it. High bias = underfit. Loosely, bias is about "points within a model".

Sampling can be biased too. A poll that random-dials a phonebook misses people with mobile-only or unlisted numbers: polling 10,000 people from the same phonebook gives a more precise but **equally wrong** answer. That's sampling bias, related to but not the same as model bias: both are systematic errors that more of the same data won't fix.

See also: [Sampling bias](#sampling-bias)

<a id="sampling-bias"></a>

### Sampling bias

_Concept_

Systematic error from **how the data were collected**, so the sample doesn't represent the population. The classic case is the 1948 "Dewey Defeats Truman" headline: quota sampling produced unrepresentative samples and pollsters stopped polling weeks before the election, and Truman won. A larger sample collected the same way only makes the wrong answer more precise. Random sampling error (variance) shrinks with sample size; sampling bias doesn't.

See also: [Selection bias (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#selection-bias)

<a id="variance-model-variance"></a>

### Variance (model variance)

_Concept_

Error from a model being **too sensitive to the particular training data** it saw: how much its prediction for the **same point** changes when it's trained on a different sample. If you split your data in half and fit the same kind of model to each half, variance is why you get different models. High variance = overfit. Loosely, bias is visible within one model ("points within a model"), while variance only shows up when you compare many models ("a point between many models").

> **Watch out:** It's easy to confuse model variance with the variance of the data (its spread). In the bias–variance sense, variance is the spread of the model's **predictions** across training samples, even though it uses the same maths as the variance of a column.

<a id="irreducible-error"></a>

### Irreducible error

_Concept_

Error caused by the randomness of the universe, sometimes just called **noise**. No model can remove it. It sets a floor on how good any model can be.

<a id="bias-variance-decomposition"></a>

### Bias–variance decomposition

_Concept_

The expected squared error of a prediction splits into three parts:

**Err(x) = Bias² + Variance + Irreducible error**

where Bias² = (E[f̂(x)] − f(x))², Variance = E[(f̂(x) − E[f̂(x)])²], and the irreducible error is σ²ₑ. A residual mixes all three. You can't calculate bias and variance directly from one dataset; holdout methods estimate their combined effect.

Read E[…] as "averaged over many possible training sets". Bias² = (average prediction − truth)²; variance = average squared distance of a prediction from the average prediction; σ²ₑ = the noise in y itself.

> **Watch out:** The decomposition describes the **expected squared error averaged over many hypothetical training sets**, not any single residual from one model. It has **three** terms, and the true relationship f doesn't have to be linear.

<a id="bias-variance-trade-off"></a>

### Bias–variance trade-off

_Concept_

Making a model more complex lowers bias but raises variance; simplifying it does the reverse. Total error is U-shaped against complexity, and the goal is the **optimum model complexity** at the bottom of the U. Almost every tuning choice in the course (k in kNN, alpha in regularization, tree depth, C in SVMs) is a move along this trade-off.

See also: [Model complexity](#model-complexity), [Training vs test error curve](#training-vs-test-error-curve)

<a id="bullseye-diagram"></a>

### Bullseye diagram

_Concept_

The classic picture of bias and variance: the bullseye is the true value and each dart is one model's prediction. **Low bias, low variance**: tight cluster on the centre (the goal). **Low bias, high variance**: scattered around the centre. **High bias, low variance**: tight cluster off-centre. **High bias, high variance**: scattered and off-centre. Which one you get depends on the training data available.

The diagram comes from Scott Fortmann-Roe's essay "Understanding the Bias-Variance Tradeoff", a clear, non-technical read.

<a id="model-complexity"></a>

### Model complexity

_Concept_

How complicated the model is to describe. More complexity gives finer control over predictions, which can be a bad thing. Things that raise complexity: more variables, "wiggliness" (polynomial terms x², x³), less regularization, and more complex architectures (neural nets vs OLS).

Rule of thumb: **more complexity → lower bias, higher variance**. The three levers for managing it are complexity itself (degree, features, model type), the amount of data (more data lowers variance), and regularization (deliberately adding a little bias to cut variance).

<a id="training-vs-test-error-curve"></a>

### Training vs test error curve

_Concept_

Plot error against model complexity. **Training error** (blue) keeps falling as complexity rises. **Test / prediction error** (red) falls at first, bottoms out, then rises again: past that point the model is overfitting. It answers the classic question "why do we care about variance if more complexity always gives a more accurate model?": it's only more accurate on the data it has already seen.

The widening gap between the curves on the right is overfitting. Train-test splits and cross-validation are how we **estimate the red curve** in practice: compute the CV error at several complexity levels and choose the lowest.

> **Watch out:** It's model **complexity** beyond the sweet spot that signals overfitting: the further right of the minimum you go, the more overfit the model.

<a id="fitting-sine-exercise"></a>

### Fitting sine exercise

_Concept, NumPy_

From the Bias–Variance Tradeoff deck (after Caltech's _Learning From Data_, Yaser Abu-Mostafa). The target is y = sin(πx) for x from −1 to 1. Fit two **hypothesis sets** (the families of models you're allowed to choose from) using only **two points**: H₀: f(x) = b (all flat lines, one parameter) and H₁: f(x) = ax + b (all straight lines, two parameters). Team H₀ draws a flat line at the two points' average height; team H₁ draws the line through both points.

- **With the whole curve visible**, H₁ wins (E_out 0.20 vs 0.50). That's pure bias: there's only one best line, so no variance.
- **Averaged over every possible pair of points**, H₀ has bias² 0.50 + variance 0.25 = **0.75**, while H₁ has bias² 0.21 + variance 1.69 = **1.90**. The simpler model's expected error is less than half.

The lesson: **choose model complexity to match the amount of data you have, not the complexity of the truth.** With lots of data, H₁'s variance would shrink and it would win. This simulation reproduces those numbers:

```python
rng = np.random.default_rng(42)
xs = np.linspace(-1, 1, 201); truth = np.sin(np.pi * xs)
preds0, preds1 = [], []
for _ in range(10_000):
    x = rng.uniform(-1, 1, 2); y = np.sin(np.pi * x)      # two random points
    preds0.append(np.full_like(xs, y.mean()))               # H0: f(x) = b
    a = (y[1] - y[0]) / (x[1] - x[0]); b = y[0] - a * x[0]
    preds1.append(a * xs + b)                               # H1: f(x) = ax + b
for name, P in [("H0", np.array(preds0)), ("H1", np.array(preds1))]:
    bias2 = np.mean((P.mean(axis=0) - truth) ** 2)
    var = np.mean(P.var(axis=0))
    print(f"{name}: bias2={bias2:.2f} var={var:.2f} total={bias2 + var:.2f}")
# H0: bias2=0.50 var=0.25 total=0.74
# H1: bias2=0.21 var=1.69 total=1.90
```

ḡ(x) ("g-bar") is the **average model** over all samples; the grey band around it shows the variance.

> **Watch out:** The "bias" figures in this comparison are **squared** bias, the term that appears in the decomposition.

<a id="np-polyfit-and-np-polyval"></a>

### np.polyfit() and np.polyval()

_NumPy_

`np.polyfit(x, y, deg)` fits a polynomial of the given degree by least squares and returns its coefficients; `np.polyval(coefs, x)` evaluates it. A quick way to see under- and overfitting live (older Java-based curve-fitting applets no longer run in modern browsers, so this is the easiest way to experiment). Degrees 0 and 1 underfit 10 noisy sine points, degree 3 looks about right, and degree 9 passes through every point and swings wildly between them.

```python
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 1, 10))
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 10)
grid = np.linspace(0, 1, 200)
for deg in [0, 1, 3, 9]:
    coefs = np.polyfit(x, y, deg)
    plt.plot(grid, np.polyval(coefs, grid), label=f"degree {deg}")
plt.scatter(x, y, color="black"); plt.ylim(-2, 2); plt.legend()
```

NumPy may warn that the degree-9 fit is poorly conditioned; that's expected.

<a id="out-of-sample-data"></a>

### Out-of-sample data

_Concept_

Data the model wasn't trained on. Performance on out-of-sample data is the honest measure of a model. Test sets and cross-validation folds are stand-ins for it.

<a id="holdout-method"></a>

### Holdout method

_Concept_

Any technique that sets data aside during training so it can be used to evaluate the model afterwards: train-test split, k-fold cross-validation, out-of-bag samples. Bias and variance are theoretical quantities, and holdout methods are how we estimate them in practice.

To compute bias and variance exactly you'd need the true relationship and many independent training sets, which we never have. Holdout methods are the practical substitute: a big gap between training and holdout error suggests high variance; high error on both suggests high bias.

<a id="train-test-split"></a>

### Train-test split

_scikit-learn_

Divide the data into a **training set** (used to build the model) and a **testing set** (used to validate it). Fit on training data, compute a metric on both, and compare. The default split is 75/25.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1653
)
```

The test set simulates future data. The **golden rule**: use it as little as possible while building the model. Every time you tweak a model based on test results, the test set becomes a little less "unseen", which is why more advanced workflows add a separate validation set.

> **Watch out:** `train_test_split` returns `X_train, X_test, y_train, y_test` in that order; mixing them up is a common bug. Without `random_state`, the split (and your scores) changes on every run.

See also: [train_test_split() (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#train-test-split), [test_size, stratify and shuffle](#test-size-stratify-and-shuffle)

<a id="test-size-stratify-and-shuffle"></a>

### test_size, stratify and shuffle

_scikit-learn_

Key arguments of `train_test_split`. `test_size` (or `train_size`) sets the proportion held out (default 0.25). `stratify=y` keeps the class proportions the same in both sets, which is important for imbalanced classification. `shuffle=False` keeps the original order, which is **required for time series** (you can't train on the future).

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_test_split(y, test_size=0.1, shuffle=False)     # time series
```

> **Watch out:** `train_size=0.3` and `test_size=0.3` are opposites, and it's an easy slip: `train_size=0.3` trains on only 30% of the data. Use `test_size` when you mean the share held out.

See also: [Time series train/test split](#time-series-train-test-split), [random_state (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#random-state)

<a id="train-test-ratio-rule-of-thumb"></a>

### Train-test ratio rule of thumb

_Concept_

You want as much training data as possible, but enough test data to represent everything the model will meet. **Small data** (say 200 rows): hold out more, maybe 50/50. **Huge data** (100M rows): 99/1 is fine because 1M test rows is plenty. Otherwise 75/25 or 70/30 is usually fine.

What matters for a reliable test estimate is the **absolute number** of test rows, not the percentage. With small data a single split is unreliable whichever ratio you pick, which is a strong argument for cross-validation.

<a id="reading-train-vs-test-scores"></a>

### Reading train vs test scores

_Concept_

Compare the same metric on training and test data. **Test much worse than train**: overfit (high variance). **Both bad**: underfit (high bias). **Both similar and good**: likely generalizes. Worked example: train RMSE 12.3 / test RMSE 658.2 is badly overfit; 233.9 / 252.1 is not overfit but weak; 132.7 / 200.5 is the best of the three because it has the best test score. Occasionally the test score beats the training score by chance; that's fine.

The lessons: pick the model with the best **test** (or cross-validated) score, then sanity-check the gap; and the **right** features matter more than the number of features (two well-chosen features beat the kitchen-sink model). A small gap doesn't mean the model is good: always compare against the baseline too.

<a id="k-fold-cross-validation"></a>

### k-fold cross-validation

_Concept_

Split the data into k **folds** (usually 5 or 10). Train on k − 1 folds and test on the remaining one; repeat so every fold is the test set once; then **average** the k scores. Example: fold MSEs 42.7, 45.1, 72.8, 38.2, 49.2 give a cross-validated MSE of 49.6. More robust than one split, and no data is "thrown away".

Fold 3 in that example (72.8) is much worse than the rest; if it had been your only test set, you'd have concluded the model was far worse than it is. Reporting **mean ± standard deviation** of the fold scores shows how much performance depends on which data the model sees. CV evaluates a **modelling approach**, not one fitted model: after choosing, fit the final model on all the data. 5 and 10 folds are popular because they balance the bias and variance of the estimate at reasonable cost.

See also: [cross_val_score()](#cross-val-score), [Cross-validation (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#cross-validation)

<a id="fold"></a>

### Fold

_Concept_

One of the k equal-sized chunks of data in k-fold cross-validation. k is not a tuning parameter: choose it large enough that averaging is meaningful but small enough that each fold isn't tiny.

<a id="kfold"></a>

### KFold

_scikit-learn_

Generates the train/test indices for each fold. Use `shuffle=True` with a `random_state` unless the order matters. You can loop over `kf.split()` to see each fold's model.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1318)
scores = []
for train_index, test_index in kf.split(X, y):
    lm = LinearRegression().fit(X.iloc[train_index], y.iloc[train_index])
    scores.append(lm.score(X.iloc[test_index], y.iloc[test_index]))
np.mean(scores)
```

See also: [StratifiedKFold](#stratifiedkfold)

<a id="cross-val-score"></a>

### cross_val_score()

_scikit-learn_

Runs the whole k-fold loop in one line and returns one score per fold. `cv` can be a number of folds or a `KFold` object; `scoring` chooses the metric (default: R² for regressors, accuracy for classifiers).

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(LinearRegression(), X, y, cv=kf)
scores.mean()
cross_val_score(knn, X_train_sc, y_train, cv=10).mean()
```

> **Watch out:** With a plain integer (`cv=5`), scikit-learn does **not** shuffle the rows for regression. If the data are in time order (bikeshare is), pass a shuffled `KFold`, or a `TimeSeriesSplit` if order genuinely matters.

See also: [scoring parameter](#scoring-parameter)

<a id="scoring-parameter"></a>

### scoring parameter

_scikit-learn_

Tells `cross_val_score`, `GridSearchCV` and friends which metric to use, as a string: `"r2"`, `"neg_mean_squared_error"`, `"neg_root_mean_squared_error"`, `"accuracy"`, `"recall"`, `"f1"`, `"roc_auc"`, `"balanced_accuracy"`, and more.

> **Watch out:** Error metrics are negated (`neg_…`) because scikit-learn always maximizes the score. Flip the sign back: `-cross_val_score(model, X, y, scoring="neg_mean_squared_error").mean()`.

<a id="leave-one-out-cross-validation-loocv"></a>

### Leave-one-out cross-validation (LOOCV)

_Concept_

The special case of k-fold where k = n: every observation is its own test set. Rarely used now because it means fitting n models.

LOOCV uses almost all the data to train each model, but its estimate has high variance and it's very slow for large n.

<a id="other-cross-validation-schemes"></a>

### Other cross-validation schemes

_scikit-learn_

Variants for data where a plain random k-fold would mislead:

- **StratifiedKFold:** keeps class proportions in every fold (classification; the default when you pass `cv=5` to a classifier).
- **TimeSeriesSplit:** always trains on the past and tests on the future, for time-ordered data.
- **GroupKFold:** keeps related rows (the same customer or patient) together, so the model can't "recognize" someone it trained on.

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
cross_val_score(model, X, y, cv=GroupKFold(n_splits=5), groups=df["customer_id"])
```

See also: [StratifiedKFold](#stratifiedkfold), [Time series train/test split](#time-series-train-test-split)

<a id="three-way-split-validation-set"></a>

### Three-way split (validation set)

_Concept_

Hold out a final **test set** first, then use cross-validation (or a separate **validation set**) on the rest to tune and choose models. The test set is touched only once, at the very end, so its score is an unbiased estimate. This is what you are doing whenever you run `GridSearchCV` on `X_train` and score on `X_test`.

<a id="cross-validation-pros-and-cons"></a>

### Cross-validation pros and cons

_Concept_

**Pros:** a more stable estimate of performance on new data, because it averages over k different test sets rather than relying on one lucky or unlucky split (one awkward fold can't dominate); and every row is used for testing once and for training k − 1 times. **Cons:** you fit the model k times (negligible for linear regression, prohibitive for a neural net that trains for hours); the fold scores aren't independent (the training sets overlap heavily), so the exact uncertainty of the average is hard to know; and each model trains on only (k − 1)/k of the data, so the estimate is slightly pessimistic. Train-test split is still useful when CV is too slow, for a quick confusion matrix, or as a final check.

See also: [Other cross-validation schemes](#other-cross-validation-schemes)

<a id="data-leakage-in-model-evaluation"></a>

### Data leakage (in model evaluation)

_Concept_

Information that won't be available at prediction time sneaks into training, making the model look better than it is. **Red flag: test performance that seems too good to be true usually is.** Causes:

- **Future information** used to predict the past, including random splits of time series.
- **Preprocessing leakage** (very common): fitting a scaler, imputer or encoder on the whole dataset before splitting. Always split first, then fit preprocessing on training data only (pipelines do this for you).
- **Target leakage**: features that are consequences or parts of the answer.
- **Duplicates**: the same or near-identical rows in both training and test sets.
- **Process changes** that add new information to the training set; and very large datasets, where subtle leaks are harder to spot.

Examples from the course: bikeshare `casual` + `registered` add up exactly to the target; College `accept` and `enroll` are consequences of applications (keeping them lifts test R² from 0.79 to about 0.92, and makes the model useless for predicting applications in advance); Iris3's `Id` column is sorted by species (a kNN on Id alone scores 100%); the possum `site` column perfectly identifies the target population. Others: a "treatment received" column when predicting disease, or a "collections agency contacted" flag when predicting loan default.

See also: [Data leakage (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#data-leakage), [Pipeline + GridSearchCV](#pipeline-gridsearchcv)

<a id="big-data-considerations"></a>

### Big data considerations

_Concept_

More data is generally good, but large datasets bring their own problems: subtle data leakage is harder to catch, compute costs and time can grow exponentially, and models can overfit spurious correlations that exist in the training data but not the real world.

With many rows and especially many columns, some features will correlate with the target **purely by chance** (Tyler Vigen's "Spurious Correlations" site has fun examples). More data reduces variance, but it doesn't fix bias and it doesn't fix leakage. (Computing costs typically grow in proportion to the data size or faster, depending on the algorithm.)

See also: [Spurious correlation (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#spurious-correlation)

<a id="refitting-for-production"></a>

### Refitting for production

_Concept_

Train-test split and cross-validation are for **choosing** a model. Once you've chosen, the usual practice is to refit that model configuration on **all** the data you have before deploying it, so it learns from as much information as possible.

The test or CV score you already have is your estimate of how the final model will perform. After deployment, **monitor** it: the world changes ("data drift") and models need retraining. Exceptions: some teams keep a final untouched test set for audits, and time series models are often refit on the most recent data.

<a id="signal-and-noise"></a>

### Signal and noise

_Concept_

You can think of ML as modeling the **signal** (the real relationship between X and y) and separating it from the **noise**. An overfit model has modeled the noise. It's always mathematically possible to fit training data perfectly; the art is fitting only the signal.

The term comes from Nate Silver's _The Signal and the Noise_, whose chapter on earthquake prediction shows how fitting overly complex patterns to historical data produced confident but misleading forecasts.

<a id="modeling-workflow-end-to-end"></a>

### Modeling workflow (end to end)

_Concept, scikit-learn_

The repeatable sequence a data scientist follows: problem statement → gather data → clean → EDA → split (and cross-validate) → build and validate the model → interpret and communicate. It's a **loop, not a line**: what you find in EDA or evaluation often sends you back to cleaning or feature choices.

The College applications walkthrough (predict applications from 777 US colleges' characteristics) strings everything together:

- **Clean:** `phd` loads as text because 29 rows contain "?"; replace with NaN, convert to float, drop (748 rows).
- **Engineer:** map `private` Yes/No to 1/0 (a two-level category needs only one column).
- **Fix impossible values:** a PhD rate of 103% and a graduation rate of 118% are capped with `np.minimum(..., 100)`.
- **Outliers:** dropping Rutgers (48,094 applications) is a **judgement call**: it's a real value, and the cost is that the model shouldn't be used for very large universities.
- **Features:** drop the name, the target, and `accept` and `enroll` (leakage).
- **Split, scale (fit on train only), fit, evaluate:** training R² ≈ 0.827, test R² ≈ 0.792, test RMSE ≈ 1,745 applications vs a baseline of ≈ 3,842.

```python
X = df.drop(columns=["university", "accept", "enroll", "apps"])
y = df["apps"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)          # transform only: no peeking at test data
lr = LinearRegression().fit(X_train, y_train)
lr.score(X_train, y_train), lr.score(X_test, y_test)
cross_val_score(LinearRegression(), X_train, y_train, cv=5).mean()   # ≈ 0.79
```

Then check LINE: the residuals are right-skewed with a fan shape, so try a log target and compare fairly (next entry).

> **Watch out:** Scaling doesn't change a plain `LinearRegression`'s predictions or R² (it changes the coefficients' units, which helps comparison); it becomes essential for Ridge, Lasso and kNN. On pandas 1.x, `df.columns.str.replace('.', '_')` treats "." as a regex and replaces every character; add `regex=False`.

See also: [StandardScaler (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#standardscaler-minmaxscaler-robustscaler)

<a id="comparing-models-on-the-same-scale"></a>

### Comparing models on the same scale

_Concept, scikit-learn_

Metrics computed on different target scales can't be compared. It's tempting to compare R² on log(applications) (test ≈ 0.77) with R² on raw applications (≈ 0.79), but they measure different things. Back-transform first and compare in the same units:

```python
preds_test = np.exp(lr_log.predict(X_test))         # undo the log
actual_test = np.exp(y_test_log)
np.sqrt(metrics.mean_squared_error(actual_test, preds_test))   # ≈ 2,746 (raw model ≈ 1,745)
```

On this split the log model is worse in raw terms, but the raw model predicts **negative** applications for 11 test colleges, which is impossible; the log model's predictions are always positive. Neither model is perfect, which is an honest and useful conclusion. For communicating, RMSE or MAE in real units is usually clearer than R².

---

<a id="s16-k-nearest-neighbors"></a>

## 16. k-Nearest Neighbors

_The first classification model: label a new point by asking its closest neighbours. Covers distance, choosing k, why scaling is essential, and how to tell whether a classifier beats a do-nothing baseline. Examples use the iris data (`iris.csv`) and the cell phone churn data (`cell_phone_churn.csv`)._

**In your capstone:** A quick, assumption-light baseline on small, scaled, numeric data. Tune k with cross-validation.

<a id="classification"></a>

### Classification

_Concept_

The branch of supervised learning that predicts a **categorical** response, often yes/no. "Given this person's demographics, will they vote for A or B?" "Given their financial status, will they default?" "Will it rain tomorrow?"

Classification models usually output a **probability** for each class (e.g. 0.7 for candidate A), which is then turned into a label. Loan default prediction is one of the most common real-world examples; defaults are rare, so the classes are imbalanced and plain accuracy misleads.

See also: [Regression vs classification tasks](#regression-vs-classification-tasks)

<a id="binary-and-multiclass-classification"></a>

### Binary and multiclass classification

_Concept_

**Binary** classification has two classes (0/1, yes/no, spam/not spam). **Multiclass** has three or more (iris species, handwritten digits 0–9). Some models handle multiclass natively (kNN, trees); logistic regression needs a little help.

See also: [Multinomial logistic regression](#multinomial-logistic-regression), [One-vs-rest (OvR)](#one-vs-rest-ovr)

<a id="k-nearest-neighbors-knn"></a>

### k-nearest neighbors (kNN)

_Concept_

Classify a new point using its **k closest neighbours** in the training data: "birds of a feather flock together". The algorithm: compute the distance from the new point to every training point, rank them, take the k nearest, and predict the **majority class** (classification) or the **average** (regression). Like guessing your favourite music by asking your five closest friends.

The analogy maps neatly: you = the new data point; friends = training points; "closest" = most similar on the features; five = k; majority vote = the prediction. "People like you also liked…" recommendations use the same nearest-neighbour idea.

See also: [Lazy learner](#lazy-learner)

See also: [KNeighborsClassifier](#kneighborsclassifier)

<a id="lazy-learner"></a>

### Lazy learner

_Concept_

A model that does no real work at training time. kNN's `.fit()` just stores the training data; all the computation happens at **prediction** time, when distances to every training point are calculated. That's why kNN trains instantly but predicts slowly on large datasets. (scikit-learn uses tree structures to speed up the search, but it still scales poorly.)

<a id="non-parametric-model"></a>

### Non-parametric model

_Concept_

A model that isn't summarized by a fixed equation with a fixed number of parameters; its complexity grows with the data. kNN is non-parametric: there are no coefficients, it just stores the training data. Decision trees are non-parametric too. Linear and logistic regression are **parametric**: after training you could throw the data away and keep only the betas.

The trade-off: parametric models are fast and interpretable but badly biased if their assumed shape is wrong; non-parametric models are flexible but data-hungry and prone to overfitting.

> **Watch out:** "Non-parametric makes no assumptions" is too strong. kNN still assumes that nearby points have similar labels and that distance is meaningful, which is exactly why scaling and feature choice matter so much.

<a id="distance-metrics-euclidean-manhattan-minkowski"></a>

### Distance metrics (Euclidean, Manhattan, Minkowski)

_Concept_

How "nearest" is measured. **Euclidean**: straight-line distance, √Σ(aᵢ − bᵢ)², the default. **Manhattan** ("taxi distance"): Σ|aᵢ − bᵢ|, moving only along grid lines. **Minkowski**: a general formula, (Σ|aᵢ − bᵢ|ᵖ)^(1/p), that is Manhattan when p = 1 and Euclidean when p = 2. In `KNeighborsClassifier`, choose with `p`.

Worked example for points (1, 2) and (4, 6): Euclidean = √(3² + 4²) = **5**; Manhattan = 3 + 4 = **7**. `KNeighborsClassifier` defaults to `metric="minkowski", p=2` (Euclidean). Manhattan can work better with many features.

> **Watch out:** In distance formulas the sum often runs to k or p, meaning the number of **features**, not the k in kNN.

See also: [Euclidean distance](#euclidean-distance), [Cosine similarity](#cosine-similarity)

<a id="choosing-k"></a>

### Choosing k

_Concept, scikit-learn_

k is a **hyperparameter**: chosen by you, not learned from the data. **Small k** (k = 1): wiggly boundaries that chase outliers, so low bias and high variance; training accuracy at k = 1 is always 100%, because every training point is its own nearest neighbour. **Large k**: smooth boundaries that ignore local detail, so higher bias and lower variance; at k = n every prediction is just the majority class. Odd values avoid ties in binary problems, and √n is a common starting point, but always validate.

Choose k by **cross-validation on the training data**, then check the test set once at the end:

```python
cv_scores = {}
for k in range(3, 32, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores[k] = cross_val_score(knn, X_train_sc, y_train, cv=5).mean()
best_k = max(cv_scores, key=cv_scores.get)

knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_sc, y_train)
knn.score(X_test_sc, y_test)          # the honest final check
```

> **Watch out:** It's tempting to pick k by **test** accuracy. That quietly tunes the model to the test set, so the test score stops being an honest estimate. Choose k by cross-validation; `GridSearchCV` automates it.

See also: [Hyperparameter](#hyperparameter), [GridSearchCV](#gridsearchcv)

<a id="decision-boundary"></a>

### Decision boundary

_Concept_

The line or surface where a classifier switches from predicting one class to another. Plotting it shows how flexible a model is: kNN's boundaries are jagged at k = 1 and smooth at k = 50; logistic regression's is a straight line; an RBF-kernel SVM's can curve.

As k increases, kNN's boundaries become **smoother and simpler**.

<a id="voronoi-diagram"></a>

### Voronoi diagram

_Concept_

A map that divides space into regions, each made up of all the points closest to one particular data point (or centroid). The decision regions of 1-nearest-neighbour and the clusters of k-means both form Voronoi cells.

> **Watch out:** Only the **k = 1** kNN plot is a true Voronoi diagram; for larger k the decision regions aren't Voronoi cells.

<a id="scaling-for-distance-based-models"></a>

### Scaling for distance-based models

_Concept, scikit-learn_

kNN measures distance, so a feature measured in thousands (square footage) swamps one measured in units (bedrooms). Always standardize first, fitting the scaler on training data only. The same applies to k-means, SVMs, PCA and regularized models.

```python
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

Worked example: two houses that differ by 1 bedroom and 500 square feet are √(1² + 500²) ≈ 500 apart; the bedroom difference contributes almost nothing. After standardizing, a difference of 1 means "one standard deviation" for every feature. In the churn data, account length (~100) and day minutes (~180) would swamp customer-service calls (0–9). In Iris all four features are centimetres with similar ranges, so scaling makes no difference there. `MinMaxScaler` (rescale to 0–1) is also valid.

See also: [Z-score standardization (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#z-score-standardization)

<a id="kneighborsclassifier"></a>

### KNeighborsClassifier

_scikit-learn_

scikit-learn's kNN classifier (`KNeighborsRegressor` for regression). Key hyperparameters: `n_neighbors` (k, default 5), `weights` (`"uniform"`: every neighbour votes equally; `"distance"`: closer neighbours count more), and `p` (1 = Manhattan, 2 = Euclidean).

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_train_sc, y_train, cv=10).mean()
knn.fit(X_train_sc, y_train)
knn.score(X_test_sc, y_test)          # accuracy
```

> **Watch out:** Voting ties aren't broken at random: scikit-learn deterministically favours the class that sorts first (e.g. "dry" beats "rain"). Another reason to use an odd k for binary problems.

<a id="knn-strengths-and-weaknesses"></a>

### kNN strengths and weaknesses

_Concept_

**Strengths:** few assumptions (non-parametric); works for classification (vote) and regression (average); simple to explain. **Weaknesses:** slow and memory-hungry on large data, since every prediction compares against every training row; sensitive to noisy, imbalanced and irrelevant features (with imbalance, most points' neighbours are from the majority class); works poorly with many predictors, because in high dimensions all points end up roughly the same distance apart and "nearest" stops meaning much; and many dummy columns add dimensions (the churn data's 51 states would add 50 columns). Best with a small number of well-chosen, scaled predictors.

Where nearest neighbours are used today: recommendation systems, **similarity search over embeddings** (finding similar products or documents in AI search and vector databases), anomaly detection, and filling in missing values (`KNNImputer`). Image and video tasks, which once used kNN widely, now use neural networks.

See also: [Curse of dimensionality (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#curse-of-dimensionality)

<a id="accuracy"></a>

### Accuracy

_Concept, scikit-learn_

The proportion of predictions that are correct: (TP + TN) / total. It's what `.score()` returns for every classifier. Easy to explain, but **misleading when classes are imbalanced**: a 99%-accurate test for a rare disease might just be saying "no" to everyone.

```python
metrics.accuracy_score(y_test, y_pred)
knn.score(X_test_sc, y_test)
```

Error rate = 1 − accuracy. Classic warning: if only 1% of people have a disease, a "test" that always says "healthy" is 99% accurate and catches nobody who's ill.

See also: [Null accuracy](#null-accuracy), [Balanced accuracy](#balanced-accuracy)

<a id="null-accuracy"></a>

### Null accuracy

_Concept, pandas_

The accuracy you'd get by **always predicting the most frequent class** in the training set: the baseline for any classifier. If your model doesn't beat it, it hasn't learned anything useful.

```python
most_freq_class = y_train.value_counts().index[0]
y_test.value_counts()[most_freq_class] / len(y_test)
y_test.value_counts(normalize=True)          # quick version
```

The course's churn data show why this matters: 85.5% of customers don't churn, so "always predict no churn" scores **85.5%**, and the best kNN reaches only about 87–88%. Most of the model's "correct" answers are customers who were never going to churn. For Iris (three equal classes) the null accuracy is about 33%. Because `True` = 1, `1 - y_train.mean()` gives the null accuracy for a True/False churn target.

<a id="dummyclassifier"></a>

### DummyClassifier

_scikit-learn_

A ready-made baseline model. `strategy="most_frequent"` always predicts the majority class, so its score is the null accuracy. `DummyRegressor(strategy="mean")` does the same for regression.

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy.score(X_test, y_test)
```

<a id="categorical-features-in-knn"></a>

### Categorical features in kNN

_Concept_

You can use dummy-coded categorical features in kNN, but each one adds a column, pushing up dimensionality and degrading performance; scaling 0/1 columns is also hard to justify. The notebook drops them for the churn model. Keep them only if they are critical to the problem.

```python
cat_cols = churn.select_dtypes("object").columns
X = churn.drop(columns=cat_cols).drop(columns=["churn"])
```

> **Watch out:** Two hidden traps in the churn data. `area_code` (415/408/510) is a category stored as a number, so kNN treats it as a distance. And each `*_charge` column is a fixed rate times its `*_mins` column (correlation 0.99999998), so minutes effectively count twice in every distance. Yes/no columns like `intl_plan` could simply be mapped to 0/1 rather than dropped.

---

<a id="s17-logistic-regression"></a>

## 17. Logistic regression

_Regression bent into a classifier: predicting the probability that y = 1 with an S-shaped curve, and reading its coefficients as changes in odds. Examples use graduate school admissions (`grad_admissions.csv`, admit predicted from GPA and GRE) and the possum data._

**In your capstone:** The go-to interpretable classifier. Start here before trees or boosting, and explain effects as odds ratios.

<a id="logistic-regression"></a>

### Logistic regression

_Concept, scikit-learn_

A supervised, white-box **classification** model. It predicts the **probability** that an observation belongs to class 1, then assigns a class using a threshold (0.5 by default). It's linear regression passed through a link function so outputs stay between 0 and 1. Fast, interpretable and by far the most common classifier.

```python
from sklearn.linear_model import LogisticRegression

X = admissions[["gpa"]]
y = admissions["admit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)          # accuracy
```

It belongs to the family of **generalized linear models (GLMs)**: a linear equation connected to the outcome through a link function. **Why is it called regression** if it classifies? Because under the hood it's a regression on the log-odds (a straight line on that scale); the S-shape appears when you convert back to probability. Unlike OLS it has **no closed-form solution** (fitted by an iterative optimizer) and is evaluated with classification metrics, not R².

On the admissions data: intercept ≈ −17.45, GPA coefficient ≈ 4.905, test accuracy 0.902 vs a baseline of 0.709. Adding GRE barely changes anything (0.904); GRE alone scores 0.709, the same as the baseline, because it predicts nobody gets in.

> **Watch out:** `grad_admissions.csv` (4,000 rows, GPAs up to 5.0) looks synthetic or augmented, so don't present the results as real admissions findings.

<a id="why-not-linear-regression-for-classification"></a>

### Why not linear regression for classification?

_Concept_

A straight line predicts any value from −∞ to ∞, so for a 0/1 target it happily predicts −0.3 or 1.4, which aren't probabilities. It also treats the relationship as constant, when the probability really changes most in the middle, and the residuals can't be normal or equal-variance when the outcome is only 0 or 1. Logistic regression "bends" the line into an S-curve that stays between 0 and 1.

See also: [Linear probability model](#linear-probability-model)

<a id="linear-probability-model"></a>

### Linear probability model

_Concept_

The name for fitting ordinary least squares to a 0/1 target. It's occasionally used in economics because its coefficients read directly as changes in probability, but it has the flaws above: predictions outside 0–1 and residuals that break the LINE assumptions. On the admissions data it predicts negative "probabilities" below a GPA of about 2.5.

<a id="logistic-sigmoid-function"></a>

### Logistic (sigmoid) function

_Concept_

The S-shaped curve that turns any number into a value between 0 and 1: p = e^z / (1 + e^z) = 1 / (1 + e^−z), where z = β₀ + β₁x₁ + … It's the inverse of the logit, and it converts log-odds back into probability.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Key values: z = 0 gives p = 0.5; large positive z gives p close to 1; large negative z gives p close to 0. For the admissions model the 50% crossover is around a GPA of 3.56; the steep middle section is where GPA matters most.

<a id="link-function-and-the-logit"></a>

### Link function and the logit

_Concept_

A **link function** connects a linear equation to the kind of output you need. Logistic regression uses the **logit**: logit(p) = ln(p / (1 − p)) = β₀ + β₁x₁ + … + βₚxₚ. It maps a probability (0 to 1) to the whole number line (−∞ to ∞), which is exactly what a linear equation produces.

Why the transform works: probabilities live between 0 and 1, odds between 0 and ∞, and log-odds between −∞ and ∞, which is exactly the range a straight line can produce.

> **Watch out:** It's easy to carry "+ ε" over from linear regression, but logistic regression has **no additive error term**: the randomness comes from the yes/no outcome itself.

<a id="odds"></a>

### Odds

_Concept_

The probability that something happens divided by the probability it doesn't: **odds = p / (1 − p)**. Convert back with **p = odds / (1 + odds)**. Shortcut from counts: odds = successes ÷ failures. Odds of 1 = a 50/50 chance; above 1 = more likely than not; below 1 = less likely.

| p   | odds | log-odds |
| --- | ---- | -------- |
| 0.2 | 0.25 | −1.39    |
| 0.5 | 1    | 0        |
| 0.8 | 4    | 1.39     |
| 0.9 | 9    | 2.20     |

The odds of rolling a 1 on a fair die are (1/6)/(5/6) = 1/5 = 0.2, written **1:5** ("one to five": one way to win, five to lose). For rare events odds and probability are close (0.2 vs 0.167); for common events they differ a lot.

> **Watch out:** Don't mix up the two conventions. **5:1** is the bookmaker's **odds against** ("5 to 1 against"). In statistics and logistic regression, "odds" always means odds **in favour**, p / (1 − p).

<a id="log-odds"></a>

### Log-odds

_Concept_

The natural log of the odds, ln(p / (1 − p)), also called the **logit**. This is what logistic regression actually predicts as a linear function of the features. Log-odds of 0 means p = 0.5; positive means more likely than not. In scikit-learn, `model.decision_function(X)` returns the log-odds.

> **Watch out:** The **odds** are the ratio p / (1 − p); the **log-odds** is its natural logarithm. The two are easy to blur.

See also: [decision_function()](#decision-function)

<a id="odds-ratio"></a>

### Odds ratio

_Concept_

The odds in one group divided by the odds in another: "how many times higher are the odds in this group?" From the slides' purchase table: returning customers' odds of buying are 22/5 = 4.4, new customers' are 2/15 ≈ 0.133, so returning customers have **33 times** greater odds of purchasing. If you fitted a logistic regression with a single 0/1 "returning customer" feature, e^β would be this odds ratio.

> **Watch out:** An odds ratio is not a probability ratio. Returning customers' purchase probability (22/27 ≈ 81%) is about **6.8 times** new customers' (2/17 ≈ 12%), not 33 times. Odds ratios exaggerate compared with probability ratios when events are common, so "33 times as likely" would be wrong.

<a id="interpreting-logistic-regression-coefficients"></a>

### Interpreting logistic regression coefficients

_Concept, NumPy_

A one-unit increase in xᵢ changes the **log-odds** of success by βᵢ, which is correct but hard to picture. Exponentiate to get an odds ratio: **a one-unit increase in xᵢ multiplies the odds of success by e^βᵢ**, holding all else constant. In the admissions model β ≈ 4.905 for GPA and e^4.905 ≈ 135: each extra GPA point multiplies the **odds** of admission by about 135. Half a point multiplies them by e^(4.905 × 0.5) ≈ 11.6.

```python
np.exp(logreg.coef_)
pd.Series(logreg.coef_[0], index=X.columns)     # pair with feature names
```

In probability terms the change depends on where you start: about **6% at GPA 3.0, 43% at 3.5 and 90% at 4.0**. The odds multiplier is the same everywhere, which is why odds are used for interpretation.

> **Watch out:** It's a common slip to say a GPA point makes admission "135 times more likely". That's a probability statement and it's wrong: exponentiated coefficients multiply the **odds**, not the probability. Also, scikit-learn's default L2 penalty slightly shrinks the coefficients, so they differ a little from statsmodels or textbook fits.

<a id="from-coefficients-to-probability"></a>

### From coefficients to probability

_Concept, NumPy_

Plug values into the equation to get log-odds, exponentiate to get odds, then convert to probability. It matches `.predict_proba()`.

```python
logodds = logreg.intercept_ + logreg.coef_[0] * 3.5     # GPA = 3.5
odds = np.exp(logodds)
prob = odds / (1 + odds)                                 # ≈ 0.43
logreg.predict_proba([[3.5]])[:, 1]                       # same answer
```

<a id="predict-proba"></a>

### .predict_proba()

_scikit-learn_

Returns the predicted **probability** of each class, one column per class in the order of `model.classes_`. Column 1 (`[:, 1]`) is P(y = 1). `.predict()` simply picks the class with the higher probability, i.e. uses a 0.5 threshold for binary problems.

```python
logreg.predict(X_test)[:10]                      # classes: 0 or 1
np.round(logreg.predict_proba(X_test), 3)        # [[P(y=0), P(y=1)], ...]
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
```

The two columns sum to 1. The first admissions test row has probabilities ≈ [0.894, 0.106], so it's predicted 0.

See also: [Classification threshold](#classification-threshold)

<a id="decision-function"></a>

### decision_function()

_scikit-learn_

Returns the model's raw score before it's turned into a probability. For logistic regression that's the **log-odds** (β₀ + β₁x₁ + …); applying the sigmoid to it gives `predict_proba(X)[:, 1]`. SVMs also have a `decision_function` (a signed distance from the boundary).

```python
z = logreg.decision_function(X_test)
np.allclose(1 / (1 + np.exp(-z)), logreg.predict_proba(X_test)[:, 1])   # True
```

<a id="plotting-the-logistic-curve"></a>

### Plotting the logistic curve

_matplotlib_

Sort the x-values, then plot predicted P(y = 1) to see the S-shape over the data.

```python
plt.scatter(X_test, y_test, s=10)
xs = X_test.sort_values("gpa")
plt.plot(xs, logreg.predict_proba(xs)[:, 1], color="red", lw=3)
```

<a id="logistic-regression-assumptions"></a>

### Logistic regression assumptions

_Concept_

Fewer than OLS: **L**inearity between each feature and the **log-odds** (not the probability); **I**ndependence of observations; and no **M**ulticollinearity (for multiple logistic regression). There's no normality or equal-variance assumption, because there are no continuous residuals to check. Practical requirements: enough data, especially enough examples of the rarer class; no extreme outliers dominating; and no perfect separation.

> **Watch out:** Don't carry all of LINEM over from linear regression: logistic regression assumes L, I and M only.

See also: [Perfect separation](#perfect-separation)

<a id="perfect-separation"></a>

### Perfect separation

_Concept_

When a feature (or combination) splits the classes perfectly, the textbook logistic regression coefficients run off to infinity, because the model can always make its predictions more confident. scikit-learn's default regularization hides this, so you get a finite but meaningless coefficient and suspiciously perfect accuracy.

The possum practice notebook is the course example: `site` perfectly identifies the target population (sites 1–2 are all Victoria), so including it gives 100% training accuracy. Drop `site` and training accuracy falls to 0.91, test to 0.77 (baseline ≈ 0.54), and the age coefficient **flips sign**, so interpreting age from the model that includes `site` isn't meaningful. Moral: **100% accuracy is suspicious**: look for leakage or separation.

<a id="log-loss-binary-cross-entropy"></a>

### Log loss (binary cross-entropy)

_Concept_

The loss function logistic regression minimizes (also called the **deviance**). It heavily penalizes confident wrong predictions. Unlike OLS there's no closed-form solution, so the β's are found by an iterative optimizer (the `solver`).

```python
metrics.log_loss(y_test, logreg.predict_proba(X_test))
```

See also: [Gradient descent](#gradient-descent), [solver and max_iter](#solver-and-max-iter)

<a id="solver-and-max-iter"></a>

### solver and max_iter

_scikit-learn_

`solver` picks the optimization algorithm that finds the coefficients (`"lbfgs"` is the default; `"liblinear"` supports L1 penalties on small data; `"newton-cg"`, `"saga"` are others). `max_iter` caps the number of iterations. If you see a **ConvergenceWarning**, scale your features, raise `max_iter`, or try another solver.

```python
LogisticRegression(max_iter=1_000)
LogisticRegression(solver="newton-cg")      # used for the possum data
```

The possum notebook uses `solver="newton-cg"` because the default `lbfgs` gives a convergence warning on its unscaled features.

<a id="regularization-is-on-by-default"></a>

### Regularization is on by default

_scikit-learn_

scikit-learn's `LogisticRegression` applies an L2 penalty with strength `C=1.0` unless told otherwise, which shrinks coefficients and makes them harder to interpret. For a classical, unpenalized model, switch it off.

```python
LogisticRegression(penalty=None)     # scikit-learn 1.2–1.7
LogisticRegression(C=np.inf)         # scikit-learn 1.8+ (penalty is deprecated)
```

> **Watch out:** This parameter has changed across versions, so check the documentation for the version you're running. scikit-learn 1.2 replaced the string `penalty='none'` with `penalty=None` (the string now raises an error), and 1.8 deprecates `penalty` itself: use `C=np.inf` for no penalty.

With unscaled features (GRE runs 200–800, GPA 1–5) the default penalty affects coefficients unevenly, another reason to scale.

See also: [C (inverse regularization strength)](#c-inverse-regularization-strength)

<a id="multinomial-logistic-regression"></a>

### Multinomial logistic regression

_Concept, scikit-learn_

Extends logistic regression to more than two classes by modelling all classes at once (the multinomial distribution instead of the Bernoulli). It's what `LogisticRegression` does automatically when y has 3+ classes with the default `lbfgs` solver, as in the iris example.

```python
iris_logreg = LogisticRegression().fit(X_train, y_train)   # y has 3 species
iris_logreg.coef_.shape                                    # (3, 4): one row per class
```

Also called **softmax regression**: one set of coefficients per class, with scores turned into probabilities that sum to 1. It's a genuine extension of logistic regression, not a hack. On Iris it scores training accuracy 0.975 and test 1.0 against a baseline of 0.30.

> **Watch out:** The `multi_class` parameter was deprecated in scikit-learn 1.5 and has since been removed; for one-vs-rest use `OneVsRestClassifier(LogisticRegression())`. Check the documentation for your version.

<a id="one-vs-rest-ovr"></a>

### One-vs-rest (OvR)

_Concept, scikit-learn_

The other multiclass "hack": for each class k, fit a binary model of "k vs everything else", then predict the class whose model gives the highest probability. Use `OneVsRestClassifier`.

```python
from sklearn.multiclass import OneVsRestClassifier
ovr = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)
```

<a id="logistic-regression-vs-decision-trees"></a>

### Logistic regression vs decision trees

_Concept_

A classic interview question. **Interpretability:** both are white-box (coefficients vs a flowchart). **Overfitting:** trees overfit easily; logistic regression rarely does. **Assumptions:** trees have none about the data's distribution; logistic regression has some. **Target:** trees do regression and classification; logistic regression only classification.

---

<a id="s18-classification-metrics"></a>

## 18. Classification metrics

_Accuracy isn't enough. The confusion matrix and the metrics built from it, how moving the probability threshold trades one kind of error for another, and the ROC curve and AUC. Examples use the breast cancer data (`load_breast_cancer`, malignant = 1) and the Whickham smoking data._

**In your capstone:** Choose your headline metric from the cost of each kind of error, and agree it with stakeholders before you look at results.

<a id="confusion-matrix"></a>

### Confusion matrix

_Concept, scikit-learn_

A table of actual vs predicted classes, named because it shows where the model gets "confused". For a binary problem it has four cells: **TN** (actual 0, predicted 0), **FP** (actual 0, predicted 1), **FN** (actual 1, predicted 0), **TP** (actual 1, predicted 1). The diagonal holds the correct predictions. Every classification metric in this section is calculated from it.

The deck's voter example: 100 people; positive = voted. TP 40, FP 20, FN 15, TN 25. In scikit-learn's layout that prints as `[[25, 20], [15, 40]]`.

```python
cm = metrics.confusion_matrix(y_test, y_pred)
pd.DataFrame(cm,
             columns=["pred benign", "pred malignant"],
             index=["actual benign", "actual malignant"])
tn, fp, fn, tp = cm.ravel()
```

> **Watch out:** Orientation varies. scikit-learn puts **actual in rows, predicted in columns**, negative class first: `[[TN, FP], [FN, TP]]`, so true positives are **bottom-right**. Textbooks and slides often use the opposite layout (predicted in rows, TP top-left), so always read the axis labels. `ConfusionMatrixDisplay` labels the axes, which avoids the confusion.

<a id="confusionmatrixdisplay"></a>

### ConfusionMatrixDisplay

_scikit-learn, matplotlib_

Plots a labelled, colour-coded confusion matrix directly from a fitted model.

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(knn, X_test_sc, y_test,
                                      display_labels=["benign", "malignant"],
                                      cmap="Blues")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)   # if you only have predictions
```

<a id="true-false-positives-and-negatives"></a>

### True/false positives and negatives

_Concept_

**Positive** means "predicted the class of interest" (malignant, spam, fraud), **true/false** means whether that prediction was right. A **false positive** (type I error) is a false alarm: a benign tumour flagged as malignant. A **false negative** (type II error) is a miss: a malignant tumour called benign. Which is worse depends on the problem, and that decides which metric to optimize.

"Positive" doesn't mean "good"; it means "the class we're looking for". True/False says whether the prediction was right.

See also: [Type I and Type II errors](#type-i-and-type-ii-errors)

<a id="type-i-and-type-ii-errors"></a>

### Type I and Type II errors

_Concept_

The hypothesis-testing names for the two mistakes. A **Type I error** is a false positive (rejecting a true null hypothesis: a false alarm); a **Type II error** is a false negative (failing to reject a false null: a miss). Rejecting H₀ ("nothing is going on") is like a classifier predicting "positive".

The matching rates: **α** (the significance level, often 0.05) is the probability of a Type I error, analogous to the false positive rate; **β** is the probability of a Type II error, analogous to the false negative rate (1 − recall); and **power = 1 − β** is the chance of detecting a real effect, analogous to recall.

Which error is worse depends on the problem. Cancer screening: usually the false negative (a missed treatment), though false positives cause stress and invasive follow-ups, so screening uses a sensitive first test and a specific confirmatory one. Online ads: neither is very costly, so advertisers tolerate many false positives.

> **Watch out:** This β has nothing to do with regression coefficients (β₀, β₁) or the F-beta score: three different uses of one Greek letter in the course.

See also: [Null hypothesis (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#null-hypothesis-h)

<a id="misclassification-rate"></a>

### Misclassification rate

_Concept_

The share of predictions that are wrong: (FP + FN) / total = 1 − accuracy. When tuning k in kNN, "error" usually means 1 minus the accuracy.

Voter example: accuracy = (40 + 25)/100 = 0.65, so the misclassification rate is 0.35. Predicting "vote" for everyone would score 55%, so the model beats the baseline by only 10 points.

<a id="recall-sensitivity-true-positive-rate"></a>

### Recall (sensitivity, true positive rate)

_Concept, scikit-learn_

Of all **actual** positives, how many did we catch? **TP / (TP + FN)**. Voter example: 40 / (40 + 15) ≈ 0.727. Optimize recall when missing a positive is costly (cancer screening, fraud). "Recall" is the ML / information-retrieval name; "sensitivity" is the medical name: a sensitive test rarely misses disease. Search-engine analogy: recall = how many of all the relevant pages were shown.

```python
metrics.recall_score(y_test, y_pred)
```

<a id="precision"></a>

### Precision

_Concept, scikit-learn_

Of all **predicted** positives, how many were right? **TP / (TP + FP)**. Voter example: 40 / (40 + 20) ≈ 0.667: when the model says someone will vote, it's right about two-thirds of the time. Optimize precision when false alarms are costly (spam filters deleting real email, interviewing unsuitable candidates). Search-engine analogy: precision = how many of the results shown were relevant.

```python
metrics.precision_score(y_test, y_pred)
```

**The airport scanner check** (actual in rows): 3 TP, 0 FN, 200 FP, 47 TN. Sensitivity = 3/3 = **100%** (every threat caught); specificity = 47/247 ≈ **19%**; precision = 3/203 ≈ **1.5%** (when the alarm sounds, it's real less than 2% of the time); accuracy = 50/250 = 20%, far below the 98.8% from calling everyone safe, which would miss every threat. Whether that's acceptable is a business decision about the cost of queues versus missed threats.

<a id="specificity-true-negative-rate"></a>

### Specificity (true negative rate)

_Concept_

Of all **actual** negatives, how many did we correctly call negative? **TN / (TN + FP)**. Voter example: 25/45 ≈ 0.556. A specific test rarely raises false alarms. scikit-learn has no `specificity_score`, so compute it from the confusion matrix.

```python
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
```

> **Watch out:** The denominator is all **actual** negatives (TN + FP), not just the true negatives.

<a id="false-positive-rate-fpr"></a>

### False positive rate (FPR)

_Concept_

FP / (FP + TN): the proportion of actual negatives wrongly flagged. **FPR = 1 − specificity**. It's the x-axis of the ROC curve.

> **Watch out:** FPR is not specificity: specificity is the true negative rate, and FPR is one minus it. A common exam and interview question.

<a id="f1-score"></a>

### F1 score

_Concept, scikit-learn_

The harmonic mean of precision and recall: **2 · (precision · recall) / (precision + recall)** = 2TP / (2TP + FP + FN). High only when both are high, so it's a good single number when you care about the positive class and classes are imbalanced.

```python
metrics.f1_score(y_test, y_pred)
```

Unlike a simple average, the harmonic mean is dragged down by whichever value is lower: precision 1.0 and recall 0.1 average 0.55 but give an F1 of only **0.18**. Voter example: F1 ≈ 0.696. F1 ignores true negatives, which is why it's popular when negatives are plentiful.

<a id="f-beta-score"></a>

### F-beta score

_Concept, scikit-learn_

A generalization of F1 that weights recall β times as much as precision. β = 2 favours recall; β = 0.5 favours precision.

```python
metrics.fbeta_score(y_test, y_pred, beta=2)
```

Voter example: F2 ≈ 0.714, higher than F1 because recall is the stronger of the two.

<a id="classification-report"></a>

### classification_report()

_scikit-learn_

Prints precision, recall, F1 and support (number of true cases) for every class, plus averages. Used in the boosting notebook.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

For multiclass problems it also prints macro and weighted averages.

See also: [Macro, weighted and micro averages](#macro-weighted-and-micro-averages)

<a id="pos-label"></a>

### pos_label

_scikit-learn_

When the classes are strings rather than 0/1, tell the metric which one counts as "positive". The imbalanced-data notebook uses the income labels.

```python
metrics.recall_score(y_test, preds, pos_label=">50K")
```

<a id="classification-threshold"></a>

### Classification threshold

_Concept, NumPy_

The probability cut-off for predicting class 1 (0.5 by default, and `.predict()` always uses 0.5). The threshold is arbitrary: a patient with a 49% predicted chance of cancer would be sent home while a 51% patient is investigated, though their risks are nearly identical. **Lowering** it predicts more positives: recall goes up, precision and specificity go down. The cancer notebook drops it to 0.1 to catch more malignant cases.

```python
y_pred_prob = knn.predict_proba(X_test_sc)[:, 1]
y_pred_class = np.where(y_pred_prob > 0.1, 1, 0)
metrics.confusion_matrix(y_test, y_pred_class)
```

On the Whickham survival data (logistic regression, positive = died): threshold 0.3 gives recall 0.85 / precision 0.63; 0.5 gives 0.67 / 0.70; 0.7 gives 0.51 / 0.92. The threshold should reflect the **costs** of each type of error.

See also: [Changing the threshold for imbalanced data](#changing-the-threshold-for-imbalanced-data)

<a id="precision-recall-trade-off"></a>

### Precision–recall trade-off

_Concept_

Because both depend on the threshold, you can almost always raise one by lowering the other. Decide which error is worse **before** you model ("framing the problem is a necessary first step"), then choose the metric and threshold to match.

<a id="roc-curve"></a>

### ROC curve

_Concept, scikit-learn_

The Receiver Operating Characteristic curve plots the **true positive rate** (y-axis) against the **false positive rate** (x-axis) at every possible threshold from 0 to 1. A perfect model hugs the top-left corner; a useless one follows the diagonal. One curve per model, and it doesn't change your model's actual predictions.

```python
metrics.RocCurveDisplay.from_estimator(lr, X_test, y_test)
plt.plot([0, 1], [0, 1], linestyle="--")      # random-guess line
```

> **Watch out:** `metrics.plot_roc_curve()` from older tutorials was removed in scikit-learn 1.2; use `RocCurveDisplay.from_estimator()`. Check the documentation when older code fails.

Each point on the curve is one threshold: threshold 1.0 predicts nothing positive (bottom-left, FPR 0 / TPR 0); threshold 0 predicts everything positive (top-right). The name comes from World War II radar engineering: judging how well radar receivers separated real aircraft from noise.

<a id="auc-roc-auc"></a>

### AUC (ROC AUC)

_Concept, scikit-learn_

The **area under the ROC curve**, a single number summarizing how well the model separates the classes across all thresholds. Its cleanest interpretation: **the probability that the model gives a randomly chosen positive a higher score than a randomly chosen negative**. **1.0**: perfect ranking. **0.5**: no better than chance. **Below 0.5**: worse than random; flipping the predictions gives 1 − AUC, so it usually signals a bug such as swapped labels. A rough guide: 0.7–0.8 acceptable, 0.8–0.9 good, above 0.9 excellent (field-dependent). Needs predicted **probabilities**, not classes, and it doesn't change when you move the threshold.

```python
y_pred_prob = lr.predict_proba(X_test)[:, 1]
metrics.roc_auc_score(y_test, y_pred_prob)
metrics.roc_auc_score(y, probs, multi_class="ovr")     # multiclass: one-vs-rest average
```

> **Watch out:** With heavy class imbalance, ROC AUC can look flattering; the precision–recall curve is often more informative.

See also: [Precision–recall curve](#precision-recall-curve)

<a id="precision-recall-curve"></a>

### Precision–recall curve

_Concept, scikit-learn_

Plots precision against recall at every threshold. Unlike the ROC curve it ignores true negatives, so it shows clearly how a model handles a rare positive class, where a high ROC AUC can hide poor precision. Summarized by the **average precision** (area under the curve); a no-skill model sits at the positive-class rate.

```python
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
```

<a id="macro-weighted-and-micro-averages"></a>

### Macro, weighted and micro averages

_Concept, scikit-learn_

How precision, recall and F1 are combined for **multiclass** problems, where there's no single positive class. Each class is treated in turn as "positive" vs the rest (one-vs-rest), then: **macro** = the simple average across classes (each class counts equally, so rare classes matter); **weighted** = weighted by class size; **micro** = pool every prediction first (for single-label problems this equals accuracy). Accuracy still works directly, and a multiclass confusion matrix is k × k with the confusions off the diagonal.

```python
metrics.f1_score(y_test, y_pred, average="macro")
metrics.recall_score(y_test, y_pred, average="weighted")
```

<a id="simpson-s-paradox"></a>

### Simpson's paradox

_Concept_

A trend that appears in pooled data reverses (or vanishes) once you account for another variable. In the Whickham data (1,314 women followed for 20 years), smokers **appear** more likely to survive (76% alive vs 69% of non-smokers), because the smokers were younger on average. Once age is in the logistic regression, smoking slightly **raises** the odds of death (odds ratio ≈ 1.04). Never interpret a single variable without asking what else differs between the groups.

See also: [Confounding variable (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#confounding-variable)

<a id="choosing-a-classification-metric"></a>

### Choosing a classification metric

_Concept_

Start from the cost of each mistake. **Accuracy**: classes balanced and errors equally bad (e.g. digit recognition, where misreading a 4 as a 3 or a 9 is equally wrong). **Recall**: misses are expensive. **Precision**: false alarms are expensive. **F1**: imbalanced classes and you care about the positive class. **Balanced accuracy**: you care about all classes equally. **ROC AUC**: comparing models' ranking ability independent of threshold.

Probability-based measures such as **log loss** and the **Brier score** score the predicted probabilities themselves rather than counting right and wrong answers. Remember the metric you optimize changes how you tune the model.

---

<a id="s19-imbalanced-classes"></a>

## 19. Imbalanced classes

_When one class vastly outnumbers another, a model can look brilliant by ignoring the rare class. How to spot it, measure it properly, and fix it with resampling, synthetic data, class weights or a new threshold, using the `imbalanced-learn` library and the adult census income data._

**In your capstone:** If your positive class is rare, judge models on recall, precision, F1 or balanced accuracy, and resample only inside a pipeline.

<a id="imbalanced-classes"></a>

### Imbalanced classes

_Concept, pandas_

When the classes in the target aren't equally represented (also called **unbalanced classes** or **imbalanced data**). In the West Nile Virus example, 99.9% of observations are "no WNV", so predicting "no WNV" every time scores **0.999 accuracy** while being useless. Very common in practice: fraud, disease, churn, equipment failure, defaults.

```python
claims["class"].value_counts(normalize=True)
```

Strictly, almost every real dataset is somewhat imbalanced. It becomes a problem when the minority class is rare (a rough guide is under 10–20%) **and** it's the class you care about; a 40/60 split usually needs no special treatment. The course's census data: 98.8% earn ≤ $50K and 1.2% earn > $50K, and a plain logistic regression scores 98.9% accuracy while catching only 5 of 75 high earners (recall 0.07).

> **Watch out:** The West Nile Virus 99.9% / 0.1% split is illustrative, not real data.

See also: [Class imbalance (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#class-imbalance)

<a id="majority-and-minority-class"></a>

### Majority and minority class

_Concept_

The **majority class** is the common one; the **minority class** is the rare one, and usually the one you care about. Models learn what they see most, so they struggle to predict the minority class.

Why: most models minimize overall error, and when 99% of examples are one class, predicting it almost always keeps error low. The few minority examples barely move the fit, and with only 300 of them the model has little evidence about what distinguishes them.

<a id="when-imbalance-matters"></a>

### When imbalance matters

_Concept_

Not every imbalance needs fixing. It matters when the minority class is important and the model is under-predicting it. Diagnose with the confusion matrix and minority-class recall rather than accuracy, and always compare with a null model.

<a id="balanced-accuracy"></a>

### Balanced accuracy

_Concept, scikit-learn_

The average of the recall for each class: (1/N) Σ TPᵢ / (TPᵢ + FNᵢ). A null model that always predicts the majority class scores 0.5 for a binary problem, however imbalanced the data. Use it when you care about all classes equally, and as the `scoring` for grid searches on imbalanced data.

```python
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, preds)
null_preds = ["<=50K"] * len(y_test)
balanced_accuracy_score(y_test, null_preds)     # 0.5
```

In the imbalanced-data notebook the plain logistic regression scores accuracy 0.988 but balanced accuracy only **0.533** (recall 0.067). The rebalanced models reach 0.72–0.75: random oversampling 0.722, class weights 0.741, SMOTE 0.743, ADASYN 0.749.

<a id="undersampling"></a>

### Undersampling

_Concept, imbalanced-learn_

Throwing away majority-class rows until the classes are balanced. The usual advice is not to undersample the majority class: you lose information, and on the census data it would shrink the training set from about 18,500 rows to about 450. That's a sensible default rather than a rule: with **large** datasets undersampling is fast and often works about as well. On the course data, random undersampling scored recall 0.69 vs 0.72 for oversampling, with similar precision and AUC.

```python
from imblearn.under_sampling import RandomUnderSampler
make_pipeline(StandardScaler(), RandomUnderSampler(random_state=42), LogisticRegression())
```

<a id="oversampling"></a>

### Oversampling

_Concept_

Increasing the number of minority-class rows by sampling them **with replacement**, so the model sees more of them during training. Risk: the duplicated rows can lead to overfitting the minority examples.

**Golden rule for every resampling method: resample only the training data.** The test set must keep its real-world imbalance, or your evaluation is meaningless.

<a id="randomoversampler"></a>

### RandomOverSampler

_imbalanced-learn_

Randomly duplicates minority-class rows. `sampling_strategy` sets the target ratio of minority to majority after resampling (1.0 = equal).

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

pipe = make_pipeline(RandomOverSampler(random_state=34),
                     LogisticRegression(max_iter=1_000))
gs = GridSearchCV(pipe,
                  {"randomoversampler__sampling_strategy": [.25, .5, .75, 1]},
                  scoring="balanced_accuracy", n_jobs=-1)
gs.fit(X_train_ss, y_train)
```

> **Watch out:** Run this grid with the default scoring (accuracy) and it prefers the **least** oversampling (CV accuracy 0.96 at 0.25, falling to 0.80 at 1.0), because accuracy rewards ignoring the minority class. Re-scored on `balanced_accuracy` it picks 0.75 (balanced accuracy 0.722, recall 0.587, precision 0.047). The metric you tune on decides the "best" model.

<a id="fit-resample"></a>

### fit_resample()

_imbalanced-learn_

How samplers are used directly: they return a resampled copy of the data. Apply it to **training data only**. Inside an `imblearn` pipeline this happens automatically during `fit`, and never during `predict` or `score`.

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
y_res.value_counts()                    # now balanced
```

<a id="synthetic-data-generation"></a>

### Synthetic data generation

_Concept_

Instead of copying minority rows, create new, plausible ("fake") minority observations by interpolating between existing ones. The main methods are SMOTE and ADASYN.

<a id="smote"></a>

### SMOTE

_imbalanced-learn_

**S**ynthetic **M**inority **O**ver-sampling **TE**chnique. For each minority point, pick one of its nearest minority neighbours and create a new point somewhere on the line between them.

```python
from imblearn.over_sampling import SMOTE

pipe3 = make_pipeline(SMOTE(random_state=34), LogisticRegression(max_iter=1_000))
gs3 = GridSearchCV(pipe3, {"logisticregression__C": [.1, 1, 10]},
                   scoring="balanced_accuracy")
gs3.fit(X_train_ss, y_train)
```

SMOTE uses nearest-neighbour distances (kNN again), so **scale inside the pipeline before SMOTE**. It adds more variety than duplicates, but can create unrealistic points, especially near the other class.

<a id="adasyn"></a>

### ADASYN

_imbalanced-learn_

**Ada**ptive **Syn**thetic sampling: like SMOTE, but generates more synthetic points next to the minority observations that a kNN classifier gets **wrong**, i.e. it focuses on the hard cases.

```python
from imblearn.over_sampling import ADASYN
pipe4 = make_pipeline(ADASYN(random_state=34), LogisticRegression(max_iter=1000))
```

On the census data, ADASYN gives the highest recall of the notebook's methods (0.747, precision 0.035, balanced accuracy 0.749).

> **Watch out:** If the model was trained on scaled data, every evaluation call needs the **scaled** test data (`X_test_ss`), including `ConfusionMatrixDisplay.from_estimator`. Passing the unscaled `X_test` is an easy slip that silently gives wrong results; a pipeline avoids it.

<a id="smotenc-and-smoten"></a>

### SMOTENC and SMOTEN

_imbalanced-learn_

SMOTE interpolates numbers, which makes no sense for categories. Use **SMOTENC** when features are a mix of categorical and continuous (tell it which columns are categorical) and **SMOTEN** when all features are categorical.

```python
from imblearn.over_sampling import SMOTENC
SMOTENC(categorical_features=[0, 3], random_state=34)
```

<a id="imbalanced-learn-pipeline"></a>

### imbalanced-learn pipeline

_imbalanced-learn_

`imblearn.pipeline.make_pipeline` (or `Pipeline`) works like scikit-learn's but supports samplers. Crucially, resampling happens **only when fitting**, i.e. on the training folds, so the test set and validation folds keep their real class balance and your scores aren't inflated. "Use `make_pipeline` to keep your test set safe."

> **Watch out:** scikit-learn's own `Pipeline` won't accept a sampler, and resampling before splitting leaks duplicated or synthetic rows into the test data.

Install with `pip install imbalanced-learn`; import as `imblearn`.

See also: [make_pipeline()](#make-pipeline), [Data leakage (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#data-leakage)

<a id="class-weights"></a>

### Class weights

_scikit-learn_

Option 3: make errors on the minority class count for more in the model's loss function instead of changing the data. `class_weight="balanced"` sets weights inversely proportional to class frequencies. Available on logistic regression, SVMs, trees and random forests. On the course's census data it gave **identical** results to random oversampling, with no extra rows and no extra library.

```python
logr5 = LogisticRegression(max_iter=1000, class_weight="balanced")
logr5.fit(X_train_ss, y_train)
```

<a id="resampling-is-a-trade-off"></a>

### Resampling is a trade-off

_Concept_

Rebalancing doesn't make a model better at telling the classes apart; it **moves the effective decision threshold**. Results on the census test set (75 high earners among 6,255 people):

| Method                               | Recall | Precision | AUC  |
| ------------------------------------ | ------ | --------- | ---- |
| Plain logistic regression            | 0.07   | 1.00      | 0.84 |
| Class weights / oversampling / SMOTE | 0.72   | 0.039     | 0.84 |
| Random undersampling                 | 0.69   | 0.041     | 0.84 |

Every method catches far more high earners (54 instead of 5) but flags about 1,300 people who aren't, and accuracy drops from 0.989 to about 0.79. AUC hardly moves. Which result is "better" depends on the cost of a miss versus a false alarm, so always report both sides. Better features (here, education and occupation) usually help more than any resampling trick.

<a id="changing-the-threshold-for-imbalanced-data"></a>

### Changing the threshold for imbalanced data

_Concept, NumPy_

Option 4: lower the probability cut-off for the minority class to get more minority predictions. Only sensible if the model is reasonably **calibrated**.

```python
prob = logr5.predict_proba(X_test_ss)[:, 1]
(prob >= .5).sum(), (prob >= .4).sum()   # more positives at the lower threshold
```

With the class-weighted census model, lowering the cut-off from 0.5 to 0.4 raises the number of predicted high earners from 1,278 to 1,932.

See also: [Classification threshold](#classification-threshold)

<a id="calibration"></a>

### Calibration

_Concept_

A classifier is **well calibrated** if its probabilities mean what they say: of all the cases it gives a 40% probability, about 40% are actually positive. Threshold-moving relies on calibration.

<a id="stratified-split-for-imbalanced-data"></a>

### Stratified split for imbalanced data

_scikit-learn_

With a rare class, a random split might put almost none of it in the test set. `stratify=y` keeps the same class proportions in train and test.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=6)
```

<a id="make-imbalance"></a>

### make_imbalance()

_imbalanced-learn_

A convenience function that makes a dataset more imbalanced on purpose, for practising these techniques.

```python
from imblearn.datasets import make_imbalance
X, y = make_imbalance(X, y, sampling_strategy={">50K": 300}, random_state=1)
```

<a id="imbalanced-data-checklist"></a>

### Imbalanced data checklist

_Concept_

From the notebook summary: think carefully about your metrics; change the grid search `scoring` (e.g. to `balanced_accuracy`); always build a null model; look at the confusion matrix; then try oversampling, SMOTE/ADASYN, class weights or a lower threshold, using the `imblearn` pipeline so the test set stays untouched. Getting more data is often the best fix of all. For image models with neural nets, having enough minority samples (or data augmentation) matters more than the ratio.

Other options: collect more minority data, or **reframe** the problem as anomaly detection when positives are extremely rare.

> **Watch out (course data):** `imbl_data.csv` has trailing colons in its column names (`fnlwgt:`, `hours-per-week:`); fix with `df.columns = df.columns.str.rstrip(":")`. Drop `fnlwgt`: it's a census **sampling weight** (how many similar people the row represents), not a characteristic of the person.

---

<a id="s20-regularization"></a>

## 20. Regularization

_Adding a penalty to a model's loss function so it can't make its coefficients too large, trading a little bias for a big drop in variance. Ridge, Lasso and ElasticNet, how to tune them, and the regularization that has been quietly switched on in logistic regression all along. Examples use the wine quality data (`winequality_merged.csv`) blown up with polynomial features to force overfitting._

**In your capstone:** Use when a linear model overfits (many or correlated features). Standardize first and choose alpha or C by cross-validation.

<a id="regularization"></a>

### Regularization

_Concept_

A method for constraining ("regularizing") the size of a model's coefficients by adding a **penalty term** to the loss function. It shrinks coefficients toward zero, reduces variance and overfitting, and tends to reduce variance more than it increases bias. Use it when evidence suggests high variance: training scores much better than test scores.

It can only help a model that is **overfitting**: regularizing an underfit model makes it worse. On the wine data, OLS scores training R² 0.407 vs test 0.220 (overfit); a cross-validated ridge scores training 0.348 (slightly lower) but test **0.295** (higher): exactly the intended trade.

<a id="why-linear-models-overfit"></a>

### Why linear models overfit

_Concept_

Linear models are low-complexity and high-bias, so they don't tend to overfit, but they can if you allow them high variance. Three causes from the slides: **irrelevant features** (the model learns a coefficient for every feature, signal or noise, especially when p is close to n); **correlated features** (the design matrix becomes nearly singular, so estimates swing wildly with small changes in the data); and **large coefficients** after standardization (a big coefficient has more power to move predictions).

With p ≥ n there's no unique OLS solution at all. Correlated features let many combinations fit almost equally well (+100 on one and −98 on the other), so tiny changes in the data swing the coefficients wildly; large, opposite-signed coefficients are a classic overfitting signature. Wine evidence: the largest standardized OLS coefficient on the 90 polynomial features is about **137** (for a 3–9 quality score); after ridge it's about **0.23**.

<a id="penalty-term"></a>

### Penalty term

_Concept_

The extra piece added to the loss function. OLS minimizes MSE; ridge minimizes **MSE + α Σβⱼ²**; lasso minimizes **MSE + α Σ|βⱼ|**. The model now has to balance fitting the data against keeping coefficients small. The intercept is not penalized.

Shrinkage in one line of algebra: minimizing (y − b)² gives b = y, but minimizing (y − b)² + αb² gives **b = y / (1 + α)**, which shrinks towards 0 as α grows. "argmin" in the formulas means "the values that make this as small as possible".

<a id="ridge-regression-l2"></a>

### Ridge regression (L2)

_scikit-learn, Concept_

Penalizes the **sum of squared coefficients** (the L2 norm). It shrinks all coefficients toward zero but never exactly to zero. Preferred when you have many relevant features that are correlated with each other. Sometimes called **weight decay** in neural networks.

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10)
ridge.fit(Z_train, y_train)
ridge.score(Z_train, y_train), ridge.score(Z_test, y_test)
```

Ridge handles correlated features well: its penalty prefers **sharing** credit between them rather than giving huge opposite-signed coefficients. As α → ∞ every slope goes to 0 and the prediction becomes the mean. The intercept is never penalized.

<a id="lasso-regression-l1"></a>

### Lasso regression (L1)

_scikit-learn, Concept_

Penalizes the **sum of absolute coefficients** (the L1 norm, i.e. Manhattan distance). It shrinks coefficients and can push unimportant ones **exactly to zero**, so it also does feature selection. Preferred when you suspect many features are irrelevant or want a sparse model. "More brutal" than ridge. LASSO = **L**east **A**bsolute **S**hrinkage and **S**election **O**perator.

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01, max_iter=50_000).fit(Z_train, y_train)
(lasso.coef_ == 0).sum()        # how many features were dropped
```

Wine evidence: LassoCV keeps **17 of 90** features (73 coefficients exactly zero) with test R² ≈ 0.286, close to ridge's 0.295 while using a fraction of the features. With a group of correlated features, lasso tends to keep one arbitrarily and drop the rest.

See also: [Why lasso gives exact zeros](#why-lasso-gives-exact-zeros)

<a id="why-lasso-gives-exact-zeros"></a>

### Why lasso gives exact zeros

_Concept_

The geometry, for two coefficients β₁ and β₂. Regularization restricts the coefficients to a region whose size is set by α: a **diamond** for lasso (|β₁| + |β₂| ≤ budget) and a **circle** for ridge (β₁² + β₂² ≤ budget). The error contours expand outward from the unregularized solution until they touch the region. The diamond has sharp **corners on the axes**, and contours usually touch it at a corner, where one coefficient is exactly zero. The circle has no corners, so ridge's solution usually keeps both coefficients non-zero.

<a id="shrinkage-and-selection"></a>

### Shrinkage and selection

_Concept_

The two things Lasso's name promises. **Shrinkage:** a higher α pulls β's toward zero. **Selection:** a higher α zeros out small β's entirely. Ridge only shrinks.

<a id="elasticnet"></a>

### ElasticNet

_scikit-learn, Concept_

Combines both penalties: SSE + α[ρ Σ|βⱼ| + (1 − ρ) Σβⱼ²]. It can do feature selection like Lasso while handling correlated features like Ridge. In scikit-learn, `alpha` is the overall strength and `l1_ratio` (ρ) the mix: 0 is pure Ridge, 1 is pure Lasso.

```python
from sklearn.linear_model import ElasticNetCV

enet = ElasticNetCV(alphas=np.linspace(0.5, 1.0, 100), l1_ratio=0.5, cv=5)
enet.fit(Z_train, y_train)
enet.alpha_
```

> **Watch out:** The notebook notes that an `l1_ratio` below about 0.05 can cause numerical problems in scikit-learn.

Wine data: ElasticNetCV with `l1_ratio=0.5` picks α ≈ 0.013, test R² ≈ 0.286, keeping 22 of 90 features. scikit-learn advises using `Ridge` directly rather than `l1_ratio=0`.

> **Watch out:** Reusing names like `X_train` for a second dataset in the same notebook makes it easy to fit a model to the wrong data (here, unscaled data with a 0/1 target). Give each dataset its own variable names, and fit ElasticNet on the scaled wine `Z_train`.

<a id="alpha-regularization-strength"></a>

### alpha (regularization strength)

_Concept, scikit-learn_

The tuning parameter that sets how strong the penalty is. **α = 0**: plain OLS. **α → ∞**: every coefficient is forced to 0. **Raising α**: more bias, less variance. **Lowering α**: less bias, more variance. Many textbooks (including ISLR) call it **λ**.

> **Watch out:** This α has nothing to do with the significance level α from hypothesis testing, or the learning rate α in gradient descent.

> **Watch out:** Ridge and lasso alphas aren't on the same scale in scikit-learn. Ridge uses the plain sum of squared errors plus α‖β‖²; lasso uses (1/2n) × the sum plus α‖β‖₁. That's why sensible search ranges differ: on the wine data, ridge alphas of 1–100,000 but lasso alphas of 0.001–1.

<a id="ridge-and-lasso-trace-plot"></a>

### Ridge and Lasso trace plot

_Concept_

A plot of each coefficient's value against α. Ridge coefficients glide smoothly toward zero; Lasso coefficients hit zero one by one as α grows.

```python
alphas = np.logspace(-3, 2, 50)
coefs = [Lasso(alpha=a, max_iter=50_000).fit(Z_train, y_train).coef_ for a in alphas]
plt.plot(alphas, coefs)
plt.xscale("log")
```

<a id="standardize-before-regularizing"></a>

### Standardize before regularizing

_Concept, scikit-learn_

Penalties are based on coefficient size, and coefficient size depends on a feature's units. Without scaling, features are penalized for their scale rather than their importance. Always standardize first (fit on training data). Scaled data is often named `Z`.

```python
sc = StandardScaler()
Z_train = sc.fit_transform(X_train)
Z_test = sc.transform(X_test)
```

> **Watch out:** Standardizing and regularizing are different things; you need both. Standardizing isn't about the intercept (scikit-learn's ridge and lasso **never** penalize it); you standardize so every coefficient is penalized fairly, regardless of its feature's units.

<a id="ridgecv-and-lassocv"></a>

### RidgeCV and LassoCV

_scikit-learn_

Versions of Ridge and Lasso that try a list of alphas with built-in cross-validation and keep the best one (`.alpha_`). Give them a log-spaced grid.

```python
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = RidgeCV(alphas=np.logspace(0, 5, 100), scoring="r2", cv=5)
ridge_cv.fit(Z_train, y_train)            # fit the CV object itself
ridge_cv.alpha_

lasso_cv = LassoCV(alphas=np.logspace(-3, 0, 100), cv=5, max_iter=50_000)
lasso_cv.fit(Z_train, y_train)
lasso_cv.alpha_
```

> **Watch out:** An easy slip is `ridge_cv = ridge_model.fit(Z_train, y_train)`: that refits an earlier `Ridge(alpha=10)` and throws the cross-validation away. Call `.fit()` on the `RidgeCV` object itself, then check `ridge_cv.alpha_`.

On the wine data, RidgeCV picks α ≈ **104.8** (test R² 0.295, against 0.244 for an untuned α = 10). LassoCV picks α ≈ 0.0076. Compare `ridge_cv.coef_` (all small, none zero) with `lasso_cv.coef_` (mostly exact zeros): that's the key visual moment.

<a id="np-logspace"></a>

### np.logspace()

_NumPy_

Creates numbers evenly spaced on a **log** scale: `np.logspace(0, 5, 100)` gives 100 values from 10⁰ = 1 to 10⁵ = 100,000. Ideal for regularization grids, where trying 0.001, 0.01, 0.1, 1, 10 matters more than 1, 2, 3.

```python
np.logspace(-2, 0, 5)     # [0.01, 0.0316, 0.1, 0.316, 1.0]
```

See also: [np.arange() and np.linspace() (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#np-arange-and-np-linspace)

<a id="c-inverse-regularization-strength"></a>

### C (inverse regularization strength)

_scikit-learn_

Logistic regression and SVMs specify regularization with **C = 1/α**, so the direction is reversed: **small C = strong regularization** (more bias), **large C = weak regularization** (more variance). `C=1e9` effectively turns regularization off.

```python
LogisticRegression(C=1e9)          # practically unregularized: overfits 200 features
LogisticRegression(C=0.1)          # strong penalty
```

On the synthetic 200-feature data, a nearly unregularized logistic regression (`C=1e9`) scores training 0.931 but test 0.636: clear overfitting that a cross-validated C fixes.

See also: [Regularization is on by default](#regularization-is-on-by-default), [C (SVM)](#c-svm)

<a id="penalty-and-l1-ratio-in-logisticregression"></a>

### penalty and l1_ratio in LogisticRegression

_scikit-learn_

Which kind of penalty logistic regression uses. The course writes `penalty="l1"` or `"l2"` (L1 needs `solver="liblinear"` or `"saga"`). From scikit-learn 1.8, `penalty` is deprecated in favour of `l1_ratio`: 0 = L2, 1 = L1, values between = ElasticNet.

```python
LogisticRegression(penalty="l1", solver="liblinear")   # course syntax
LogisticRegression(l1_ratio=1, solver="liblinear")     # scikit-learn 1.8+
```

<a id="logisticregressioncv"></a>

### LogisticRegressionCV

_scikit-learn_

Logistic regression with built-in cross-validation over a list (or number) of C values. The chosen value is in `.C_`.

```python
from sklearn.linear_model import LogisticRegressionCV

logreg_cv = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="liblinear")
logreg_cv.fit(X_train_sc, y_train)
logreg_cv.C_
```

<a id="regularizing-categorical-variables"></a>

### Regularizing categorical variables

_Concept_

Scaling a 0/1 dummy and then shrinking its coefficient is hard to interpret, and it's a problem for every algorithm that needs scaling. Options (none perfect): separate penalties per variable or grouped Lasso (not in scikit-learn), choosing a scale by hand, dropping the variables, or "just do it anyway ¯\\_(ツ)_/¯", which is usually fine in practice.

<a id="fit-regularized-in-statsmodels"></a>

### fit_regularized() in statsmodels

_statsmodels_

statsmodels' way to regularize an OLS model. `L1_wt=0` gives ridge, `L1_wt=1` gives lasso. As alpha increases the SSE rises and the coefficients move toward 0, as in the review lab.

```python
model = sm.OLS(y, X)
for a in [0, .001, .01, .25, .5, 1, 10]:
    res = model.fit_regularized(alpha=a, L1_wt=0)
    print(a, np.mean(abs(res.params)))
```

Review lab results: α = 0 → SSE 0.276; α = 0.25 → 0.460; α = 1 → 0.642; α = 10 → 0.963, with the mean |coefficient| falling from 0.099 to 0.010. Higher training error isn't automatically worse: regularization's benefit shows up on **new** data, which the lab doesn't test.

> **Watch out:** Unlike scikit-learn, statsmodels' `fit_regularized` also penalizes the constant, and unscaled polynomial features (X⁵ reaches 3.2 million in the review lab) are penalized very unevenly, so scale first.

<a id="make-classification"></a>

### make_classification()

_scikit-learn_

Generates a synthetic classification dataset with a chosen number of informative and noise features. The regularization notebook uses 200 features, only 15 informative, to show an unregularized logistic regression overfitting.

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=200,
                           n_informative=15, random_state=123)
```

> **Watch out:** `test_size=0.7` trains on only 30% of the data (1,949 rows for 90 features), which makes overfitting more likely. That's useful for demonstrating regularization, but in your own work hold out a smaller share.

---

<a id="s21-hyperparameter-tuning-grid-search-and-pipelines"></a>

## 21. Hyperparameter tuning: grid search and pipelines

_Automating the search for the best model settings, and wrapping preprocessing and modeling into one object so tuning and evaluation are leak-free. Examples use the abalone data (rings predicted with kNN), South African heart disease data, and NBA game data._

**In your capstone:** Wrap every preprocessing step and model in a pipeline and tune it with GridSearchCV on the training data: the backbone of a leak-free capstone.

<a id="hyperparameter"></a>

### Hyperparameter

_Concept_

A model configuration setting **chosen before training** rather than learned from the data: k in kNN, α in Ridge, C in logistic regression and SVMs, `max_depth` in trees. Also called a **tuning parameter**. Choosing them is "challenging and time-consuming", which is why scikit-learn has tools for it.

<a id="hyperparameter-vs-parameter"></a>

### Hyperparameter vs parameter

_Concept_

A classic interview question. **Statistical parameters** are quantities the model learns from data, such as β₀ and β₁ in linear regression (scikit-learn's `coef_`, `intercept_`). **Hyperparameters** can't be learned by the model but affect its fit, such as k or α, so you set them and compare.

Rule of thumb: if it goes in the **brackets** when you create the model (`Ridge(alpha=10)`), it's a hyperparameter; if it appears with a **trailing underscore** after fitting (`coef_`), it's a learned parameter. `test_size` is neither: it's an evaluation choice.

<a id="hyperparameter-tuning"></a>

### Hyperparameter tuning

_Concept_

Trying different hyperparameter values and keeping the ones with the best **cross-validated** score. The course started with "sloppy loops" over k or max_depth; grid search automates it.

<a id="grid-search"></a>

### Grid search

_Concept_

A systematic approach to tuning that **exhaustively** evaluates every combination of the parameter values you list in a grid, scoring each with cross-validation, to find the optimal set.

<a id="gridsearchcv"></a>

### GridSearchCV

_scikit-learn_

scikit-learn's grid search. Pass an estimator (model or pipeline) and a `param_grid` dictionary; `.fit()` cross-validates every combination and then **refits the best one on all the training data** (`refit=True` by default), so the fitted object can predict and score directly. A grid of 4 values of k × 2 weightings with 5-fold CV fits 40 models, plus 1 refit.

```python
from sklearn.model_selection import GridSearchCV

knn = KNeighborsRegressor()
pgrid = {"n_neighbors": np.arange(1, 202, 5)}
gs = GridSearchCV(estimator=knn, param_grid=pgrid, cv=5)
gs.fit(X_train_sc, y_train)          # training data only: it does its own CV
gs.best_params_
gs.score(X_test_sc, y_test)          # check the test set once, at the end
```

> **Watch out:** Never tune on the test set. The workflow from now on: split → build pipeline → `GridSearchCV` on training data → inspect `cv_results_` and `best_params_` → evaluate `best_estimator_` on the test data once.

See also: [Reading grid search results](#reading-grid-search-results)

<a id="param-grid"></a>

### param_grid

_scikit-learn, Python_

A dictionary mapping hyperparameter names to lists of values to try. The number of models fitted is the product of the list lengths, times the number of folds.

```python
pgrid = {
    "lr__penalty": ["l1", "l2"],
    "lr__C": np.logspace(-2, 0, 100),
}                                  # 2 × 100 = 200 combinations × 5 folds = 1,000 fits
```

Choose ranges sensibly: log-spaced values for strengths (`np.logspace(-3, 3, 7)` gives 0.001 … 1,000), start coarse, then refine around the best value.

<a id="best-params-best-score-and-best-estimator"></a>

### best*params*, best*score* and best*estimator*

_scikit-learn_

Attributes of a fitted grid search. `best_params_`: the winning combination. `best_score_`: its mean cross-validated score (on the training data). `best_estimator_`: the winning model, already refitted on all of the training data.

```python
gs.best_params_          # {'n_neighbors': 26}
gs.best_score_           # mean CV score of that setting
gs.best_estimator_       # KNeighborsRegressor(n_neighbors=26)
```

<a id="cv-results"></a>

### cv*results*

_scikit-learn, pandas_

A dictionary with the full results of every combination: parameter values (`param_...`), `mean_test_score`, `std_test_score`, `rank_test_score`, fit times. Turn it into a DataFrame to plot how the score changes with a hyperparameter.

```python
df_gs = pd.DataFrame(gs.cv_results_)
plt.plot(df_gs.param_n_neighbors, df_gs.mean_test_score)
```

<a id="reading-grid-search-results"></a>

### Reading grid search results

_Concept_

Three habits from the course demonstrations:

- **Check the edges.** If the winner sits at the boundary of your grid (the best k was 35, the largest value tried), the true best may lie beyond it: extend the grid (45, 55, …) and search again.
- **Treat small differences as noise.** On the heart-disease data every C from 1 to 100 tied on CV accuracy (0.752); when scores tie, prefer the simplest, most regularized option.
- **Compare with the baseline.** The tuned kNN scored 0.672 on test against a null accuracy of 0.655, barely better; the tuned logistic regression scored 0.741.

Searching a huge grid can also find settings that are lucky on the CV folds (overfitting the CV score), which is why the final test-set check matters.

<a id="n-jobs-and-verbose"></a>

### n_jobs and verbose

_scikit-learn_

`n_jobs` sets how many CPU cores to use in parallel (`-1` = all of them), which speeds up big searches. `verbose=1` prints progress, e.g. "Fitting 5 folds for each of 48 candidates, totalling 240 fits".

<a id="the-cost-of-grid-searching"></a>

### The cost of grid searching

_Concept_

Grid search fits **every** combination. The notebook's cautionary grid of 150 × 3 × 4 × 151 × 2 × 2 = 1,087,200 combinations with 5-fold CV means 5,436,000 model fits. The search space grows exponentially with each added parameter, so understand what each hyperparameter does, choose sensible ranges, and start small. "Friends don't let friends use Grid Search" (without thinking).

Quick arithmetic: 4 hyperparameters with 10 values each is 10,000 combinations; with 5-fold CV that's 50,000 fits.

<a id="randomizedsearchcv"></a>

### RandomizedSearchCV

_scikit-learn_

Tries a fixed number (`n_iter`) of **randomly chosen** combinations instead of all of them, so it covers large or high-dimensional parameter spaces far more cheaply. Also called random or stochastic search. Can sample from distributions as well as lists.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

rs = RandomizedSearchCV(KNeighborsRegressor(),
                        {"n_neighbors": randint(1, 200), "p": [1, 2]},
                        n_iter=25, cv=5, random_state=42)
rs.fit(X_train_sc, y_train)
```

<a id="stratifiedkfold"></a>

### StratifiedKFold

_scikit-learn_

Like `KFold`, but each fold keeps the same class proportions as the whole dataset. scikit-learn uses it automatically when you pass `cv=5` to a classifier; create it yourself to add shuffling.

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
GridSearchCV(svc, pgrid, cv=cv)
```

<a id="pipelines-in-tuning"></a>

### Pipelines (in tuning)

_scikit-learn_

A pipeline chains transformers and a final model into one estimator, giving reproducible workflows, consistent preprocessing every time new data arrives, faster tuning and protection from data leakage. The same steps are applied to training folds, validation folds and test data automatically.

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("sc", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=27)),
])
pipe.fit(X_train, y_train)          # unscaled data in: the pipeline scales it
pipe.score(X_test, y_test)
```

Rules: every step **except the last** must be a transformer (it has `fit`/`transform`, like `StandardScaler`); the last is usually a model. `pipe.fit(X_train, y_train)` fits each step in turn; `pipe.predict(X_test)` applies the already-fitted transforms, so you can't accidentally refit the scaler on test data. With a pipeline, grid search can tune preprocessing too, e.g. `"poly__degree": [1, 2, 3]` alongside `"ridge__alpha"`.

See also: [Pipeline with ColumnTransformer](#pipeline-with-columntransformer)

See also: [Pipeline (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#pipeline), [ColumnTransformer (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#columntransformer)

<a id="pipeline-with-columntransformer"></a>

### Pipeline with ColumnTransformer

_scikit-learn_

Different preprocessing for different columns inside one leak-free object. The course demonstration on the South African heart disease data (462 men; target `chd`, 34.6% positive; one text column, `famhist`) scales the numeric columns, one-hot encodes the text column, and tunes kNN:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

heart = pd.read_csv("../data/SAheart.data", index_col="row.names")
X, y = heart.drop(columns="chd"), heart["chd"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)

prep = ColumnTransformer([
    ("num", StandardScaler(), X.columns.drop("famhist")),
    ("cat", OneHotEncoder(drop="first"), ["famhist"]),
])
pipe = Pipeline([("prep", prep), ("knn", KNeighborsClassifier())])
params = {"knn__n_neighbors": [3, 5, 9, 15, 25, 35],
          "knn__weights": ["uniform", "distance"]}     # 12 combos x 5 folds = 60 fits
gs = GridSearchCV(pipe, params, cv=5).fit(X_tr, y_tr)
gs.best_params_, gs.best_score_, gs.score(X_te, y_te)
```

During cross-validation the `ColumnTransformer` is refitted on each training fold; `gs.best_estimator_.named_steps["prep"]` is the version fitted on the full training set.

See also: [ColumnTransformer (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#columntransformer)

<a id="make-pipeline"></a>

### make_pipeline()

_scikit-learn_

A shortcut that builds a pipeline without naming the steps; names are the lowercased class names (`standardscaler`, `logisticregression`). The `imblearn` version is the same but also accepts samplers.

```python
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.named_steps        # {'standardscaler': ..., 'logisticregression': ...}
```

<a id="step-parameter-naming"></a>

### step\_\_parameter naming

_scikit-learn_

To tune a hyperparameter inside a pipeline, prefix it with the **step name and two underscores**. The same works for models nested inside models, such as the base tree of a bagging classifier.

```python
{"knn__n_neighbors": [3, 5, 10], "ss__with_mean": [True, False]}
{"randomoversampler__sampling_strategy": [.25, .5, 1]}     # make_pipeline names
{"estimator__max_depth": [1, 2, 3, None]}                  # BaggingClassifier's tree
```

> **Watch out:** One underscore (`knn_n_neighbors`) raises an "Invalid parameter" error. List valid names with `pipe.get_params().keys()`.

<a id="pipeline-gridsearchcv"></a>

### Pipeline + GridSearchCV

_scikit-learn_

Put the pipeline inside the grid search. Each fold then fits the scaler on that fold's training part only, so scaling never sees validation data, and you can tune preprocessing and model hyperparameters together.

```python
pipe = Pipeline([("sc", StandardScaler()),
                 ("lr", LogisticRegression(solver="liblinear"))])
pgrid = {"lr__penalty": ["l1", "l2"], "lr__C": np.logspace(-2, 0, 100)}
gs = GridSearchCV(pipe, param_grid=pgrid, n_jobs=8)
gs.fit(X_train, y_train)
gs.best_params_
```

<a id="pipeline-memory-and-debugging"></a>

### Pipeline memory and debugging

_scikit-learn_

Practical tips. **Memory:** `Pipeline(steps, memory="cache_dir")` caches **fitted transformers** on disk, so grid search doesn't refit an expensive step (a slow feature extractor) for every combination; it saves time rather than avoiding data copies. **Errors:** a failure in an early step breaks every later step. **Debugging:** build and test each step alone first, then inspect intermediate output.

```python
Pipeline(steps, memory="cache_dir")
pipe[:-1].transform(X_train)          # everything except the model
pipe.named_steps["sc"]                 # access one step
pipe.get_params().keys()               # valid step__parameter names
```

Common errors: a typo in a `step__param` name ("Invalid parameter…"), a `ColumnTransformer` column list that doesn't match the data, and text columns reaching a scaler.

<a id="choosing-a-scoring-metric-for-tuning"></a>

### Choosing a scoring metric for tuning

_Concept_

`GridSearchCV` picks whatever scores best on its `scoring` metric (default `.score()`: R² or accuracy). Always set a metric aligned with your problem, e.g. `scoring="balanced_accuracy"` for imbalanced classes or `"recall"` when misses are costly.

For imbalanced data, `scoring="f1"`, `"recall"`, `"roc_auc"` or `"balanced_accuracy"` match the classification metrics lessons.

<a id="season-based-train-test-split"></a>

### Season-based train/test split

_pandas_

Not every split should be random. In the basketball practice notebook, the test set is the most recent season and the training set is everything before it, mimicking how the model would be used.

```python
train = bball[bball["Season"] != 2016]
test = bball[bball["Season"] == 2016]
```

> **Watch out:** The basketball data has a **leakage trap**: `total_score`, `winner` and `loser` are game outcomes, so drop them from X. A sensible target is "host team won" (59.4%: home advantage).

---

<a id="s22-model-explainability"></a>

## 22. Model explainability

_Understanding how a model makes its decisions, why that matters (trust, fairness, regulation), which models are white or black boxes, and four tools for explaining them: tree diagrams, feature importance, partial dependence plots and SHAP. Examples use the iris data._

**In your capstone:** Stakeholders will ask why. Plan global and local explanations (permutation importance, partial dependence, SHAP) as part of your results.

<a id="model-explainability"></a>

### Model explainability

_Concept_

The ability to understand **how a model makes its decisions**: why it made this prediction, which factors mattered most, whether you can trust the output, and what would happen if the inputs changed slightly. If an AI model gave you a surprisingly low credit score, these are the questions you'd ask.

The questions people ask when a model decides about them: what data was used, was it accurate, which factors hurt my score, what could I change, was I treated fairly, can I appeal?

See also: [Global vs local explanations](#global-vs-local-explanations), [Model-specific vs model-agnostic methods](#model-specific-vs-model-agnostic-methods)

<a id="transparency"></a>

### Transparency

_Concept_

One of three components of explainability: how easy it is to understand the model's decision-making process and the relationships between inputs and outputs.

Transparency is about the model's **mechanism** (can you see inside?); interpretability is about **communicating** it to a person; feature importance **quantifies** each input's influence. A linear regression scores well on all three; a deep neural network poorly on the first two, though importance can be estimated afterwards.

<a id="interpretability"></a>

### Interpretability

_Concept_

Whether the model's decision can be explained in human-friendly terms, e.g. "the customer was denied a loan because their income was below the threshold".

<a id="why-explainability-matters"></a>

### Why explainability matters

_Concept_

Five reasons from the slides: **trust** (people use models they understand); **fairness** (explainable models are easier to audit for bias, e.g. a postcode feature acting as a proxy for ethnicity or income); **regulation** (healthcare and finance decisions must often be explainable, and financial regulators in many countries require reasons for credit decisions); **model improvement** (explanations often reveal leakage: if an ID or date column dominates the importance, something's wrong); and **ethical risk reduction** (spotting unsafe, biased or unexpected behaviour before it affects people).

How much explanation you need depends on the stakes, not the technique. A medical scan or a loan decision: high stakes, regulated, explainability essential (a common compromise in medical imaging is a model that highlights the regions it relied on, with a human deciding). Product recommendations: low stakes per decision, so performance usually matters more. Image recognition is low-stakes for tagging holiday photos but not for facial recognition in policing.

> **Watch out:** GDPR is often described as giving a "right to an explanation", but that's legally debated. GDPR gives people rights around solely automated decisions with significant effects (Article 22) and to "meaningful information about the logic involved" (Articles 13–15). Safer wording: **GDPR requires transparency about automated decisions.**

See also: [Proxy variable](#proxy-variable)

<a id="proxy-variable"></a>

### Proxy variable

_Concept_

A feature that stands in for something else, often something you shouldn't or can't use directly. Postcode can proxy for ethnicity or income; the credit start-up that scored applicants on **handwriting** ("we don't know why, it just is") may have been picking up age, disability, education or first language. Proxies can make a model discriminatory without any protected attribute in the data, can't be justified to the person affected, and may not hold in future data. Correlation alone isn't a justification for high-stakes decisions.

<a id="white-box-model"></a>

### White-box model

_Concept_

A model whose decision process is easy to see: **linear regression** (a simple equation), **logistic regression** (a weighted sum giving log-odds), **decision trees** (an intuitive flowchart). The King's College COVID symptom model is a simple, explainable but still powerful example.

The King's College COVID symptom model (Menni et al., _Nature Medicine_, 2020) is a **logistic regression**: one equation combining a handful of symptoms (loss of smell and taste was the strongest) with age and sex. The lesson: simple, interpretable models can be good enough, and their transparency is a huge advantage in high-stakes settings, so always try a simple model first.

> **Watch out:** "White box" depends on use, not just model type: a linear model with 90 polynomial features, or a very deep tree, is hard to explain. And logistic regression outputs the **log-odds** of the outcome; its exponentiated coefficients are odds ratios.

<a id="black-box-model"></a>

### Black-box model

_Concept_

A model whose decisions are hard to explain: **deep neural networks** (including most AI models and LLMs), **random forests** (hundreds of trees), **SVMs** in high dimensions, boosted ensembles. The credit-scoring start-up that used handwriting ("we don't know why, it just is") is the cautionary example.

Nuances: random forests still provide `feature_importances_`; what you lose is a single readable decision path. An SVM with a linear kernel has interpretable coefficients; non-linear kernels (RBF) don't. Post-hoc methods (PDPs, SHAP, permutation importance) exist precisely to explain black boxes after training.

<a id="global-vs-local-explanations"></a>

### Global vs local explanations

_Concept_

A **global** explanation describes the model as a whole: "which factors matter most overall?" A **local** explanation describes one prediction: "why was **my** loan refused?" Tree plots, feature importance and partial dependence plots are global; SHAP gives local explanations and, averaged across many rows, a global ranking too.

<a id="counterfactual-explanation"></a>

### Counterfactual explanation

_Concept_

An explanation of the form "if your input had been different in this way, the decision would have changed": "if your income were $5,000 higher, you'd have been approved." It answers the "what if the inputs changed slightly?" question people naturally ask, and is often the most useful explanation to give the person affected.

<a id="model-specific-vs-model-agnostic-methods"></a>

### Model-specific vs model-agnostic methods

_Concept_

**Model-specific** methods only work for one kind of model: tree diagrams and Gini feature importance (trees), coefficients (linear models). **Model-agnostic** methods treat the model as a black box and work with almost anything: partial dependence plots, permutation importance and SHAP.

| Method                  | Works with | Scope            | Answers                              |
| ----------------------- | ---------- | ---------------- | ------------------------------------ |
| Tree plot               | Trees      | Global           | What are the rules?                  |
| Gini importance         | Trees      | Global           | How much does each feature matter?   |
| Permutation importance  | Any model  | Global           | How much does score drop without it? |
| Partial dependence plot | Any model  | Global           | In which direction, with what shape? |
| SHAP                    | Any model  | Local and global | Why this prediction?                 |

<a id="visualizing-a-decision-tree-plot-tree"></a>

### Visualizing a decision tree (plot_tree)

_scikit-learn, matplotlib_

Draws a fitted tree. Each node shows its split rule, Gini impurity, number of samples and class counts; leaves show the prediction. The most important features sit near the top, and the thresholds tell you the cut-off values.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True)
```

The iris tree (`max_depth=3`) reads as plain rules: petal_length ≤ 2.45 → setosa; otherwise petal_length ≤ 4.85 → versicolor; otherwise petal_width ≤ 1.70 → versicolor, else virginica. "Closer to the top = more important" is a good rule of thumb but not a law: a feature used many times lower down can matter more overall.

See also: [export_text()](#export-text)

<a id="feature-importance"></a>

### Feature importance

_Concept, scikit-learn_

A quantitative score for how much each feature contributed to a model. For trees, it's the total amount a feature **reduces Gini impurity** (or MSE for regression) across all its splits, weighted by the number of samples reaching them and normalized to sum to 1: the **mean decrease in impurity (MDI)**. Ensembles average it over their trees. On the iris tree: petal_length 0.95, petal_width 0.05, sepals 0.

```python
importances = clf.feature_importances_
order = np.argsort(importances)[::-1]
plt.barh(np.array(feature_names)[order], importances[order])
plt.gca().invert_yaxis()
```

> **Watch out:** Importances show how much a feature was **used**, not the direction of its effect. They're computed on training data (so they can reward features that helped the tree overfit), they favour features with many possible split points, and correlated features can split credit arbitrarily. An importance of 0 for the sepal measurements doesn't mean sepals are useless, only that this tree didn't need them once petal length had done the job.

See also: [Gini impurity](#gini-impurity), [Permutation importance](#permutation-importance)

<a id="permutation-importance"></a>

### Permutation importance

_scikit-learn_

A model-agnostic alternative to Gini importance: shuffle one feature at a time on **test** data and measure how much the score drops. A big drop means the model relies on that feature. Often more reliable than impurity-based importance, because it's measured on unseen data and doesn't favour high-cardinality features.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
```

<a id="partial-dependence-plot-pdp"></a>

### Partial dependence plot (PDP)

_Concept, scikit-learn_

Shows **how changing one feature affects the model's predictions on average**. To build one: fix the feature at a value for every training row, predict, average, and repeat across the feature's range. **Linear** PDP: proportional effect. **Curved**: diminishing or accelerating effect; a plateau means the feature stops mattering past a point. **Step-like**: the model uses thresholds.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(clf, X_train, features=[0, 1, 2, 3],
                                        feature_names=feature_names,
                                        target=clf.classes_[0])
```

Intuition: "if everyone in the dataset had a temperature of 25 °C, what would the model predict on average?" Repeat for 10, 15, 20 °C… and join the dots. For classifiers the y-axis is usually the average predicted probability of a chosen class (`target=`, needed for multiclass). The PDP's shape partly reflects the model type: **trees always give step-shaped PDPs** (predictions only change at split thresholds) and linear models always give straight lines.

> **Watch out:** PDPs average over everything else, so they can hide interactions and create unrealistic feature combinations when features are correlated. They describe what the **model** learned, not causes: a "road accidents" effect on house prices may really be about busy city-centre roads.

<a id="individual-conditional-expectation-ice"></a>

### Individual conditional expectation (ICE)

_scikit-learn_

The per-row version of a PDP: one line per observation instead of the average, which reveals when the effect differs between observations. Request it with `kind="individual"` (or `"both"` for ICE lines plus the PDP average), as the notebook does.

If the ICE lines fan out or cross, the feature's effect depends on other features (an interaction the PDP's average hides).

<a id="shap-shapley-additive-explanations"></a>

### SHAP (SHapley Additive exPlanations)

_Concept, shap_

Explains individual predictions by showing how much **each feature pushed this prediction up or down** from the average prediction. **Additive** means: prediction = base value (the average prediction) + the sum of every feature's SHAP value. Averaging absolute SHAP values over many rows gives a global importance ranking. Based on Shapley values from game theory and named after Lloyd Shapley (spelled Shapley, not "Shapely"). Works with any model, with fast exact versions for trees.

```python
import shap   # pip install shap

explainer = shap.TreeExplainer(tree)       # fast, exact for tree models
sv = explainer(X_te)                       # shape: (rows, features, classes)
shap.plots.bar(sv[:, :, 2])                # average impact, class index 2 (virginica)
shap.plots.waterfall(sv[0, :, 2])          # explanation for one flower
```

Choose the class by its **position** in `tree.classes_`; indexing by class name doesn't work in current SHAP versions. On iris (virginica), the base value is about 0.33 and mean |SHAP| is about 0.33 for petal_length and 0.12 for petal_width.

> **Watch out:** Gini importance and SHAP can disagree (petal_width 0.05 of the Gini importance but about a quarter of the SHAP importance). Both are "right": Gini counts impurity reduction during training; SHAP measures each feature's effect on predictions. Neither is "the" importance.

See also: [KernelExplainer and force plot](#kernelexplainer-and-force-plot)

<a id="shapley-value"></a>

### Shapley value

_Concept_

From cooperative game theory: a player's fair share of the total payout, computed by averaging how much they add across every possible order of joining the team. SHAP treats each feature as a player and the prediction as the payout.

Analogy: a team wins a prize; how do you share it fairly when players help more in some combinations than others? Average each player's extra contribution over every possible order in which they could join. Lloyd Shapley shared the 2012 Nobel Memorial Prize in Economic Sciences.

<a id="kernelexplainer-and-force-plot"></a>

### KernelExplainer and force plot

_shap_

`shap.KernelExplainer` works with any model's prediction function (slow but general; `TreeExplainer` is much faster for tree models). A **force plot** shows features pushing the prediction above (red) or below (blue) the base value, the model's average output.

`KernelExplainer` is model-agnostic but slow (it approximates Shapley values by sampling); for tree models prefer `TreeExplainer`.

---

<a id="s23-framing-a-data-science-problem-and-roi"></a>

## 23. Framing a data science problem and ROI

_Before building a model, make sure it's the right model for the right problem, and that it's worth building. The five key considerations for framing a project (business problem, stakeholders, success metrics and hypotheses, feasibility and ROI, and ethical, legal and strategic implications), applied to the course's running case study: predictive maintenance for a wind farm. "Framing decides whether the model matters; modelling decides whether it works."_

**In your capstone:** Start your capstone here: a specific, measurable problem statement, agreed success metrics, a baseline and the expected value of the work.

<a id="problem-framing"></a>

### Problem framing

_Concept_

Turning a vague business wish ("use AI to fix our turbines") into a **specific, measurable, feasible** data science question with agreed success criteria. It's the first stage of the data science lifecycle and widely considered the most important and most neglected: a brilliant model that answers the wrong question is worthless. Framing is best done as early as possible, because changing direction is cheap at the start and expensive after months of modelling.

> "The formulation of a problem is often more essential than its solution, which may be merely a matter of mathematical or experimental skill." Albert Einstein and Leopold Infeld, _The Evolution of Physics_ (1938).

See also: [Five key considerations for framing](#five-key-considerations-for-framing)

<a id="data-science-lifecycle"></a>

### Data science lifecycle

_Concept_

Five stages: **framing the problem** (understand the business need, define success metrics); **preparing and exploring data**; **modelling** (select, train, evaluate with appropriate metrics); **communicating** (insights, deploying the model); and **deploying, monitoring and maintaining**. It's a loop, not a line: monitoring often reveals new problems, which sends you back to framing. A similar published framework is **CRISP-DM** (Cross-Industry Standard Process for Data Mining), whose first stage is "Business Understanding".

See also: [Modeling workflow (end to end)](#modeling-workflow-end-to-end)

<a id="why-data-science-projects-fail"></a>

### Why data science projects fail

_Concept_

A commonly cited industry estimate is that around **85%** of data science projects fail. Treat the number with caution: it traces to a 2017 Gartner analyst comment about big data projects and a 2019 VentureBeat article ("87% never make it into production"), not rigorous research. "Fail" can mean never deployed, deployed but unused, or used but delivering no measurable value. The usual causes are about framing, not modelling: no clear objectives or success metrics, misalignment with business goals, rushing into analysis, no specific problem to solve, and no understanding of the potential ROI.

<a id="five-key-considerations-for-framing"></a>

### Five key considerations for framing

_Concept_

The lesson's map, each with its own entry below:

1. **Define the business problem**: specific, measurable outcomes, not vague goals.
2. **Understand the stakeholders**: who uses the insights and how they benefit.
3. **Clarify success metrics and modelling hypotheses**: how success is measured, and clear, testable hypotheses.
4. **Assess feasibility**: solvable within data, time and resource constraints, and worth the cost (ROI).
5. **Consider ethical, legal and strategic implications**: biases, unintended consequences, compliance.

They aren't strict steps in order: talking to stakeholders often changes the problem statement.

<a id="predictive-maintenance"></a>

### Predictive maintenance

_Concept_

Using data (sensor readings, maintenance records) to predict when equipment will fail, so maintenance happens just before failure. It contrasts with **reactive** maintenance (fix it when it breaks: costly downtime) and **preventive/scheduled** maintenance (service on a fixed timetable, sometimes replacing parts that were fine). Wind turbines are a classic case: expensive, remote, hard to reach and full of sensors (vibration, temperature, oil, power output). The course's running case study starts with the COO asking "can your team predict turbine failures before they happen?", a question still too vague to model: which failures, how far ahead, how accurate?

<a id="problem-statement"></a>

### Problem statement

_Concept_

A specific, measurable description of the business problem and the goal. Do: identify the core challenge, make it measurable, align it with broader business objectives, consult stakeholders and experts, and consider the impact on cost, efficiency and operations. Don't: frame it too broadly, assume without data, or set objectives disconnected from company goals. The wind farm template:

> "Wind turbines are experiencing unplanned breakdowns due to equipment failure, leading to **X** hours of downtime per year and **Y** in repair costs. We aim to reduce turbine downtime by **Z%** by leveraging predictive maintenance to anticipate mechanical failures before they occur."

X, Y and Z are filled in with real figures from operations and finance records (in the ROI notebook: $182,000 a year lost to unplanned downtime and a 20% target reduction). A good question has a clear target, a number, a timeframe and a link to data: "What data indicators best predict turbine failures, to cut unplanned repair costs by 15% within a year?"

See also: [SMART goals](#smart-goals)

<a id="smart-goals"></a>

### SMART goals

_Concept_

A checklist for objectives: **S**pecific, **M**easurable, **A**chievable, **R**elevant, **T**ime-bound. "Reduce downtime" fails it; "reduce unplanned turbine downtime by 20% within 12 months" passes. "Improve customer satisfaction" fails; "increase satisfaction scores on a 5-point scale by 10% next quarter compared with this quarter" passes.

<a id="stakeholder"></a>

### Stakeholder

_Concept_

Anyone with an interest in, or affected by, a project's outcomes: the people who **use** the insights, those who **make decisions** or implement changes, and those **affected** by the decisions, including people who never see the model (job applicants screened by an algorithm, customers refused a loan). Stakeholders differ in information needs, technical understanding, decision-making authority and concerns (field engineers may worry a model will judge their performance or replace their expertise).

Do: identify all stakeholders (internal and external), understand their needs, communicate clearly, and **engage early and often**. Don't: assume their needs, talk only to technical teams, or wait until the project is finished to involve them. Many technically excellent projects fail simply because nobody uses them.

In the wind farm case, each conversation produced a concrete requirement: the COO (budget, timeline, KPIs), the field engineers (failure history, a tablet-friendly tool, their current accuracy as a baseline), operations (**at least 72 hours' warning**, which defines the prediction target), IT (integration and computing) and fellow analysts (data sources, including a colleague's bespoke weather dataset).

See also: [Influence/interest grid](#influence-interest-grid)

<a id="influence-interest-grid"></a>

### Influence/interest grid

_Concept_

A stakeholder map (also called a power/interest grid) that sorts people by influence and interest. **High influence, high interest: engage closely** (the COO, operations, field engineers). **High influence, low interest: keep satisfied.** **Low influence, high interest: keep informed** (IT, other analysts). **Low on both: monitor** (other departments). Positions are judgement calls and change over time (IT's influence rises sharply at deployment), and stakeholders in the top-right quadrant, especially the sponsor, can help you manage the difficult ones.

<a id="project-sponsor-and-kpi"></a>

### Project sponsor and KPI

_Concept_

The **project sponsor** is the senior person who asks for and pays for the project (the COO, Chief Operating Officer, in the case study); a supportive sponsor can unblock resources, resolve conflicts and persuade sceptical teams. A **KPI** (key performance indicator) is a metric the business already tracks, such as downtime hours or maintenance cost; tying model targets to KPIs makes success visible to decision-makers.

<a id="technical-debt"></a>

### Technical debt

_Concept_

Future work created by shortcuts taken now: undocumented code only one person understands, hard-coded settings, quick fixes that never get cleaned up. It's a feasibility risk (what happens if the one expert leaves?) and a reason to share knowledge across the team.

<a id="success-metrics-and-hypotheses"></a>

### Success metrics and hypotheses

_Concept_

Define, before modelling, how success will be measured: **technical metrics** (model performance), **business metrics** (the impact the business cares about) and any **constraints** from stakeholders, all agreed by the people involved. A **hypothesis** is a specific, testable claim that could turn out to be false. The wind farm framing:

- **Technical:** true positive rate > 80%, false positive rate < 5%, at least 72 hours' warning.
- **Business:** 20% less unplanned downtime, 15% lower maintenance costs.
- **Primary hypothesis:** sensor patterns (vibration, temperature, pressure) plus maintenance records can predict failures 72+ hours ahead within those error rates. **Supporting hypotheses** (testable in EDA): vibration anomalies precede mechanical failures; temperature spikes correlate with electrical failures.

> **Watch out:** Don't set success as "70% accuracy" for a rare event. Failures are rare, so a model that never predicts one would score well over 90% accuracy and be useless; use recall and precision. And make targets unambiguous: "25% better than the engineers' current assessments" could mean they now catch 64% (relative) or 55% (absolute points).

See also: [Base rate and false alarms](#base-rate-and-false-alarms), [Imbalanced classes](#imbalanced-classes)

<a id="base-rate-and-false-alarms"></a>

### Base rate and false alarms

_Concept_

Targets that sound strict can still bury people in false alarms when the event is rare. Suppose 1% of turbine-days end in failure. Per 1,000 turbine-days: 10 failures, and an 80% TPR catches **8**; 990 non-failures, and a 5% FPR flags about **50**. Only about 8 of every 58 alarms are real: **precision ≈ 0.14**. That's why the wind farm framing should also set a precision target, and why the base rate of the positive class matters as much as the error rates.

```python
base_rate, tpr, fpr, n = 0.01, 0.80, 0.05, 1_000
tp = tpr * base_rate * n                     # 8 true alarms
fp = fpr * (1 - base_rate) * n               # 49.5 false alarms
tp / (tp + fp)                               # precision ≈ 0.14
```

See also: [Precision](#precision), [False positive rate (FPR)](#false-positive-rate-fpr), [Base rate fallacy](#base-rate-fallacy)

<a id="framing-the-problem-type-early"></a>

### Framing the problem type early

_Concept_

Deciding early whether the task is **classification** (categories), **regression** (continuous values) or **clustering** (grouping without labels) clarifies the goal, decides which algorithms are possible, shapes data preparation (imbalanced classes for classification; outliers and linearity for regression; scaling for clustering), and sets stakeholder expectations. "We'll flag turbines likely to fail in the next 72 hours" is a very different promise from "we'll tell you how many days each turbine has left", and neither tells engineers **what** is wrong unless that's agreed up front.

The wind farm could be framed as **classification** (failure within 72 hours: yes/no), usually the simpler, more actionable start, or as predicting **time to next failure**. The latter is often described as time-series forecasting; more precisely it's a regression (or survival-analysis) problem known as **remaining useful life (RUL)** prediction, with time-series methods often used to build its features.

See also: [Regression vs classification tasks](#regression-vs-classification-tasks)

<a id="feasibility-assessment"></a>

### Feasibility assessment

_Concept_

Checking that data science can deliver a viable solution within the data, time, budget and people available. Do: check the availability, quality and relevance of data; assess technical and human resources against the expected ROI; consider time and budget; identify risks and mitigations. Don't: proceed without verifying the data exists, over-promise, ignore bottlenecks (privacy rules, technical debt), or pursue unscalable solutions.

The most common feasibility failure is the **label**: for the wind farm, every past turbine-day needs a "failed within 72 hours" label, which depends on accurate, timestamped maintenance records. The COO's condition (promising signs within 2–3 months) suggests delivering a simple working baseline quickly, such as logistic regression on a few key sensors, rather than a perfect model late.

Typical risks and mitigations: poor sensor data (validation, cleaning, calibration); inaccurate predictions (rigorous testing on a hold-out set, which for time-ordered data should be the most recent period); integration problems (involve IT early); and lack of adoption (involve users in design and show clear benefits). Two of those four risks are about people, not technology.

<a id="return-on-investment-roi"></a>

### Return on investment (ROI)

_Concept, Python_

How much a project returns relative to what it costs: **ROI (%) = (benefits − total investment) / total investment × 100**. A positive ROI means the project pays for itself. In the wind farm example, first-year investment is $76,000 and benefits are $99,175, giving an ROI of about 30.5%.

```python
total_investment = 21_000 + 45_000 + 10_000          # IT + development + maintenance
total_benefits = 418_500 * 0.15 + 182_000 * 0.20      # maintenance + downtime savings
roi = (total_benefits - total_investment) / total_investment * 100
print(f"The ROI for the first year is {roi:.1f}%")
```

ROI is the language of business stakeholders: it aligns the project with business goals, justifies its money and people, defines what success looks like, and lets decision-makers compare it with other options. Simple check: spend $100k, gain $130k in benefits, and ROI = (130 − 100)/100 × 100 = 30%. **ROI of 0%** means you just got your money back; negative means a loss.

**Benefits** include revenue uplift (personalized marketing), cost savings (automation) and risk reduction (fraud detection). **Costs** include development (data acquisition, model building), deployment and infrastructure (cloud, APIs), and maintenance and monitoring.

> **Watch out:** Keep cost labels unambiguous. Here "maintenance costs" ($10,000) means maintaining the **model**, while "current operations" ($418,500) is **turbine** maintenance, and the $45,000 development cost is one-time, not per year.

See also: [Risk-adjusted ROI](#risk-adjusted-roi)

<a id="one-time-vs-recurring-costs"></a>

### One-time vs recurring costs

_Concept_

**One-time costs** are paid once (initial development labour). **Recurring costs** are paid every year (cloud infrastructure, APIs, maintenance). Separating them matters because a project that loses money in year 1 can be very profitable by year 3.

<a id="multi-year-roi"></a>

### Multi-year ROI

_Concept, Python_

Over n years: ROI = (Σ annual benefits − (one-time costs + Σ yearly costs)) / (one-time costs + Σ yearly costs) × 100.

```python
years = 3
one_time = 45_000
recurring = 21_000 + 10_000
investment = one_time + recurring * years
roi_3yr = (total_benefits * years - investment) / investment * 100
```

Here the 3-year ROI is **115.6%** ($297,525 of benefits against $138,000 of investment). ROI rises with more years because the one-time development cost is spread over more of them.

> **Watch out:** This adds dollars from different years without discounting, so it slightly overstates the value; see time value of money below.

<a id="risk-adjusted-roi"></a>

### Risk-adjusted ROI

_Concept, Python_

Not assuming every projected benefit will happen: multiply benefits by a realisation share, a probability, or test pessimistic scenarios. It's sobering on the wind farm numbers: a healthy-looking 30.5% first-year ROI becomes a **loss** if only three-quarters of the savings arrive.

```python
for share in [1.0, 0.75, 0.5]:
    roi_adj = (total_benefits * share - total_investment) / total_investment * 100
    print(f"{share:.0%} of benefits realised -> ROI {roi_adj:.1f}%")
# 100% -> 30.5%   75% -> -2.1%   50% -> -34.8%

print(f"Break-even needs {total_investment / total_benefits:.1%} of projected benefits")   # 76.6%
```

The useful discussion question: how confident are we in the 20% and 15% savings estimates?

<a id="payback-period-and-npv"></a>

### Payback period and NPV

_Concept, Python_

Two other common ways to judge an investment. The **payback period** is how long it takes for net savings to cover the up-front cost. **Time value of money:** a dollar in year 3 is worth less than a dollar today, so finance teams discount future cash flows to a **net present value (NPV)**; a positive NPV means the project adds value after accounting for timing.

```python
one_time, recurring, benefits = 45_000, 31_000, 99_175
net_yearly = benefits - recurring                         # 68,175 per year
payback_months = one_time / (net_yearly / 12)             # ≈ 7.9 months

rate = 0.08
npv = -one_time + sum(net_yearly / (1 + rate) ** t for t in range(1, 4))   # ≈ $130,700
```

<a id="baseline-costs-and-projected-savings"></a>

### Baseline costs and projected savings

_Concept_

To estimate benefits, start from what the business spends now (the **baseline**: $418,500 maintenance and $182,000 downtime losses a year) and apply the improvement the model is expected to deliver (15% and 20%). The quality of an ROI estimate depends on how credible those percentages are, so they're worth agreeing with stakeholders.

<a id="ethical-legal-and-strategic-implications"></a>

### Ethical, legal and strategic implications

_Concept_

The fifth consideration. **Legal:** data protection laws (GDPR in the EU/UK, CCPA in California), intellectual property, industry regulations. **Ethical:** bias and fairness, transparency and explainability, privacy and confidentiality, informed consent, accountability. **Strategic:** organizational change, investment in technology and skills, and whether to **build, buy or partner**.

The group exercise scenarios show the range: screening job applicants (bias against gender or race, justifying rejections, anti-discrimination law; a hiring model trained on biased history may favour men); predicting disease (the costs of false positives vs false negatives, health-data privacy, consent, clinician oversight); sharing customer data with a third party (consent, GDPR, anonymization, the partner's security, customer trust); and AI chatbots (telling customers they're talking to AI, escalation to humans, liability for wrong answers, job impacts).

Not every consideration applies to every project. For the wind farm: protect novel methods (IP), follow energy-sector rules, make the model explainable to engineers (feature importance shows which sensors drive predictions), and decline the maintenance team's offer of its whole database, staff salaries included.

See also: [Why explainability matters](#why-explainability-matters), [Data minimization and anonymization](#data-minimization-and-anonymization)

<a id="data-minimization-and-anonymization"></a>

### Data minimization and anonymization

_Concept_

**Data minimization** (a GDPR principle) means collecting and using only the data you need for the purpose: request the turbine maintenance history, not the staff records that come with it. **Anonymization** removes or masks identities (names, IDs) so individuals can't be identified; you seldom need real identities to build a model. Do: comply with privacy law, anonymize where possible, write guidelines against misuse, and take legal advice. Don't: use sensitive data without anonymization, hoard data, or ignore intellectual property rights.

<a id="build-buy-or-partner"></a>

### Build, buy or partner

_Concept_

The strategic choice for any data science capability: **build** it in-house (control and IP, but needs skills and time), **buy** an existing product (fast, but less tailored), or **partner** with a specialist provider. For predictive maintenance, a third-party provider may already do this well, so "should we build this at all?" is a fair framing question.

<a id="framing-checklist"></a>

### Framing checklist

_Concept_

A reusable template for any project or capstone. If you can answer all five, the project is well framed:

1. **Business problem:** what is it, and why does it matter?
2. **Stakeholders:** who are they, and what do they need and expect?
3. **Success metrics and hypotheses:** what are the metrics, the input and target variables, and the expected ROI?
4. **Feasibility:** can data science provide a viable solution? Is it technically and financially feasible, with the data and tools available?
5. **Ethical, legal and strategic:** any risks, and is it aligned with strategy?

A good practice exercise: rewrite "use AI to reduce customer complaints" as a specific, measurable problem statement.

---

<a id="s24-leveraging-genai-as-a-data-scientist"></a>

## 24. Leveraging GenAI as a data scientist

_Using generative AI tools well, not building them: how large language models work at a high level, which data science tasks suit them, coding and data parsing with chat tools and APIs, and above all how to check their output. Specific facts in this area (model names, sizes, context lengths, which tool is "best") date within months; the principles (be specific, never trust unchecked output, protect sensitive data) don't._

**In your capstone:** Use GenAI to speed up code and text processing, but test everything it produces, never paste confidential data, and state how you used it.

<a id="generative-ai-genai"></a>

### Generative AI (GenAI)

_Concept_

AI systems that **generate new content** (text, code, images, audio) rather than only classifying or predicting a number. **Large language models (LLMs)** are the text-generating kind, such as GPT, Gemini and Claude. For a data scientist, GenAI is a productivity tool whose output you verify, not a replacement for understanding: "great as a helper, but not as a substitute".

See also: [Large language model (LLM)](#large-language-model-llm)

<a id="large-language-model-llm"></a>

### Large language model (LLM)

_Concept_

A neural network trained on huge amounts of text to predict the **next token**. It links to models you already know: like a classifier choosing the most likely class, an LLM predicts a probability for every possible next token. The deck's five steps: (1) **input**, the prompt; (2) **tokenization**, breaking it into tokens; (3) **encoding**, turning each token into numbers (an embedding); (4) **processing**, analysing relationships between tokens; (5) **output**, generating new text one token at a time, each new token fed back in as input for the next.

**Parameters** are the weights a network learns, like regression coefficients but billions of them. GPT-3 has 175 billion and was trained on about 570 GB of filtered text (selected from roughly 45 TB of raw text, about 300 billion tokens).

> **Watch out:** Size figures for the newest models are usually unofficial. OpenAI hasn't disclosed GPT-4's size, data or compute ("~1.76 trillion parameters" is an unconfirmed outside estimate), and Google hasn't disclosed Gemini's sizes. Treat model-size claims with caution and check the source.

See also: [Token](#token), [Transformer and attention](#transformer-and-attention)

<a id="token"></a>

### Token

_Concept_

The unit an LLM reads and writes. Tokens are often **pieces of words** ("unbelievable" → "un", "believ", "able"), not always whole words; in English one token is roughly **three-quarters of a word** on average. Context windows, prices and rate limits are all measured in tokens.

<a id="embedding"></a>

### Embedding

_Concept_

The "encoding" step: each token becomes a vector of numbers that captures its meaning, learned from how it's used. Words used in similar contexts end up with similar vectors. This is the **distributional hypothesis**: "You shall know a word by the company it keeps" (linguist J. R. Firth, 1957). Embeddings also power similarity search: finding the nearest vectors is the nearest-neighbour idea from kNN.

See also: [kNN strengths and weaknesses](#knn-strengths-and-weaknesses)

<a id="transformer-and-attention"></a>

### Transformer and attention

_Concept_

The architecture introduced in Google's 2017 paper "Attention Is All You Need", which set the stage for generative AI. **Attention** lets the model weigh how relevant every other word is when interpreting each word, for example working out what "it" refers to in "the cake was in the oven; it was delicious". Transformers process whole sequences in **parallel** (unlike earlier recurrent networks), which made training on huge datasets practical. GPT = **G**enerative **P**re-trained **T**ransformer.

<a id="temperature-and-sampling"></a>

### Temperature and sampling

_Concept_

Always picking the single most likely next token tends to produce repetitive, dull text, so models **sample** from the probabilities instead; lower-probability choices sometimes give a better output. **Temperature** controls how adventurous the sampling is: low = predictable and consistent, high = creative but more error-prone. **Beam search**, an alternative, keeps several candidate sequences ("beams") and picks the best overall; it's common in translation, while chatbots more often sample.

For data work, use a **low temperature** when you want consistent, repeatable answers (classifying text, extracting fields). Because of sampling, the same input can get different answers on different runs, so check consistency.

<a id="how-llms-are-trained"></a>

### How LLMs are trained

_Concept_

Three stages. **Pre-training**: exposure to vast amounts of text, learning to predict the next word; usually called **self-supervised**, because the "labels" are just the next words. **Fine-tuning** (instruction tuning): a smaller, task-specific dataset of example prompts and good responses, teaching the model to follow instructions and hold conversations. **RLHF** (reinforcement learning from human feedback): human raters compare pairs of responses, a **reward model** learns their preferences, and the LLM is then trained to produce responses the reward model scores highly. Newer approaches (e.g. DPO, AI-feedback methods) are also used. Without tuning, a pre-trained model just continues text; fine-tuning makes it conversational; human feedback aligns its responses with broad human values.

> **Watch out:** In RLHF, feedback comes mainly from trained raters and a learned reward model, not from live conversations with users.

See also: [Reinforcement learning](#reinforcement-learning)

<a id="context-window"></a>

### Context window

_Concept_

Everything the model can "see" at once, measured in tokens: your prompt, any pasted documents or data, and the conversation so far. When a conversation exceeds it, the earliest content drops out, and models may use information in the middle of a long document less reliably. For large datasets, send the data in **batches through code** rather than pasting everything into one chat.

> **Watch out:** Context lengths change quickly: GPT-4 launched in ChatGPT with 8,192 tokens in March 2023, and many current models handle hundreds of thousands of tokens, some over a million. Check the documentation for the model you're using.

<a id="multimodal-llm"></a>

### Multimodal LLM

_Concept_

A model that processes two or more modalities (text, images, audio, video) and the relationships between them, for example interpreting an X-ray together with the doctor's notes. Most major assistants are now multimodal: they can read charts, screenshots and PDFs. For data scientists that means asking about a chart image or turning a photographed table into data, always checking the result.

<a id="hallucination"></a>

### Hallucination

_Concept_

Fluent, confident, **invented** content. The classic case: ChatGPT claimed a law professor was at the centre of a harassment scandal, citing a newspaper article that didn't exist. Never trust citations, numbers or facts from an LLM without checking them against a real source.

<a id="ethical-concerns-with-genai"></a>

### Ethical concerns with GenAI

_Concept_

Four from the deck: **bias and fairness** (models reflect biases in their training data); **transparency and accountability** (it's often unclear what a model was trained on, why it gave an answer, and who's responsible for errors; undisclosed training data also raises copyright questions); **data privacy and security** (data sent to an AI service may be stored or used depending on its terms; one 2023 analysis found about 11% of what employees pasted into ChatGPT was confidential); and **social and environmental impact** (jobs and skills, and the large energy and water use of training and running models).

See also: [Ethical, legal and strategic implications](#ethical-legal-and-strategic-implications), [Data privacy with GenAI tools](#data-privacy-with-genai-tools)

<a id="data-privacy-with-genai-tools"></a>

### Data privacy with GenAI tools

_Concept, Python_

Never paste confidential, personal or client data into tools your organization hasn't approved; check your AI policy. And never put **API keys** directly in notebook code you might share: read them from an environment variable or a secrets store.

```python
import os
api_key = os.environ["GEMINI_API_KEY"]            # set outside the notebook

from google.colab import userdata                 # in Google Colab: the Secrets panel
api_key = userdata.get("GEMINI_API_KEY")
```

See also: [Data minimization and anonymization](#data-minimization-and-anonymization)

<a id="where-genai-helps-in-the-data-science-lifecycle"></a>

### Where GenAI helps in the data science lifecycle

_Concept_

Good candidates are tasks that can be distilled into a precise prompt, need less expert judgement, and give **checkable** results: boilerplate and repetitive code, documentation, regular expressions, explaining errors or unfamiliar APIs, first-draft visualizations, and parsing messy text (free text, addresses, scraped data). Poor candidates: framing the problem, choosing metrics or features, judging whether results make business sense, ethical decisions, and anything needing confidential data in an unapproved tool.

The course's judgement calls: choosing a model's independent variables, **not suitable** (needs domain expertise; risk of features that correlate but fail a sense check); documenting a function, **good** (easy to check in code review); understanding an unfamiliar API, **good** (quick to test); standardizing messy addresses, **good with checking** (watch for invented details); sentiment analysis of reviews, **reasonable**, if validated against a hand-labelled sample.

**Rule of thumb: use GenAI where you can verify the answer more easily than you could produce it yourself.**

<a id="generation-mode-vs-analysis-mode"></a>

### Generation mode vs analysis mode

_Concept_

Two ways a data scientist uses GenAI. In **generation mode** the model produces content, typically code, from a prompt; you check it by running and testing it. In **analysis mode** you give the model content (text data) to clean, classify, extract from or summarize; you check it by spot-checking against the source and measuring accuracy on a labelled sample.

See also: [Testing AI-generated code](#testing-ai-generated-code), [Validating GenAI classifications](#validating-genai-classifications)

<a id="chat-interface-vs-api"></a>

### Chat interface vs API

_Concept_

Chat interfaces are fine for quick, one-off tasks. For anything large-scale or repeatable, call the model through its **API** (application programming interface) from your own Python code: you can process thousands of rows in a loop, control settings such as temperature, ask for structured output and log the results. It usually needs an API key and may cost money per token.

See also: [Calling an LLM from Python](#calling-an-llm-from-python)

<a id="calling-an-llm-from-python"></a>

### Calling an LLM from Python

_Python_

The typical pattern: install the provider's SDK, read the API key securely, create a client, then loop over rows of data, sending each in a prompt with fixed instructions, a low temperature and a constrained output format, and collect the responses. With Google's current Gemini SDK (`pip install google-genai`):

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)
config = types.GenerateContentConfig(temperature=0, system_instruction=(
    "Classify the news article into exactly one of: politics, business, sport, health, other. "
    "Reply with the label only."))

labels = []
for text in df["article"].head(20):                 # start small, then scale up
    response = client.models.generate_content(model="gemini-2.5-flash", contents=text, config=config)
    labels.append(response.text.strip().lower())
df.loc[df.index[:20], "ai_label"] = labels
```

> **Watch out:** SDKs and model names change often: the older `google-generativeai` package has been replaced by `google-genai`, and model names are retired regularly. Check the provider's current documentation, and test the install cell before class.

<a id="prompting-for-code"></a>

### Prompting for code

_Concept_

Say what the code should do, the inputs and outputs, the libraries to use and any constraints; paste the **full** error message when debugging; ask the model to explain its code; and iterate rather than expecting perfection first time. Useful coding tasks: write code from scratch, write documentation, explain undocumented code, explain technical documentation, review code and suggest improvements, and debug. A good reliability test is a web scraper (e.g. "print Wikipedia's featured-article box"): generated code often guesses HTML element IDs and breaks when the page layout has changed.

<a id="genai-tools-for-data-scientists"></a>

### GenAI tools for data scientists

_Concept_

As of August 2025: **GPT** (OpenAI; chat and API); **Gemini** (Google; integrates with Google Colab); **Claude** (Anthropic; chat and API); **Cursor** (an AI-first code editor built on VS Code, whose inline edit can see your project files); **GitHub Copilot** (coding assistant inside VS Code, JetBrains and other editors); **Perplexity** (AI search); and **NotebookLM** (turns uploaded documents into summaries and podcast-style audio overviews). The "best" tool changes every few months, so choose based on your organization's approved tools, data-privacy terms, cost, and how well it does on **your** tasks: run the same prompt in several and compare whether the code runs, does what was asked, is concise, and handles errors.

<a id="parsing-unstructured-data-with-genai"></a>

### Parsing unstructured data with GenAI

_Concept_

Analysis-mode tasks: standardizing formats, extracting entities (names, places, dates), classifying text into fixed categories, and summarizing. Ask for **structured** output (a fixed set of labels, or JSON) so results can be loaded into pandas and checked. Messy numeric data needs special care, because AI tools and AI-written code often get some cases **silently** wrong: currency symbols and codes, thousands separators vs decimal commas ("4,50" in European formats), units ("per lb" vs "per kg"), abbreviations ("2.5k") and missing values.

> **Watch out:** Learn EDA without AI first; AI can then speed you up, but it isn't a replacement for learning the basics.

<a id="modelling-with-genai-what-to-check"></a>

### Modelling with GenAI: what to check

_Concept_

AI-generated modelling code often skips a step. Check for: a train-test split **before** any fitting; scaling and other preprocessing fitted on training data only (ideally in a pipeline); no leakage (the target, or features derived from it); sensible handling of categorical and missing values; and metrics computed on the test set and compared with a baseline.

See also: [Data leakage (in model evaluation)](#data-leakage-in-model-evaluation), [Pipeline + GridSearchCV](#pipeline-gridsearchcv)

<a id="how-genai-output-goes-wrong"></a>

### How GenAI output goes wrong

_Concept_

When writing code, GenAI can produce code that (1) doesn't run; (2) **runs and looks correct but does a different task**; or (3) works but is longer and more verbose than necessary. Number 2 is the most dangerous: no error message, just wrong results. When analysing data, it can draw the wrong conclusions, misclassify records, clean data inconsistently, and give different answers for the same row on different runs.

<a id="testing-ai-generated-code"></a>

### Testing AI-generated code

_Python_

Check generated code against inputs where **you** know the right answer, including awkward edge cases. A typical first attempt at "convert price strings to numbers":

```python
def clean_price(text):                       # what a GenAI tool might write
    return float(text.replace("$", "").replace(",", "").strip())

tests = {"$1,234.50": 1234.5, "  99 ": 99.0, "USD 3.20": 3.2, "2.5k": 2500.0, "€4,50": 4.5}
for raw, expected in tests.items():
    try:
        result = clean_price(raw)
        status = "OK" if abs(result - expected) < 1e-9 else f"WRONG (got {result})"
    except ValueError:
        status = "ERROR"
    print(f"{raw!r:13} -> {status}")
# '$1,234.50' OK | '  99 ' OK | 'USD 3.20' ERROR | '2.5k' ERROR | '€4,50' ERROR
```

Two inputs work and three fail. The worst case is silent: if the function also stripped the € sign, "€4,50" would become 450, wrong by a factor of 100 with no error. Tests with known answers catch this; eyeballing a few rows might not. Other checks: a line-by-line code review, reading the documentation it wrote against what the code actually does, and **rubber duck debugging** (explaining the code out loud, line by line, until gaps in understanding become obvious).

<a id="validating-genai-classifications"></a>

### Validating GenAI classifications

_scikit-learn_

When GenAI labels data, treat it like any classifier: hand-label a random sample (say 50–100 rows), compare with the AI's labels, and build a confusion matrix. Report precision and recall per class, not just accuracy, exactly as in the classification metrics section.

```python
from sklearn.metrics import confusion_matrix, classification_report

sample = df.sample(100, random_state=42)          # hand-label these in a "human_label" column
print(confusion_matrix(sample["human_label"], sample["ai_label"]))
print(classification_report(sample["human_label"], sample["ai_label"]))
```

See also: [Confusion matrix](#confusion-matrix), [Macro, weighted and micro averages](#macro-weighted-and-micro-averages)

---

<a id="s25-time-series-data-handling-and-stationarity"></a>

## 25. Time series: data handling and stationarity

_Data where order matters. Working with dates in pandas (indexing, resampling, rolling windows, lags and differences), stationarity, and reading autocorrelation plots. Examples use Apple stock prices (`AAPL.csv`), Australian population, New Haven temperatures and UK lung disease deaths._

**In your capstone:** If your data has a time order, plot it, check stationarity and split chronologically before anything else.

<a id="time-series"></a>

### Time series

_Concept_

A series of observations taken at **equally spaced intervals** (days, months, quarters, years). The data are **dependent**: each row is related to the rows before it, so we care about row-to-row relationships. That breaks the independence assumption behind OLS and ordinary train-test splitting. Forecasting works best when the past genuinely predicts the future: weather, sales and deaths, yes; the stock market, no.

Most classical methods assume **equally spaced** observations; irregular data (transactions at random times) is usually resampled onto a regular grid first. A league table at one moment is **cross-sectional** data; a team's points recorded every week would be a time series. Because the order carries information, you can't shuffle the rows, and there's no random train-test split.

Notation: capital Xₜ is the random process; lower-case xₜ the values actually observed; xₜ₋ₕ is the value h periods ago, where **h is the lag**.

See also: [Johnson & Johnson earnings (running example)](#johnson-and-johnson-earnings-running-example)

See also: [Statistical independence](#statistical-independence)

<a id="johnson-and-johnson-earnings-running-example"></a>

### Johnson & Johnson earnings (running example)

_statsmodels, pandas_

The quarterly earnings per share of Johnson & Johnson, 1960–1980, is the running example in the Time Series I and II notes. It's non-stationary in three ways at once: an upward **trend**, quarterly **seasonality**, and swings that **grow over time** (increasing variance). It's available through statsmodels (an internet connection is needed to download it).

```python
import statsmodels.api as sm

jj = sm.datasets.get_rdataset("JohnsonJohnson").data
jj.index = pd.period_range("1960Q1", periods=len(jj), freq="Q").to_timestamp()
x = jj["value"]                                    # quarterly earnings per share
x.groupby(x.index.year).mean().head(3)             # yearly averages: 0.6575, 0.6925, 0.7525
```

**Offline alternative:** the course `data/` folder has the same series as `jj-sales.csv`, alongside `lung-deaths.csv` (monthly UK lung-disease deaths, 1974–1979) and `bj-sales.csv` (the Box–Jenkins sales series). These are the files `graph-gen-solution.ipynb` uses to draw the slide figures.

```python
jj = pd.read_csv("../data/jj-sales.csv")
jj["date"] = pd.to_datetime(jj["date"])
jj = jj.set_index("date")
```

<a id="datetime-index"></a>

### Datetime index

_pandas_

Most time series tools expect the dates to be the DataFrame's index. Convert the column to datetime, set it as the index and sort.

```python
aapl = pd.read_csv("data/AAPL.csv")
aapl["Date"] = pd.to_datetime(aapl["Date"])
aapl.set_index("Date", inplace=True)
aapl.sort_index(inplace=True)
```

See also: [pd.to_datetime() (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#pd-to-datetime), [.set_index() (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#set-index-and-reset-index)

<a id="date-based-slicing"></a>

### Date-based slicing

_pandas_

With a datetime index, `.loc` accepts partial date strings: a year, a year-month, or a range. Index attributes such as `.month` filter across years.

```python
aapl.loc["2018"]                          # all of 2018
aapl.loc["2019-01"]                       # January 2019
aapl.loc["2018-06":"2018-09"]             # a range
aapl.loc[aapl.index.month == 1]           # every January
```

<a id="resample-for-time-series"></a>

### .resample() for time series

_pandas_

Like `groupby`, but by time period: change the frequency of the data and aggregate. Common codes: `"D"` day, `"W"` week, `"ME"` month end, `"MS"` month start, `"QE"` quarter end, `"QS"` quarter start, `"YE"` year end.

```python
aapl.resample("ME").mean()      # monthly averages
aapl.resample("QS").sum()       # quarterly totals, labelled by quarter start
```

> **Watch out:** Frequency codes changed recently, so check the pandas documentation for your version: pandas 2.2 deprecated `"M"`, `"Q"` and `"Y"` and pandas 3 rejects them; use `"ME"`, `"QE"` and `"YE"`. Many tutorials still use the old codes. Resampling to `"D"` also creates rows for weekends, filled with 0 when you `.sum()`.

`groupby(x.index.year)` is a version-proof alternative for yearly summaries.

See also: [.resample() (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#resample)

<a id="rolling-statistics"></a>

### Rolling statistics

_pandas_

A statistic over a **moving window** of the last h periods. The 3-day rolling mean is Mₜ = (Xₜ + Xₜ₋₁ + Xₜ₋₂)/3. The first h − 1 values are NaN because the window isn't full yet. A rolling mean is a simple **smoother** and even a simple forecast.

```python
aapl["Adj Close"].rolling(3).mean()
aapl["Adj Close"].rolling(12).std()
```

> **Watch out:** A rolling forecast uses "delayed" information, and choosing the wrong window makes it either too jumpy or too slow to react.

To use a rolling mean as a **forecast**, shift it so each prediction only uses past values: `y.rolling(h).mean().shift(1)`. The window size is a bias–variance trade-off: a small window reacts quickly but is noisy; a large window is smooth but lags behind. A 4-quarter window also averages out yearly seasonality in quarterly data.

<a id="macd-chart"></a>

### MACD chart

_matplotlib, pandas_

The "fun fact" in the notebook: plot a fast (12-day) and slow (26-day) moving average of a price. A simple trading rule buys or sells when the lines cross.

```python
adj = aapl["Adj Close"]
plt.plot(adj, color="whitesmoke")
plt.plot(adj.rolling(12).mean(), label="Fast Line")
plt.plot(adj.rolling(26).mean(), label="Slow Line")
plt.legend()
```

<a id="lag-backshift-and-shift"></a>

### Lag (backshift) and .shift()

_pandas_

**Backshifting** slides every value down one period: BXₜ = Xₜ₋₁. The new column is called the **first lag**. Lag features let a regular model use yesterday's value to predict today's.

```python
aapl["open_lag1"] = aapl["Open"].shift()       # yesterday's open
aapl["Open"].shift(7)                          # one week ago
```

The first row of a shift (or diff) is always NaN: you lose one row per lag, so decide whether to drop it.

```python
pd.DataFrame({"x": x, "shift_1": x.shift(1), "diff_1": x.diff()}).head(3)
# x = 0.71, 0.63, 0.85; shift_1 = NaN, 0.71, 0.63; diff_1 = NaN, -0.08, 0.22
```

<a id="differencing-and-diff"></a>

### Differencing and .diff()

_pandas_

Today's value minus yesterday's: ∇Xₜ = Xₜ − Xₜ₋₁ = (1 − B)Xₜ (∇ is "nabla", B the backshift). Differencing removes a steady trend: if sales rise by about 10 a month, the differences hover around 10 with no trend. It's the main way to make a series stationary.

```python
aapl["open_diff"] = aapl["Open"].diff()        # same as Open - Open.shift()
air.diff().diff()                              # second difference: the change in the change
x.diff(12)                                     # seasonal difference: this month minus 12 months ago
```

> **Watch out:** Differencing twice, `x.diff().diff()` = Xₜ − 2Xₜ₋₁ + Xₜ₋₂ (like acceleration vs speed), is **not** the same as `x.diff(2)` = Xₜ − Xₜ₋₂, a lag-2 difference.

See also: [Seasonal differencing](#seasonal-differencing)

<a id="seasonal-differencing"></a>

### Seasonal differencing

_pandas_

Subtracting the value from the same point in the previous cycle: `x.diff(12)` for monthly data, `x.diff(4)` for quarterly. It removes a repeating seasonal pattern the way ordinary differencing removes a trend, and the two are often combined (`x.diff().diff(4)`). Seasonal ARIMA models (SARIMA) build it in.

<a id="stationarity"></a>

### Stationarity

_Concept_

A series is **(weakly) stationary** if three things don't change over time: (1) its **mean** (no trend); (2) its **variance** (no growing or shrinking swings); and (3) its **autocorrelation**, which depends only on how far apart two values are, not on when you look. Most classical forecasting models (AR, MA, ARMA) assume it, because they learn patterns from the past and need those patterns to persist.

Australia's population is clearly not stationary (strong upward trend) but its first difference roughly is; New Haven's annual temperature is "trendless enough" already. Johnson & Johnson earnings need a **log transform first**: differencing alone leaves it non-stationary (ADF p-value 0.909), while log-then-difference makes it stationary (p ≈ 0.000). Logs turn swings that grow with the level into constant-sized ones, and differences of logs are approximately percentage changes.

> **Watch out:** It's easy to treat a trendless seasonal series as stationary. Strictly, seasonality breaks stationarity: with a regular seasonal pattern (like the UK lung-disease deaths) the expected value in January differs from July, so the mean depends on time. Such a series has no long-term **trend** but still needs seasonal differencing or a SARIMA model.

See also: [Augmented Dickey–Fuller (ADF) test](#augmented-dickey-fuller-adf-test), [Differencing and .diff()](#differencing-and-diff)

<a id="autocorrelation"></a>

### Autocorrelation

_Concept, pandas_

The correlation (−1 to 1) of a series with **a copy of itself shifted back h periods** ("auto" = self). Autocorrelation at lag 1 compares each value with the previous one; at lag 12 (monthly data), with the same month last year. Strong autocorrelation is why time series data aren't independent, and high autocorrelation at lag h means the value h periods ago helps predict today: the basis of autoregressive models. Look for the lag that matches the cycle: 7 for weekly patterns in daily data, 12 for yearly patterns in monthly data, 4 for quarterly.

```python
x.autocorr(lag=4)                                   # simplest
h = 4; np.corrcoef(x[h:], x.shift(h)[h:])[0, 1]      # same thing by hand
```

> **Watch out:** A common hand-written version, `np.corrcoef(x[1:], x.shift(h)[1:])`, only works for h = 1; for larger lags it returns NaN, because shifting by h creates h missing values but `[1:]` drops only one. statsmodels' `acf()` uses a slightly different formula (the overall mean and variance), so its values can differ, especially on trending data (raw J&J at lag 4: 0.995 vs 0.824). On non-stationary data autocorrelations are dominated by the trend, so make the series stationary first.

<a id="acf-plot"></a>

### ACF plot

_statsmodels_

The **autocorrelation function** plot shows the autocorrelation at every lag, with a shaded band of values not significantly different from zero. A slow steady decline signals a trend (non-stationary); spikes at 12, 24, 36 in monthly data signal seasonality, as in the lung deaths data (more deaths in colder months).

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(lung)
```

Lag 0 is always 1 (a series correlates perfectly with itself). A **wave** in the ACF reflects a cycle: values correlate with the same month in other years (lags 12, 24) and anti-correlate six months apart. A **slow, steady decline** is the signature of non-stationarity. After log-differencing the J&J series, the seasonal structure appears as spikes at lags 4, 8 and 12 (lag-4 autocorrelation 0.85).

```python
stationary = np.log(x).diff().dropna()
plot_acf(stationary, lags=16); plot_pacf(stationary, lags=16)
```

<a id="pacf-plot"></a>

### PACF plot

_statsmodels_

The **partial autocorrelation function** shows the correlation with each lag **after removing** the effect of the shorter lags in between. Helps choose the number of AR terms p. Values can also be computed with `acf()` and `pacf()` from `statsmodels.tsa.stattools`.

```python
plot_pacf(lung)
from statsmodels.tsa.stattools import acf, pacf
acf(lung.deaths, nlags=24)
```

It shows the **direct** link between xₜ and xₜ₋ₕ rather than links passed along through intermediate days. With a pure 7-day pattern, the ACF shows echoes at every multiple of 7, while the PACF shows only the original lag-7 link. A slowly decaying ACF plus a PACF with one big spike at lag 1 is the classic picture of a trending, random-walk-like series: difference first.

See also: [Reading ACF and PACF to choose p and q](#reading-acf-and-pacf-to-choose-p-and-q)

<a id="seasonality"></a>

### Seasonality

_Concept_

A pattern that repeats with a fixed, known **period** d: d = 12 for monthly data, d = 4 for quarterly, d = 7 for daily data with a weekly cycle. Mathematically sₜ = sₜ₊d: the December 2010 effect equals the December 2011 effect.

Common periods: 7 (daily data, weekly cycle), 12 (monthly, yearly), 4 (quarterly, yearly), 24 (hourly, daily). A series can have several cycles at once, which makes it harder to model. ("Paid every two weeks" is fortnightly or biweekly.)

<a id="trend"></a>

### Trend

_Concept_

The long-run upward or downward movement of a series (mₜ). It can be estimated with OLS on time, a rolling mean, or removed by differencing.

<a id="white-noise"></a>

### White noise

_Concept_

A series of independent random values with mean zero and constant variance: no pattern left to model. The goal of time series modeling, as with OLS residuals, is to reduce what's left over to white noise.

Checking residuals for white noise is the time series version of the LINE checks: plot them, look at their ACF (no significant bars = random), or run a Ljung–Box test.

See also: [Ljung–Box test](#ljung-box-test)

---

<a id="s26-time-series-forecasting-models"></a>

## 26. Time series: forecasting models

_From simple decomposition to ARIMA, then three extensions from Time Series III: GLS (adding x-variables), GARCH (modeling changing volatility) and VAR (forecasting several series together). Examples use Nottingham temperatures, airline passengers, Disney stock and U.S. macroeconomic data (`sm.datasets.macrodata`)._

**In your capstone:** Compare every forecast with naive and seasonal-naive baselines, choose orders by AIC, check the residuals, then refit on all the data.

<a id="classical-decomposition"></a>

### Classical decomposition

_Concept_

Splitting a series into three parts. The **additive** model is **Yₜ = mₜ + sₜ + εₜ**: a **trend**, a **seasonal** component with a fixed period, and random **noise**. If what's left after removing trend and seasonality is pure noise, you have a sensible forecasting model: project the trend and repeat the seasonal pattern. In an additive model the seasonal values average to zero over a cycle; they push the series above or below the trend.

When seasonal swings **grow with the level** (as in J&J), use the **multiplicative** model **Yₜ = mₜ × sₜ × εₜ**, which is equivalent to an additive model on log(Yₜ). Classical decomposition also assumes the seasonal pattern is identical in every cycle, a simplification.

> **Watch out:** The letter d is overloaded in time series: it can mean the deseasonalized series dₜ, the number of differences in ARIMA(p, d, q), or the seasonal period. Check which one a formula means.

See also: [seasonal_decompose()](#seasonal-decompose)

<a id="seasonal-decompose"></a>

### seasonal_decompose()

_statsmodels_

Decomposes a series and plots the observed data, trend, seasonal and residual components. Choose `model="additive"` or `"multiplicative"` and set the `period`.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

add = seasonal_decompose(y, model="additive", period=4)
mult = seasonal_decompose(y, model="multiplicative", period=4)
add.plot(); mult.plot()                     # compare the residual panels
add.seasonal[:4].round(3).tolist()          # J&J: [0.222, 0.244, 0.309, -0.774]  amounts added
mult.seasonal[:4].round(3).tolist()         # J&J: [0.993, 1.033, 1.114, 0.86]   factors multiplied
```

On J&J the additive residuals fan out over time (wrong model for growing swings), while the multiplicative residuals are much more even: Q4 sits about 14% below trend. That motivates taking logs before ARIMA.

> **Watch out:** `seasonal_decompose` is itself a classical centred-moving-average method, much like decomposing by hand; for a more robust modern decomposition use `statsmodels.tsa.seasonal.STL`.

<a id="naive-and-seasonal-naive-forecasts"></a>

### Naive and seasonal naive forecasts

_Concept, NumPy_

The baselines every forecast must beat. The **naive** forecast says the next value equals the latest observed value (that's exactly ARIMA(0, 1, 0), a random walk). The **seasonal naive** forecast repeats the value from the same point in the previous cycle (same quarter last year). On J&J's last two years, MAE is 4.88 for naive, 2.54 for seasonal naive, and 0.67 for a seasonal ARIMA: baselines show how much the model really adds.

```python
naive = np.repeat(train.iloc[-1], len(test))                      # repeat the last value
seasonal_naive = np.tile(train.iloc[-4:].values, len(test) // 4)   # repeat the last year
np.mean(np.abs(test.values - seasonal_naive))                     # MAE
```

See also: [Null model (baseline) for regression](#null-model-baseline-for-regression)

<a id="seasonal-mean-forecast"></a>

### Seasonal-mean forecast

_pandas_

A "simple" model from the notebook that's surprisingly good for Nottingham temperatures: assume no trend (mₜ = mean), estimate sₜ as each month's mean minus the overall mean, and predict trend + seasonal effect.

```python
mt = y_train.temp.mean()
st = y_train.groupby("month", as_index=False).temp.mean()
st["seasonal"] = st.temp - mt
pred = y_test.merge(st[["month", "seasonal"]], how="left")
pred["y_pred"] = mt + pred["seasonal"]
```

<a id="time-series-train-test-split"></a>

### Time series train/test split

_scikit-learn, pandas_

Never shuffle a time series: the test set must come **after** the training set, or you'd be training on the future. Either pass `shuffle=False` or slice off the last h rows.

```python
y_train, y_test = train_test_split(nottem, test_size=0.1, shuffle=False)
y_train, y_test = air.iloc[:-24], air.iloc[-24:]       # hold out the last two years
```

The full forecasting workflow from the course: parse dates and set a DatetimeIndex → plot → check stationarity (look, then ADF) → transform and difference → read the ACF/PACF → split chronologically → build naive baselines → fit candidate models and compare AIC → check residuals → forecast the test period and compare MAE with the baselines → **refit on all the data** before forecasting the real future.

<a id="autoregression-ar"></a>

### Autoregression (AR)

_Concept_

A model that regresses a series on **its own past values**: AR(p) uses the last p values, Yₜ = φ₀ + φ₁Yₜ₋₁ + … + φₚYₜ₋ₚ + εₜ. For certain φ values it's the same as a rolling mean.

φ (phi) is the weight on a past value: if φ₁ = 0.8, today is 80% an echo of yesterday plus a fresh shock. Autocorrelation **measures** the relationship with the past; autoregression **uses** it to predict. In practice a constant is usually included. For a pure AR(p), the **PACF cuts off after lag p**. One-line memory aid: AR = tomorrow looks a bit like today.

<a id="moving-average-model-ma"></a>

### Moving average model (MA)

_Concept_

A model that predicts the current value from **past error terms**: MA(q) uses the last q errors. It captures autocorrelation structure, especially seasonality at a given lag.

> **Watch out:** This is **not** the rolling mean, which is also commonly called a "moving average".

It's an average of past **errors** (shocks), not of the data: θ (theta) is the weight on a past shock, εₜ₋₁ is how far off you were yesterday, and MA(1) carries a fraction θ₁ of it into today. Examples: a promotion that boosts sales this week and a bit next week; a viral post that lifts traffic for a few days. For a pure MA(q), the **ACF cuts off after lag q**. Memory aid: MA = yesterday's surprise echoes into today.

<a id="arma-p-q"></a>

### ARMA(p, q)

_Concept_

AR(p) and MA(q) combined: Yₜ = intercept + AR terms + MA terms + error. Requires a stationary series.

ARMA captures both "momentum" (AR) and "shock carry-over" (MA) with few parameters.

<a id="arima-p-d-q"></a>

### ARIMA(p, d, q)

_Concept, statsmodels_

An ARMA model fitted to a series that has been **differenced d times** to make it stationary; the I stands for **Integrated**. p = number of AR terms, d = number of differences, q = number of MA terms. The base model for time series forecasting, with many variants.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y_train, order=(1, 2, 1)).fit()
model.summary()
y_test_pred = model.predict(start=y_test.index[0], end=y_test.index[-1])
```

"Integrated" because the original series is recovered by summing (integrating) the differences. Memory aid: I = remove the trend first. Useful special cases: ARIMA(1, 0, 0) = AR(1); **ARIMA(0, 1, 0) = a random walk** (forecast = last value, the naive forecast); ARIMA(0, 1, 1) is similar to simple exponential smoothing. `model.forecast(steps)` forecasts beyond the end of the training data.

If you modelled log(y), convert forecasts back with `np.exp()` before computing MAE, so errors are in the original units.

See also: [SARIMA](#sarima), [Naive and seasonal naive forecasts](#naive-and-seasonal-naive-forecasts)

<a id="sarima"></a>

### SARIMA

_statsmodels_

Seasonal ARIMA adds seasonal terms (seasonal AR, differencing and MA at the seasonal lag) through `seasonal_order=(P, D, Q, s)`. The course demonstration fits J&J's log earnings with a first difference plus a seasonal difference, choosing p and q by AIC:

```python
import itertools
from statsmodels.tsa.arima.model import ARIMA

y = np.log(jj["value"])
train, test = y[:-8], y[-8:]                                  # last 2 years held out
results = []
for p, q in itertools.product(range(3), range(3)):
    fit = ARIMA(train, order=(p, 1, q), seasonal_order=(0, 1, 0, 4)).fit()
    results.append(((p, 1, q), round(float(fit.aic), 1)))
best = min(results, key=lambda r: r[1])                       # ((1, 1, 1), -125.4)
fit = ARIMA(train, order=best[0], seasonal_order=(0, 1, 0, 4)).fit()
forecast = np.exp(fit.forecast(8))                            # back to earnings per share
np.mean(np.abs(np.exp(test) - forecast))                      # MAE ≈ 0.67 (seasonal naive ≈ 2.54)
```

**SARIMAX** adds external predictors (the X), like ARIMA's `exog`. statsmodels may print convergence warnings for some grid combinations; that's common.

See also: [Other forecasting models](#other-forecasting-models)

<a id="other-forecasting-models"></a>

### Other forecasting models

_Concept_

Beyond ARIMA, from the deck's "what else?" slide: **ETS / Holt–Winters** exponential smoothing models level, trend and seasonality with exponentially weighted averages; **Prophet** combines trend, seasonality, holidays and events and is designed for accessible business forecasting; and **tree-based models** (e.g. XGBoost) forecast using lag features, rolling statistics, calendar features and external predictors.

<a id="choosing-p-d-and-q"></a>

### Choosing p, d and q

_Concept_

**d** is the smallest number of differences that makes the series stationary, judged by eye or with the ADF test. **p** and **q** are shortlisted from the ACF/PACF of the **stationary** series, then decided by searching over values and keeping the model with the lowest AIC (and residuals that look like white noise).

```python
from itertools import product

results = []
for p, q in product(range(13), range(13)):
    m = ARIMA(y_train, order=(p, 2, q)).fit()
    results.append((p, q, m.aic))
pd.DataFrame(results, columns=["p", "q", "aic"]).sort_values("aic").head()
```

> **Watch out:** Don't use `GridSearchCV` here: its random folds would train on the future and test on the past. Time series need a chronological hold-out or `TimeSeriesSplit`.

See also: [Reading ACF and PACF to choose p and q](#reading-acf-and-pacf-to-choose-p-and-q)

<a id="reading-acf-and-pacf-to-choose-p-and-q"></a>

### Reading ACF and PACF to choose p and q

_Concept_

Rules of thumb for a **stationary** series: for a pure **AR(p)**, the PACF cuts off after lag p (and the ACF decays gradually); for a pure **MA(q)**, the ACF cuts off after lag q (and the PACF decays). Mixed ARMA processes show gradual decay in both. Real data is messier than the textbook pictures: on differenced J&J, both plots spike at lag 1 (about −0.38) and at the seasonal lag 4, suggesting several candidates; the AIC search then picks ARIMA(1, 1, 1). Use the plots to shortlist, and let AIC and residual checks decide.

> **Watch out:** Read the plots of the **stationary** (differenced) series. If the ACF decays slowly, the series isn't stationary yet: difference again before choosing p and q.

<a id="itertools-product"></a>

### itertools.product()

_Python_

Generates every combination of items from several lists: the Cartesian product, a home-made grid. `product([0, 1], [0, 1])` gives (0,0), (0,1), (1,0), (1,1).

<a id="augmented-dickey-fuller-adf-test"></a>

### Augmented Dickey–Fuller (ADF) test

_statsmodels_

A hypothesis test for stationarity. **H₀: the series is non-stationary (has a unit root); H₁: it is stationary.** A small p-value (below 0.05) means you can treat the series as stationary. Students often get this backwards.

```python
from statsmodels.tsa.stattools import adfuller

for name, series in [("raw", x), ("diff", x.diff()), ("log diff", np.log(x).diff())]:
    print(name, round(adfuller(series.dropna())[1], 3))
# J&J: raw 1.0, diff 0.909 (still non-stationary), log diff 0.0 (stationary)
```

> **Watch out:** Drop the NaNs that differencing creates, or `adfuller` will fail. Recent statsmodels releases also warn that `adfuller`'s return format is changing; the p-value is the second value returned.

See also: [p-value (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#p-value)

<a id="ljung-box-test"></a>

### Ljung–Box test

_statsmodels_

Tests whether a model's residuals still contain autocorrelation. **H₀: no autocorrelation up to the chosen lag** (the residuals look like white noise). Here a **large** p-value is what you want: nothing left to model. On the J&J seasonal ARIMA, p = 0.22 at lag 8.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(fit.resid[5:], lags=[8])["lb_pvalue"]    # skip the first residuals lost to differencing
```

<a id="aic-akaike-information-criterion"></a>

### AIC (Akaike Information Criterion)

_Concept, statsmodels_

The preferred evaluation metric for ARIMA models in place of R². It balances fit against the number of parameters, penalizing complexity. **Lower is better.** Only compare AICs of models fitted to the same data.

```python
model.aic
```

George Box and Gwilym Jenkins' 1970 textbook popularized the ARIMA workflow (identify, estimate, check), often called the **Box–Jenkins method**; AIC-based selection is its modern form.

<a id="generalized-least-squares-gls"></a>

### Generalized least squares (GLS)

_Concept, statsmodels_

OLS assumes errors are iid: independent (the "~" in εₜ ~ N(0, σ²)) and identically distributed with one constant variance (no subscript on σ²). In matrix terms, the errors' covariance matrix is σ²I, diagonal with no correlation. **GLS allows a general covariance matrix Σ**, so errors close in time can be correlated. In practice Σ is given a simple structure such as ARMA errors: yₜ = xₜᵀβ + Wₜ, where Wₜ is an ARMA(p, q) process. That's **regression with ARMA errors**: the regression explains y using x, and the ARMA part soaks up the leftover time-correlation.

The course demonstration on US macro data: does disposable-income growth predict consumption growth?

```python
macro = sm.datasets.macrodata.load_pandas().data
macro.index = pd.period_range("1959Q1", periods=len(macro), freq="Q").to_timestamp()
growth = np.log(macro[["realcons", "realdpi"]]).diff().dropna() * 100      # % growth
ols = sm.OLS(growth["realcons"], sm.add_constant(growth["realdpi"])).fit()
gls = ARIMA(growth["realcons"], exog=growth["realdpi"], order=(1, 0, 0)).fit()   # AR(1) errors
ols.params["realdpi"], gls.params["realdpi"]                                # 0.341 vs 0.297
```

statsmodels also offers `GLSAR` for regression with autoregressive errors.

> **Watch out:** OLS isn't forbidden for time series: its coefficients can still be reasonable, but their standard errors and p-values are wrong (usually too optimistic), and the information in the error correlation is wasted.

<a id="exogenous-and-endogenous-variables"></a>

### Exogenous and endogenous variables

_Concept_

Time series vocabulary for x and y. **Exogenous** (the x-variables, from outside the system; note the **x** in e**x**ogenous) and **endogenous** (the y-variable being forecast). statsmodels arguments are named `exog` and `endog`.

Exogenous = determined **outside** the model (an input); endogenous = determined **inside** it (the target).

<a id="arima-with-exog-gls-in-python"></a>

### ARIMA with exog (GLS in Python)

_statsmodels_

Pass x-variables as `exog` to fit a GLS-style model, and supply the future x-values when predicting.

```python
y = df["realgdp"]
X = df[["realcons", "realinv", "realgovt", "cpi", "m1"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=20)
gls = ARIMA(y_train, exog=X_train, order=(2, 1, 2), freq="QE-DEC").fit()
preds = gls.predict(start=X_test.index[0], end=X_test.index[-1], exog=X_test)
```

> **Watch out:** Frequency strings changed in pandas 2.2: use `freq="QE-DEC"` (older code writes `"Q-DEC"`, which pandas 3 rejects).

<a id="statsmodels-macrodata"></a>

### statsmodels macrodata

_statsmodels, pandas_

A built-in quarterly U.S. economic dataset (real GDP, consumption, investment, unemployment, population, CPI…) used in Time Series III. Build a datetime index from the year and quarter columns.

```python
import statsmodels.api as sm
from statsmodels.tsa.base.datetools import dates_from_str

df = sm.datasets.macrodata.load_pandas().data
qtr = df["year"].astype(int).astype(str) + "Q" + df["quarter"].astype(int).astype(str)
df.index = pd.to_datetime(dates_from_str(qtr))
df = df.drop(columns=["year", "quarter"])
```

<a id="log-returns"></a>

### Log returns

_Concept, NumPy_

For a price Pₜ: Pₜ itself is non-stationary; Xₜ = log Pₜ behaves roughly like a **random walk** (also non-stationary); and the difference Zₜ = Xₜ − Xₜ₋₁, the log return (≈ the daily percentage change), is approximately **stationary** and looks like white noise. But it is **not independent** white noise: big moves cluster together. Returns themselves are nearly uncorrelated (you can't predict tomorrow's direction), but their **size** is predictable.

```python
z = 100 * np.log(prices["Adj Close"]).diff().dropna()    # scaled log returns
z.autocorr(1), (z ** 2).autocorr(1)                      # returns ≈ 0; squared returns clearly > 0
```

> **Watch out:** A random walk is **not** stationary, so log prices aren't either; it's the log **returns** that are approximately stationary. And the dependence in returns is invisible in their ACF but clearly visible in the ACF of the **squared** returns.

See also: [Volatility clustering](#volatility-clustering)

<a id="volatility-clustering"></a>

### Volatility clustering

_Concept_

Calm periods are followed by calm periods and turbulent periods by turbulent ones: the size of today's move predicts the size (not the direction) of tomorrow's. GARCH models exist to capture this.

On simulated GARCH data with 2,000 observations, the lag-1 autocorrelation is 0.017 for returns but 0.077 for squared returns (significant at that sample size).

<a id="conditional-heteroscedasticity"></a>

### Conditional heteroscedasticity

_Concept_

Variance that changes over time and depends on recent history. "Heteroscedastic" = unequal variance; "conditional" = conditional on the past. Series like this "jump around too much" for ARIMA.

It relaxes the **E** (equal variance) assumption, just as GLS relaxes the **I** (independence) assumption.

<a id="arch-p"></a>

### ARCH(p)

_Concept_

**A**uto**R**egressive **C**onditional **H**eteroscedasticity. Models the series as Zₜ ~ N(0, hₜ) where the variance hₜ = α₀ + Σαᵢ Z²ₜ₋ᵢ depends on the last p squared values. It looks like an AR(p) process, but on the conditional **variance** of Zₜ rather than its value.

An equivalent form: Zₜ = √hₜ · εₜ with εₜ ~ N(0, 1). A big move yesterday, up **or** down, raises today's variance. ARCH was introduced by Robert Engle (1982), who shared the 2003 Nobel Memorial Prize in Economics partly for it.

<a id="garch-p-q"></a>

### GARCH(p, q)

_Concept, arch_

**G**eneralized ARCH (Tim Bollerslev, 1986): adds past variances so volatility decays gradually instead of depending only on the last few returns: hₜ = α₀ + Σαᵢ Z²ₜ₋ᵢ + Σβⱼ hₜ₋ⱼ. Intuition: today's volatility = baseline + reaction to recent shocks (α) + persistence of recent volatility (β). The α sum is "AR-like" and the β sum "MA-like", though the analogy is loose. Used heavily in finance to forecast **volatility** (risk, option pricing). Its mean forecast is usually flat near zero; the useful output is the forecast **variance**.

**GARCH(1, 1)** is the workhorse; α + β close to 1 means volatility shocks are persistent (typical values: α ≈ 0.05–0.1, β ≈ 0.85–0.95). In the `arch` package, `p` counts the squared-return (α) terms and `q` the variance (β) terms.

```python
from arch import arch_model   # pip install arch

z_train, z_test = train_test_split(z, shuffle=False, test_size=0.05)
garch = arch_model(z_train, vol="Garch", p=1, q=1).fit(disp="off")
garch.params                        # omega, alpha[1], beta[1]
preds = garch.forecast(horizon=len(z_test), reindex=False)
preds.variance                      # forecast volatility along the horizon
```

Fitting simulated data with known parameters (ω 0.1, α 0.1, β 0.85) recovers them well (0.108, 0.083, 0.866).

> **Watch out:** GARCH isn't in statsmodels; it needs the `arch` package. `arch` asks you to rescale small numbers (like daily returns) for numerical stability, which is why returns are multiplied by 100. The letter h means the conditional variance here, but the lag in Time Series I and hot-coffee sales in the VAR example.

<a id="univariate-vs-multivariate-time-series"></a>

### Univariate vs multivariate time series

_Concept_

**Univariate**: forecasting one response variable. **Multivariate**: forecasting several at once in a single model, like simultaneous regression, because the series influence each other (hot and iced coffee sales).

Examples: GDP and unemployment; sales of related products; temperatures in neighbouring cities. In the coffee case the relationship works through a shared driver (the weather) and substitution (a customer buys one or the other).

<a id="vector-autoregression-var"></a>

### Vector autoregression (VAR)

_Concept, statsmodels_

A multivariate AR model in which each series is predicted from the past values of **every** series. Two series with one lag: hₜ = β₁₁hₜ₋₁ + β₁₂iₜ₋₁ + ε₁ₜ and iₜ = β₂₁hₜ₋₁ + β₂₂iₜ₋₁ + ε₂ₜ, written compactly as **yₜ = Byₜ₋₁ + εₜ** with a coefficient matrix B. Each series must be made stationary first.

```python
from statsmodels.tsa.api import VAR

df_var_diff = pd.DataFrame({
    "pop_d2": df["pop"].diff().diff(),
    "unemp_d1": df["unemp"].diff(),
    "realgdp_d1": df["realgdp"].diff(),
}).dropna()
model = VAR(df_var_diff, freq="QE-DEC").fit(maxlags=15, ic="aic")
model.k_ar               # number of lags chosen (9 in the notebook)
model.plot_forecast(10)
```

The **cross terms** (β₁₂: yesterday's iced sales → today's hot sales; β₂₁ the reverse) are what VAR adds over separate AR models; if they matter, one series helps forecast the other (tested formally with **Granger causality**). The number of parameters grows quickly, k² per lag for k series, so VARs usually use few series and lags. A second course demonstration models GDP growth and consumption growth together:

```python
data = growth[["realgdp", "realcons"]]
var = VAR(data).fit(maxlags=8, ic="aic")
var.k_ar                                                  # 3 lags chosen by AIC
var.forecast(data.values[-var.k_ar:], steps=2)            # next 2 quarters, both series
```

---

<a id="s27-simulation"></a>

## 27. Simulation

_Building a computer model of a system and running it to see what happens under different assumptions, without touching the real system. The components and pillars of a simulation, the four common approaches (discrete event, agent-based, Monte Carlo and system dynamics), building simulations in Python with NumPy, and checking that they can be trusted. Examples use a coffee shop queue, a flood-risk estimate and a town's population._

**In your capstone:** Use simulation for "what if" questions and to put uncertainty ranges (percentiles) around projections such as ROI.

<a id="simulation"></a>

### Simulation

_Concept_

A model of a real-world system or process that you can run to experiment safely and cheaply: test scenarios, make predictions and understand behaviour without real-world risks or costs. Flight simulators, weather forecasts and video games are all simulations. Businesses use them to forecast demand and test pricing, hospitals to predict patient flow, engineers to model traffic.

**Simulation vs machine learning:** ML learns patterns from historical data; a simulation **encodes how we believe a system works** (rules plus randomness) and explores the consequences, including scenarios that have never happened.

See also: [Simulation (Monte Carlo)](#simulation-monte-carlo)

<a id="analytical-model-vs-simulation"></a>

### Analytical model vs simulation

_Concept_

An **analytical model** is a formula you can solve exactly: the flood example's 20% × 10% = 2% is one. Simulations earn their keep when systems are too complex for neat formulas: queues, interactions between many individuals, feedback loops, continuously varying inputs. Simple cases with known answers are still useful for **checking** that a simulation works.

<a id="benefits-of-simulation"></a>

### Benefits of simulation

_Concept_

From the slides: **safe** (test "what-ifs" such as staffing changes without risking production); **cost-effective** (test a marketing campaign without alerting competitors); **experiment** (a hospital trials staff schedules for peak hours); **understand dynamics** (watch behaviour unfold over time, such as traffic-light timing); and **forecast accuracy**.

> **Watch out:** More detail doesn't automatically mean more accuracy. A detailed simulation built on wrong assumptions is precisely wrong (see the Simplification pillar and the limitations of simulation).

<a id="key-components-of-a-simulation"></a>

### Key components of a simulation

_Concept_

Five components describe any simulation, shown here for a coffee shop:

- **System and entities:** the scenario and its objects (the shop; customers, baristas, machines).
- **State and events:** the current condition (queue length, barista busy) and what changes it (an arrival, an order, a drink served).
- **Processes and rules:** behaviour logic, sometimes random (arrive → order → wait → receive → leave; a drink takes 2–5 minutes).
- **Time and execution:** how the clock advances.
- **Inputs and outputs:** parameters (arrival rate, preparation times) and results (average wait, customers served per hour, **utilization**).

See also: [Time handling in simulations](#time-handling-in-simulations), [Utilization](#utilization)

<a id="time-handling-in-simulations"></a>

### Time handling in simulations

_Concept_

Three ways the clock can move. **Next-event:** jump straight from one event to the next, because nothing changes in between (discrete event simulation). **Fixed time step:** update everything every tick (many agent-based models). **Continuous:** quantities change smoothly (system dynamics); computers approximate this with small steps, and smaller steps give a closer approximation.

<a id="four-pillars-of-simulation"></a>

### Four pillars of simulation

_Concept_

1. **Abstraction:** include only what's important, like a map that shows main roads, not every alley. (Abstraction decides **what** to include.)
2. **Simplification:** as simple as possible while still capturing the key behaviour. (Simplification decides **how much detail** for what's included.)
3. **Randomness:** model real-world uncertainty. Because each run differs, you run the simulation many times, and the **distribution of results** is the real output.
4. **Experimentation:** run it, check it makes sense, adjust, like testing a recipe.

"All models are wrong, but some are useful" (George Box).

<a id="utilization"></a>

### Utilization

_Concept_

The fraction of time a resource (a barista, a cashier, a server) is busy. It's the number that predicts trouble: **above roughly 85–90% utilization, queues grow rapidly**, and once arrivals outpace service, waits don't just increase, they explode.

<a id="discrete-event-simulation-des"></a>

### Discrete event simulation (DES)

_Concept_

Models a system as a sequence of distinct **events** over time (a customer arrives at a bank, a machine finishes a task). Between events nothing changes, so the simulation jumps straight to the next one, which makes DES efficient. The classic structure is **source → queue → server → sink**: entities are generated at the source, wait in the queue if the server is busy, are processed one at a time, and leave via the sink. Bottlenecks appear when entities arrive faster than the server can handle them. Used for queues (banks, call centres, emergency departments), manufacturing lines and logistics; the typical outputs are waiting times and queue lengths. Python library: SimPy.

See also: [Coffee shop queue simulation](#coffee-shop-queue-simulation), [SimPy](#simpy)

<a id="coffee-shop-queue-simulation"></a>

### Coffee shop queue simulation

_NumPy, Python_

A runnable version of the course's coffee shop: one barista, random arrivals (exponential gaps averaging 4 minutes), drinks taking 2–5 minutes (uniform). Each customer starts being served at whichever comes later, their arrival or the barista becoming free; the difference is their wait. The whole shift is repeated 1,000 times (Monte Carlo) to show the range of outcomes.

```python
rng = np.random.default_rng(1)

def simulate_shift(n_customers=200, mean_gap=4.0, service=(2, 5)):
    arrivals = np.cumsum(rng.exponential(mean_gap, n_customers))   # arrival times (minutes)
    service_times = rng.uniform(*service, n_customers)            # each drink takes 2-5 minutes
    barista_free_at, waits = 0.0, []
    for arrive, serve in zip(arrivals, service_times):
        start = max(arrive, barista_free_at)    # wait if the barista is busy
        waits.append(start - arrive)
        barista_free_at = start + serve
    return np.mean(waits)

results = [simulate_shift() for _ in range(1000)]       # repeat the shift 1,000 times
print(f"Average wait: {np.mean(results):.1f} min "
      f"(5th-95th percentile: {np.percentile(results, 5):.1f}-{np.percentile(results, 95):.1f})")
busy = [simulate_shift(mean_gap=3.0) for _ in range(1000)]
print(f"Busier shop: {np.mean(busy):.1f} min average wait")
# Average wait: 11.1 min (5th-95th percentile: 4.6-23.2)
# Busier shop: 57.5 min average wait
```

The surprise: drinks average 3.5 minutes. With a customer every 4 minutes the barista is about 88% busy and waits average about 11 minutes; with a customer every 3 minutes arrivals outpace service, and waits balloon to nearly an hour and keep growing through the shift. **Small changes near full capacity cause huge changes in waiting time.** Good extensions: add a second barista, or report the share of customers waiting more than 10 minutes.

See also: [Poisson arrivals and exponential gaps](#poisson-arrivals-and-exponential-gaps), [Utilization](#utilization)

<a id="poisson-arrivals-and-exponential-gaps"></a>

### Poisson arrivals and exponential gaps

_Concept, NumPy_

The standard way to model random arrivals. The **number** of arrivals per time period follows a **Poisson** distribution; the **time between** arrivals follows an **exponential** distribution. In code you usually sample the gaps and add them up to get arrival times, as in the coffee shop.

```python
gaps = rng.exponential(scale=4.0, size=200)    # average 4 minutes between customers
arrival_times = np.cumsum(gaps)
per_hour = rng.poisson(lam=15, size=8)         # arrivals in each of 8 hours
```

> **Watch out:** "Arrivals follow a Poisson distribution" is shorthand: the counts per period are Poisson, the gaps between arrivals are exponential.

<a id="np-random-default-rng"></a>

### np.random.default_rng()

_NumPy_

NumPy's modern random number generator. Create one generator (optionally with a seed for reproducible runs) and call its methods: `rng.random()` (uniform 0–1), `rng.uniform()`, `rng.normal()`, `rng.exponential()`, `rng.poisson()`, `rng.integers()` and `rng.choice()`. It replaces the older `np.random.choice`-style functions in new code.

```python
rng = np.random.default_rng(42)          # seed: same results every run
rng.normal(loc=100, scale=15, size=5)
rng.integers(1, 7, size=10)              # ten die rolls (upper bound excluded)
```

See also: [np.random.choice()](#np-random-choice), [Random seed (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#random-seed)

<a id="agent-based-modelling-abm"></a>

### Agent-based modelling (ABM)

_Concept_

A bottom-up approach that models many individual **agents**, each with its own attributes and rules (vaccinated or not, cautious or social), and lets them interact step by step. Large-scale patterns such as an epidemic curve or a traffic jam **emerge** from the local interactions; nobody programs them directly. The course example simulates influenza in a city: agents move, meet and pass on infection with set probabilities, and interventions are tested by changing the rules (raise vaccination from 40% to 70%) and comparing many runs. ABMs were widely used during COVID-19 to explore lockdowns and school closures. Other examples: traffic, retail shoppers, predator–prey systems, supply chains, warehouse robots.

See also: [Emergence](#emergence)

<a id="emergence"></a>

### Emergence

_Concept_

Complex group behaviour arising from simple individual rules: birds flocking, traffic jams forming with no accident, an outbreak spreading through a population. It's the key idea behind agent-based modelling, and the reason ABMs show effects at the individual level rather than as averages.

<a id="monte-carlo-simulation"></a>

### Monte Carlo simulation

_Concept, NumPy_

Uses **repeated random sampling** to estimate outcomes under uncertainty: if you can't calculate an answer exactly, simulate the random process many times and look at the distribution of results. More runs give a more precise estimate (the Law of Large Numbers). Named after the Monte Carlo casino and developed in the 1940s by Stanislaw Ulam and John von Neumann during the Manhattan Project. Each iteration is one possible "world"; time often isn't involved at all. Inputs are probability distributions; outputs are risk estimates, intervals and result distributions, summarized by mean, standard deviation and percentiles.

The course's flood example: heavy rain happens 20% of the time and, when it does, the river overflows 10% of the time.

```python
rng = np.random.default_rng(42)
n = 100_000                                         # simulated days
heavy_rain = rng.random(n) < 0.20                   # heavy rain 20% of the time
overflow = heavy_rain & (rng.random(n) < 0.10)      # if heavy rain, 10% chance of overflow
print(f"Estimated flood probability: {overflow.mean():.4f}")   # ≈ 0.0201 (exact: 0.02)
```

Monte Carlo is overkill for this simple case, but realistic versions (rainfall varying continuously, river levels depending on several days, upstream flows interacting) have no neat formula. It's also often combined with other approaches, such as running a DES 1,000 times with random inputs, as in the coffee shop.

See also: [Simulation (Monte Carlo)](#simulation-monte-carlo), [Law of Large Numbers (LLN)](#law-of-large-numbers-lln), [Vectorized simulation](#vectorized-simulation), [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc)

<a id="vectorized-simulation"></a>

### Vectorized simulation

_NumPy_

Running all trials at once with array operations instead of a Python loop. The flood example draws 100,000 days in one call and combines them with `&`; it's typically far faster than the loop-based simulations in the probability notebook, which matters when you need millions of runs.

```python
rolls = rng.integers(1, 7, size=1_000_000)
(rolls % 2 == 1).mean()                  # P(odd) ≈ 0.5, no loop needed
```

See also: [Vectorized operation (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#vectorized-operation)

<a id="system-dynamics-sd"></a>

### System dynamics (SD)

_Concept_

Models a system as **stocks** (accumulations) and **flows** (rates of change) connected by **feedback loops**, traditionally with differential equations. It works with aggregates ("the population", not individual people) and suits long-term, big-picture questions: population planning, policy, supply chains, ecology, climate. Developed by Jay Forrester at MIT in the 1950s. SD models can grow very complex, like the deck's causal-loop diagram of tourism, waste and pollution in the Maldives.

See also: [Stocks and flows](#stocks-and-flows), [Feedback loop](#feedback-loop)

<a id="stocks-and-flows"></a>

### Stocks and flows

_Concept, Python_

A **stock** is an amount that accumulates (a bathtub's water, a population, inventory); a **flow** is a rate that changes it (the tap and the drain; births and deaths). Each step, the stock changes by inflow minus outflow. The course example is a town of 1,000 people with a 3% birth rate and 1% death rate:

```python
population, birth_rate, death_rate = 1000.0, 0.03, 0.01
history = [population]
for year in range(1, 51):
    births = birth_rate * population        # inflow
    deaths = death_rate * population        # outflow
    population += births - deaths           # stock changes by inflow - outflow
    history.append(population)
round(history[10]), round(history[35]), round(history[50])   # (1219, 2000, 2692)
```

Net growth is 2% a year, so this is exponential growth, P(t) = 1,000 × 1.02ᵗ: about 1,219 after 10 years, doubling in about 35 years (the **rule of 70**: 70 ÷ 2% ≈ 35).

<a id="feedback-loop"></a>

### Feedback loop

_Concept_

A cycle where a quantity affects its own future values. A **reinforcing** (positive) loop amplifies change: more people → more births → faster growth. A **balancing** (negative) loop counteracts it: more people → more deaths. In the population example, with these constant rates the reinforcing loop wins, so growth is exponential.

> **Watch out:** It's easy to misread the loops: the **deaths outflow is itself the balancing loop** and births are the reinforcing loop. With constant rates of 3% and 1% the population doesn't stabilize; it grows by 2% a year.

<a id="choosing-a-simulation-approach"></a>

### Choosing a simulation approach

_Concept_

| Approach             | Time                 | Logic                         | Typical uses                               |
| -------------------- | -------------------- | ----------------------------- | ------------------------------------------ |
| Discrete event (DES) | Jumps between events | Rule-based process flows      | Queues, logistics, healthcare              |
| Agent-based (ABM)    | Steps                | Autonomous interacting agents | Epidemics, social behaviour, markets       |
| System dynamics (SD) | Continuous           | Stocks, flows, feedback       | Long-term trends, policy, ecology          |
| Monte Carlo          | Iterations           | Random sampling of inputs     | Risk, uncertainty, cost and time estimates |

Choose by the research question, how individuals differ, the level of detail needed, the data available and the computing resources. They combine in practice: a hospital DES run 1,000 times with random inputs is also a Monte Carlo study. A good quick check: to estimate the risk a project runs over budget, use Monte Carlo.

<a id="simpy"></a>

### SimPy

_Python_

A Python library for discrete event simulation: you describe processes (customers arriving, waiting for a resource, being served) and SimPy manages the clock and the queues. Usually needs installing (`pip install simpy`).

```python
import simpy

def customer(env, barista, rng):
    with barista.request() as req:            # join the queue
        yield req                              # wait for the barista
        yield env.timeout(rng.uniform(2, 5))   # make the drink

def arrivals(env, barista, rng):
    while True:
        yield env.timeout(rng.exponential(4))  # next customer arrives
        env.process(customer(env, barista, rng))

env = simpy.Environment()
barista = simpy.Resource(env, capacity=1)
env.process(arrivals(env, barista, np.random.default_rng(1)))
env.run(until=480)                             # an 8-hour shift
```

The other libraries in the deck's toolkit are NumPy (random sampling, vectorized maths), SciPy (statistical distributions, optimization), pandas (analysing outputs) and Matplotlib (visualizing results).

<a id="limitations-of-simulation"></a>

### Limitations of simulation

_Concept_

Four from the slides: the **simplification vs accuracy** trade-off (too simple misses nuance, too complex is impractical); **computational constraints** (high-fidelity models need significant power and time); **expertise** (you need both subject knowledge and modelling skill); and **data quality** (limited or biased data forces assumptions). **Garbage in, garbage out:** the coffee shop result depends entirely on the assumed arrival and service times. And even with perfect data there's **structural uncertainty**: the model's choice of rules and feedbacks may itself be wrong.

<a id="verification-and-validation"></a>

### Verification and validation

_Concept_

**Verification:** "did we build the model right?" (the code does what we intended; no bugs). **Validation:** "did we build the right model?" (it matches reality). The deck's three validation steps: **real-world testing** against what actually happened and against benchmarks; **sensitivity and uncertainty analysis**; and **peer review** by experts to catch problems and build trust.

See also: [Sensitivity analysis](#sensitivity-analysis)

<a id="sensitivity-analysis"></a>

### Sensitivity analysis

_Concept_

Changing one input at a time and watching how much the output moves. Inputs with a big effect deserve the most careful estimates; if results barely change across plausible values, the conclusion is robust. Changing the coffee shop's arrival gap from 4 to 3 minutes (and seeing the average wait jump from 11 to 57 minutes) is a sensitivity analysis; so is re-running the ROI with only 75% of the savings realised.

See also: [Risk-adjusted ROI](#risk-adjusted-roi)

<a id="s28-decision-trees"></a>

## 28. Decision trees

_A flowchart learned from data. How the CART algorithm chooses splits using Gini impurity, how to stop a tree from memorizing the training data, and how to read and visualize one. Examples use the cleaned Titanic data (`titanic_clean.csv`)._

**In your capstone:** A single tree is a transparent baseline and a good way to explain rules to stakeholders. Control its depth to avoid overfitting.

<a id="decision-tree"></a>

### Decision tree

_Concept_

A model shaped like a flowchart that ends in a decision; each split represents a condition being true or false ("age < 34?", "has degree?"). For linear models an algorithm learns β; for trees it learns **where the splits occur**. Works for classification and regression, and is white-box.

The learned "parameters" of a tree are which feature to split on at each node, the threshold, and the prediction at each leaf. A 0/1 feature is split as `has_degree < 0.5`. The same prediction can appear at several leaves via different paths. Trees will happily split on irrelevant features (the slides' left-handedness example) if they happen to fit the training data: a route to overfitting.

> **Watch out:** Don't confuse the two kinds of tree: one that predicts a category (Candidate A or B) is a **classification** tree; a regression tree predicts a number at each leaf.

<a id="root-node-internal-node-leaf-and-branch"></a>

### Root node, internal node, leaf and branch

_Concept_

The **root node** is the first split at the top. **Internal nodes** are later splits. **Leaves** (terminal nodes) hold the final predictions. **Branches** are the paths between them. **Depth** is the number of split levels from root to the deepest leaf.

<a id="cart-algorithm"></a>

### CART algorithm

_Concept_

**C**lassification **A**nd **R**egression **T**rees, the standard way to fit a tree: (1) for each variable, find the best split value according to a splitting criterion; (2) split the data at the best split overall; (3) repeat on each new subset until a stopping criterion is met; (4) optionally prune (not done by default).

CART (Breiman, Friedman, Olshen and Stone, 1984) builds **binary** trees: every split has exactly two branches. In step 1, the algorithm tries every feature and every possible threshold between its sorted values, scores each candidate split, and keeps the best. On the breast cancer data an unrestricted tree reaches depth 7 with 18 leaves and **100% training accuracy**, but 92.3% on test data.

<a id="greedy-algorithm"></a>

### Greedy algorithm

_Concept_

An algorithm that makes the best choice at each step without looking ahead. CART is greedy: it picks the best split now even if a worse split now would allow a much better tree later. Usually fine, but not guaranteed optimal.

<a id="splitting-criterion"></a>

### Splitting criterion

_Concept_

The loss a split is chosen to reduce. For **regression trees**: the MSE. For **classification trees**: Gini impurity by default (entropy is an alternative). The tree tries every possible split and keeps the one that reduces the loss most.

**Why not accuracy?** A useful split often leaves both children predicting the majority class, so accuracy doesn't change and can't tell good splits from useless ones. Gini can. In the phone example, the age split predicts Android on both sides, yet still slightly lowers Gini. Before any split, the node predicts the majority class (Android): 15/25 = 60% accuracy, the null model. For regression trees, `criterion="squared_error"`, and each leaf predicts the mean of its points.

<a id="gini-impurity"></a>

### Gini impurity

_Concept_

How mixed the classes in a node are: **Gini = 1 − Σ pᵢ²**, where pᵢ is the proportion of class i. **0**: pure node (one class). **0.5**: maximum for two classes, a 50/50 split. With 10 iPhone and 15 Android buyers: 1 − (0.4² + 0.6²) = 0.48. Lower Gini means a purer node.

```python
def gini(counts):
    p = np.array(counts) / sum(counts)
    return 1 - np.sum(p ** 2)

gini([10, 15])          # 0.48
```

Intuition: Gini is the probability that two points drawn at random (with replacement) from the node belong to **different** classes. With k equally common classes the maximum is **1 − 1/k**: 0.5 for two classes, 0.67 for three.

Another worked example: a node with 8 rainy and 2 dry days has Gini 1 − (0.8² + 0.2²) = **0.32**. scikit-learn's feature importance is built from these impurity reductions.

<a id="evaluating-a-split-with-gini"></a>

### Evaluating a split with Gini

_Concept_

Compute the Gini of each child node, then take the average **weighted by the number of observations** in each. The split with the lowest weighted Gini wins. Splitting the phone buyers by gender (males 2 iPhone/12 Android, females 8/3) beats splitting by age (≤30: 4/8, 31+: 6/7), so the tree splits on gender first.

```python
w_gender = 14/25 * gini([2, 12]) + 11/25 * gini([8, 3])   # ≈ 0.31
w_age = 12/25 * gini([4, 8]) + 13/25 * gini([6, 7])       # ≈ 0.47
```

<a id="entropy-and-cross-entropy"></a>

### Entropy and cross-entropy

_Concept_

An alternative impurity measure, −Σ pᵢ log pᵢ (`criterion="entropy"`). In practice Gini and entropy almost always choose the same or very similar splits; Gini is scikit-learn's default mainly because it's slightly cheaper to compute (no logarithms).

> **Watch out:** You'll sometimes see Gini's "differentiability" given as its advantage. Trees aren't fitted with gradients, so that doesn't matter here; the choice between Gini and entropy rarely changes the tree.

<a id="stopping-criteria"></a>

### Stopping criteria

_Concept, scikit-learn_

By default a tree keeps splitting until every leaf is pure, which is the perfectly overfit model. Any stopping rule acts as regularization, and each is a tuning parameter:

- **`max_depth`**: levels allowed.
- **`min_samples_split`**: minimum rows a node needs before it may split (default 2).
- **`min_samples_leaf`**: minimum rows each resulting leaf must have (default 1: alarm bells).
- **`min_impurity_decrease`**: the minimum reduction in impurity a split must achieve.
- `max_leaf_nodes` and `ccp_alpha` (pruning).

`min_samples_split` and `min_samples_leaf` differ: with `min_samples_split=10`, a node of 12 can still split into 11 and 1; with `min_samples_leaf=5` that split isn't allowed. `min_samples_leaf` directly prevents tiny, noisy leaves, so tune both.

```python
params = {"max_depth": [2, 3, 4, 5, None], "min_samples_split": [2, 10, 20],
          "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5).fit(X_tr, y_tr)
gs.best_params_, gs.best_score_          # breast cancer: min_samples_split=10, CV 0.934
```

<a id="ccp-alpha-cost-complexity-pruning"></a>

### ccp_alpha (cost-complexity pruning)

_scikit-learn_

A complexity penalty like α in regularization: grow the tree, then prune back branches that don't improve the fit enough to justify their complexity. Larger `ccp_alpha` = smaller tree. Default 0 (no pruning).

<a id="decisiontreeclassifier-and-decisiontreeregressor"></a>

### DecisionTreeClassifier and DecisionTreeRegressor

_scikit-learn_

scikit-learn's tree models. No scaling or polynomial terms needed.

```python
from sklearn.tree import DecisionTreeClassifier

X = pd.get_dummies(titanic.drop(columns=["PassengerId", "Name", "Survived"]),
                   columns=["Pclass", "Sex", "Embarked"], drop_first=True)
y = titanic["Survived"]
tree = DecisionTreeClassifier(max_depth=4, min_samples_split=5)
tree.fit(X, y)
pd.DataFrame({"feature": X.columns, "importance": tree.feature_importances_}) \
  .sort_values("importance", ascending=False)
```

<a id="tuning-tree-depth"></a>

### Tuning tree depth

_scikit-learn_

Loop over depths and compare scores. **Training** accuracy always rises with depth, so make the decision using test or cross-validated scores.

```python
depths = range(1, 30)
cv_scores = [cross_val_score(DecisionTreeClassifier(max_depth=d), X, y, cv=5).mean()
             for d in depths]
plt.plot(depths, cv_scores)
```

> **Watch out:** Scoring each depth on the **training** data only shows overfitting increasing; always compare depths on cross-validated or test scores.

<a id="export-text"></a>

### export_text()

_scikit-learn_

Prints a fitted tree as indented plain-text rules, handy when plotting isn't practical.

```python
from sklearn.tree import export_text
print(export_text(tree, feature_names=list(X.columns)))
```

<a id="partitioning-the-feature-space"></a>

### Partitioning the feature space

_Concept_

Another way to picture a tree: each split draws an axis-parallel line, dividing the feature space into rectangles, each with one prediction. For regression it looks like a 3D staircase. This is why trees can't draw smooth diagonal boundaries.

Each split draws a line parallel to an axis ("age < 34" is a vertical line). A consequence of the staircase shape: trees produce **step-shaped** predictions and **can't extrapolate** beyond the range of their training data.

<a id="decision-tree-pros-and-cons"></a>

### Decision tree pros and cons

_Concept_

**Pros:** white-box (easy to visualize and explain), non-parametric, scale of features doesn't matter (no scaling needed: trees only compare values to thresholds), fast, little feature engineering (interactions are built in), handle categorical features naturally. **Cons:** overfit easily without stopping rules; **unstable** (a small change in the data can produce a very different tree); step-shaped predictions that can't extrapolate; biased towards the majority class with imbalanced data (use `class_weight="balanced"`); greedy. Random forests and gradient boosting fix most of the cons, at some cost to interpretability.

---

<a id="s29-ensembles-bagging-and-random-forests"></a>

## 29. Ensembles: bagging and random forests

_Combining many models into one stronger one. Bootstrapping, bagging decision trees to cut their variance, estimating error with out-of-bag samples, and random forests' extra trick of decorrelating the trees. Examples use the ISLR heart disease data (`Heart.csv`)._

**In your capstone:** Random forests are a strong, low-effort default for tabular data. Use cross-validated or OOB scores and permutation importance.

<a id="ensemble-model"></a>

### Ensemble model

_Concept_

A model that combines the predictions of many (often weaker) models into one. It's an actual statistical fact that aggregating many "ok" models beats relying on one. Ensembling mainly reduces variance, which makes overfit decision trees a prime candidate. Three main methods: **bagging**, **boosting** and **stacking**.

Single trees are interpretable but high-variance; linear models are stable but often too simple (high bias). Ensembles trade some interpretability for much better accuracy. It works when the individual models make **different** mistakes that partly cancel out.

<a id="wisdom-of-the-crowd"></a>

### Wisdom of the crowd

_Concept_

"Individuals are dumb, but crowds are smart." In NPR Planet Money's 2015 experiment, over 17,000 people guessed Penelope the cow's weight online, and the **average** guess was within a few percent of the truth, closer than most individual guesses. Averaging cancels individual errors, and the spread of an average shrinks as more guesses are added.

The condition that matters: the crowd must make **fairly independent** errors. If everyone copies one person, averaging doesn't help. Correlated trees are bagging's main weakness, and the reason random forests exist.

> **Watch out:** It's tempting to think averaging improves any set of models. Averaging mainly reduces **variance**; averaging high-bias models leaves most of their bias in place. Reducing bias is boosting's job.

<a id="bootstrap-sample"></a>

### Bootstrap sample

_Concept, NumPy_

A random sample **with replacement**, the same size as the original data. Some rows appear several times and some (about 37%) not at all.

```python
idx = np.random.choice(len(df), size=len(df), replace=True)
boot = df.iloc[idx]
```

On average a bootstrap sample contains about 63% of the distinct rows; about **36.8% (1/e)** are left out. Those left-out rows are the **out-of-bag** observations. In the deck's 9-row example, rows 1, 4, 6 and 9 weren't picked (4 of 9).

```python
rng = np.random.default_rng(0)
idx = rng.choice(n, size=n, replace=True)
1 - len(np.unique(idx)) / n          # ≈ 0.368 for large n
```

<a id="bootstrapping"></a>

### Bootstrapping

_Concept, NumPy_

A resampling method that estimates the sampling distribution of a statistic (a mean, median, coefficient…) by recomputing it on many bootstrap samples. The notebook estimates the distribution of the median cholesterol level.

```python
chol = df["Chol"]
boot_medians = [np.median(np.random.choice(chol, size=len(chol), replace=True))
                for _ in range(500)]
plt.hist(boot_medians, bins=20)
```

<a id="bagging-bootstrap-aggregating"></a>

### Bagging (bootstrap aggregating)

_Concept_

Draw a bootstrap sample, fit a model (usually a deep, deliberately overfit decision tree: low bias, high variance) to it, and repeat B times (often 100–500). Predict with the **mean** of the B predictions (regression) or the **majority vote** (classification). Averaging unbiased, high-variance models gives an unbiased model with lower variance. More trees never hurt accuracy, only training time, and because the models are independent they can be trained in parallel (`n_jobs=-1`). scikit-learn's classifiers actually average the predicted **probabilities** across trees ("soft voting"), which usually matches the majority vote.

Why correlation limits it: if B trees each have variance σ² and pairwise correlation ρ, their average has variance **ρσ² + (1 − ρ)σ²/B**. More trees shrink the second term, but the first term never goes away. Random forests attack ρ.

On the Heart data: a single tree memorizes the training data (100%) but scores 0.720 on test; 500 bagged trees score 0.813 (CV 0.766). Same model type, much less variance.

<a id="baggingclassifier"></a>

### BaggingClassifier

_scikit-learn_

scikit-learn's bagging ensemble. `estimator` is the base model and `n_estimators` the number of bootstrap models. It can be grid-searched, including the base tree's own settings.

```python
from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100,
                        random_state=2022)
bag.fit(X_train, y_train)
bag.score(X_test, y_test)

pgrid = {"estimator__max_depth": [1, 2, 3, 4, None],
         "estimator__min_samples_leaf": [1, 3, 5, 7]}
gs = GridSearchCV(bag, pgrid, cv=5, n_jobs=4)
```

> **Watch out:** Older code uses `base_estimator=`; scikit-learn 1.2 renamed it `estimator` and 1.4 removed the old name, so check the documentation if older examples fail.

<a id="out-of-bag-oob-observations-and-error"></a>

### Out-of-bag (OOB) observations and error

_Concept, scikit-learn_

The rows a tree didn't see in its bootstrap sample are **out-of-bag** for that tree. Predict each row using only the trees for which it was out-of-bag, then compare with the truth: the OOB score is an estimate of out-of-sample performance without a train-test split or cross-validation.

```python
bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4),
                        n_estimators=100, oob_score=True, random_state=2022)
bag.fit(X, y)
bag.oob_score_
```

`oob_score_` is accuracy for classifiers and R² for regressors. Set `oob_score=True` **before** fitting (it requires `bootstrap=True`, the default). On the Heart data, the random forest's OOB accuracy (0.788) almost matches its 5-fold CV accuracy (0.784), at no extra cost; the test score (0.840) differs because the test set has only 75 patients.

<a id="feature-importance-in-ensembles"></a>

### Feature importance in ensembles

_Concept, NumPy_

Bagging gains accuracy but loses the single readable tree. You can still summarize importance: the total decrease in Gini (classification) or MSE (regression) from splits on each feature, averaged over all the trees. Random forests provide it directly as `feature_importances_`.

```python
importances = np.mean([t.feature_importances_ for t in bag.estimators_], axis=0)
```

On the Heart random forest the top five are fairly even: Oldpeak 0.113, MaxHR 0.107, Ca 0.106, Age 0.101, Thal_normal 0.097.

> **Watch out:** Forest importances are more stable than a single tree's but still **biased** towards continuous and high-cardinality features, and computed on training data. `permutation_importance` on test data is more reliable.

See also: [Permutation importance](#permutation-importance)

<a id="random-forest"></a>

### Random forest

_Concept_

Bagged decision trees plus one more source of randomness: **at each split, only a fresh random subset of m features is considered**. If one feature is very strong, plain bagging would put it at the top of every tree, so the trees would be near-copies; random forests **decorrelate** the trees so that averaging reduces variance more. A side effect: weaker features get a chance to be used, so the forest explores more patterns. When m = p (all features) it's ordinary bagging. Leo Breiman introduced the method in 2001.

> **Watch out:** It's easy to describe random forests as giving each tree its own feature subset. It's a new random subset at **each split**; Choosing one subset per tree is a different technique (the "random subspace" method).

<a id="max-features"></a>

### max_features

_scikit-learn_

How many features each split may consider (m). The original recommendations are **√p for classification** and **p/3 for regression**; _The Elements of Statistical Learning_ suggests treating it as a tuning parameter. With 16 features, √16 = 4 per split.

> **Watch out:** scikit-learn's defaults don't all follow the recommendation. `RandomForestClassifier` uses `max_features="sqrt"`, but `RandomForestRegressor` defaults to `max_features=1.0` (all features), which makes it **plain bagging** unless you change it. The default number of trees is 100 (it was 10 before scikit-learn 0.22); 500 is a common choice.

<a id="randomforestclassifier-and-randomforestregressor"></a>

### RandomForestClassifier and RandomForestRegressor

_scikit-learn_

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150, max_features=4,
                            oob_score=True, random_state=1)
rf.fit(X, y)
rf.oob_score_
pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}) \
  .sort_values("importance", ascending=False)
cross_val_score(rf, X, y, cv=10).mean()
```

The course comparison on the Heart data (297 patients, null accuracy 0.533):

| Model                     | Train | CV    | OOB   | Test  |
| ------------------------- | ----- | ----- | ----- | ----- |
| Single tree               | 1.000 | 0.730 |       | 0.720 |
| 500 bagged trees          | 1.000 | 0.766 | 0.752 | 0.813 |
| Random forest (500 trees) | 1.000 | 0.784 | 0.788 | 0.840 |

Decorrelating the trees helped on every measure. Tune `max_features`, `min_samples_leaf` and `n_estimators` with `GridSearchCV`; `max_features=16` (all features) reproduces bagging.

<a id="random-forest-pros-and-cons"></a>

### Random forest pros and cons

_Concept_

**Pros:** performance competitive with the best supervised methods; little tuning needed; no scaling; handles non-linearity and interactions; robust to outliers; OOB error for free (though you still need a test set when tuning); feature importances that are more stable than a single tree's. **Cons:** less interpretable (black box), slower to train and to predict. One-line summary: bagging averages away variance; random forests make the trees different enough for averaging to work even better.

<a id="parallelizable"></a>

### Parallelizable

_Concept_

The trees in bagging and random forests are independent, so they can be trained simultaneously on several cores or machines (`n_jobs=-1`). Boosting can't do this because each model depends on the last.

<a id="stacking"></a>

### Stacking

_Concept_

Ensembling models of **any** kind together (linear regression, Lasso, a tree, a neural net…), then combining their predictions, either by voting/averaging or with a final model that learns how to weight them (`StackingClassifier`).

In scikit-learn: `StackingClassifier` / `StackingRegressor`. Common in competitions; less common in production because of the added complexity.

<a id="votingclassifier"></a>

### VotingClassifier

_scikit-learn_

Combines different classifiers by majority vote (`voting="hard"`, default) or by averaging probabilities (`voting="soft"`).

```python
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier([
    ("lr", LogisticRegression(max_iter=1000)),
    ("knn", KNeighborsClassifier(n_neighbors=7)),
    ("tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=7)),
])
vc.fit(X_train, y_train)
vc.score(X_test, y_test)
```

<a id="which-model-is-best"></a>

### Which model is best?

_Concept_

It depends on the task. In many business cases interpretability matters more than accuracy, so a decision tree or logistic regression may be preferred over an ensemble. Every model is a trade-off between bias and variance: ensembles reduce overfitting by lowering variance, at the cost of some bias and most of the interpretability.

---

<a id="s30-ensembles-boosting"></a>

## 30. Ensembles: boosting

_The opposite strategy to bagging: chain weak, high-bias models together so each one fixes the previous one's mistakes. AdaBoost (built from scratch and in scikit-learn), gradient boosting and XGBoost. Examples use the spam detection data (`spam.csv`) and `make_moons`._

**In your capstone:** Gradient boosting often gives the best accuracy on tabular data. Tune the learning rate, number of trees and depth with cross-validation, and compare it with a random forest.

<a id="boosting"></a>

### Boosting

_Concept_

An ensemble method that fits models **sequentially**, each one paying more attention to the parts of the data the previous ones got wrong. It takes **weak learners** and successively boosts them to **reduce bias**, just as a tutor focuses on the fractions a friend keeps getting wrong rather than the multiplication they already know.

Bagging starts from low-bias, high-variance models and lowers the variance; boosting starts from high-bias, low-variance models and lowers the bias. Both aim for the sweet spot of the bias–variance curve from opposite ends. Boosting can overshoot into **overfitting** if run too long (too many trees, too high a learning rate). Gradient-boosted trees are often the most accurate models for tabular data.

<a id="weak-learner"></a>

### Weak learner

_Concept_

A model only a little better than random guessing: underfit, high-bias. Trees score well on almost everything practical (mixed feature types, missing values, outliers, irrelevant inputs, speed) except predictive power, and boosting fixes exactly that weakness. **AdaBoost** typically uses one-split **stumps** (scikit-learn's default); **gradient boosting** usually uses small trees (scikit-learn default `max_depth=3`, XGBoost default 6), which can capture interactions between features.

<a id="decision-stump"></a>

### Decision stump

_Concept, scikit-learn_

A decision tree with **a single split** (`max_depth=1`). The classic weak learner for AdaBoost. On the spam data, one stump alone is a weak classifier; a hundred boosted stumps are a strong one.

```python
stump = DecisionTreeClassifier(max_depth=1)
```

<a id="bagging-vs-boosting"></a>

### Bagging vs boosting

_Concept_

|                              | Bagging / random forest | Boosting                                             |
| ---------------------------- | ----------------------- | ---------------------------------------------------- |
| Learners                     | Independent (parallel)  | Sequential                                           |
| Goal                         | Reduce **variance**     | Reduce **bias**                                      |
| Typical learner              | Deep, overfit trees     | Stumps (AdaBoost) or small trees (gradient boosting) |
| Overfitting with more models | No (only slower)        | Yes, if run too long                                 |

One-line summary: bagging averages independent overfit trees; boosting adds sequential underfit trees, each fixing the last one's errors.

<a id="adaboost"></a>

### AdaBoost

_Concept_

**Ada**ptive **Boost**ing. Start with equal weights on every observation. Repeat: fit a weak learner on the weighted data; compute its weighted error rate εₘ; give the learner a say in the final vote of **αₘ = ½ ln((1 − εₘ)/εₘ)**; then **increase the weights of the misclassified points** and decrease the rest. The final prediction is the sign of the weighted vote: sign(Σ αₘ Gₘ(x)). For the maths, y is coded as −1/1 instead of 0/1.

AdaBoost was introduced by Freund and Schapire (1997). Reading αₘ: if err = 0.5 (guessing), α = 0 (no say); if err = 0.1, α = ½ ln 9 ≈ 1.1, and misclassified points end up weighted 9 times as heavily as correct ones relative to before. The AdaBoost.M1 version (as in _The Elements of Statistical Learning_) writes α = ln((1 − err)/err) (≈ 2.2 here) and multiplies only the misclassified weights by e^α = 9; the two forms give the same relative weights. Each stump is one vertical or horizontal line, but their weighted combination draws a complex boundary.

> **Watch out:** This α is unrelated to the regularization α or the gradient descent step size; scikit-learn's `learning_rate` scales it. AdaBoost is sensitive to **noisy labels**: it keeps up-weighting points it can't fit.

<a id="observation-weights"></a>

### Observation weights

_Concept_

In AdaBoost, a probability distribution Dₜ over the rows that says how much each one matters to the next learner. After each round: Dₜ₊₁(i) = Dₜ(i) · e^(−αₜ yᵢ hₜ(xᵢ)), then normalize to sum to 1. Correctly classified rows (yᵢhₜ(xᵢ) = +1) shrink; misclassified rows (−1) grow.

<a id="adaboost-from-scratch"></a>

### AdaBoost from scratch

_NumPy, scikit-learn_

The code-along builds AdaBoost from four small functions: a weighted vote, the learner weight, the weight update and weighted resampling.

```python
def calculate_alpha(y_true, predictions):
    error_rate = 1. - np.mean(y_true == predictions)
    return 0.5 * np.log((1. - error_rate) / error_rate)

def update_importance_weights(X, y_true, D, alpha, estimator):
    y_signed = np.where(y_true == 1, 1, -1)
    yhat_signed = np.where(estimator.predict(X) == 1, 1, -1)
    D = D * np.exp(-alpha * y_signed * yhat_signed)
    return D / D.sum()

def sample_with_weights(X, y, D):
    idx = np.random.choice(np.arange(len(y)), size=len(y), replace=True, p=D)
    return X[idx], y[idx]

D = np.ones(len(y)) / len(y)
estimators, alphas = [], []
for _ in range(10):
    Xb, yb = sample_with_weights(X, y, D)
    tree = DecisionTreeClassifier(max_depth=2).fit(Xb, yb)
    alpha = calculate_alpha(y, tree.predict(X))
    D = update_importance_weights(X, y, D, alpha, tree)
    estimators.append(tree); alphas.append(alpha)
```

<a id="signum-and-indicator-functions"></a>

### Signum and indicator functions

_Concept, NumPy_

Notation used in boosting and SVM maths. **sign(x)** is 1 if x is positive and −1 if negative (`np.sign`). The **indicator** I(A) is 1 if A is true and 0 otherwise, so the mean of an indicator is a proportion: `np.mean(y == y_hat)` is accuracy.

<a id="adaboostclassifier"></a>

### AdaBoostClassifier

_scikit-learn_

scikit-learn's AdaBoost (it implements the SAMME variant, very similar to the AdaBoost.M1 in the slides). Key hyperparameters: `estimator` (default a stump), `n_estimators` (number of rounds) and `learning_rate` (shrinks each learner's contribution).

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                         n_estimators=100, learning_rate=1)
ada.fit(X_train, y_train)
ada.score(X_train, y_train), ada.score(X_test, y_test)
```

On the Spambase data (4,601 emails, 57 word- and character-frequency features, 39.4% spam), 200 default stumps reach training 0.948 and test **0.933**: a small train–test gap, because boosting stumps doesn't overfit quickly.

<a id="learning-rate-and-n-estimators"></a>

### learning_rate and n_estimators

_scikit-learn_

`n_estimators` is the number of boosting rounds; more rounds lower bias but can eventually overfit. `learning_rate` scales how much each new model contributes; smaller values learn more slowly and usually need more rounds. They trade off against each other; "if you train for long enough, the learning rate won't matter" much.

<a id="gradient-boosting"></a>

### Gradient boosting

_Concept_

Instead of reweighting observations, each new tree is fitted to the **residuals** of the current ensemble (all of them; big residuals simply have more influence), and a small step of that tree is added on. Precisely, each tree fits the **negative gradient** of the loss (the "pseudo-residuals"); for squared error those are exactly y − ŷ, and for classification the loss ("deviance") is log loss. Each update is multiplied by a small **learning rate** (shrinkage, e.g. 0.1), so many small steps are taken, which reduces overfitting. It's gradient descent, but each step adds a tree instead of changing a coefficient. Introduced by Jerome Friedman (2001); unlike AdaBoost, you can choose the loss.

Analogy: golf. The first shot gets you near the hole; each following putt corrects the remaining distance. Each tree is a putt.

> **Watch out:** Each tree fits the **negative** gradient, the direction that reduces the loss; it's easy to drop the minus sign.

<a id="gradientboostingclassifier"></a>

### GradientBoostingClassifier

_scikit-learn_

scikit-learn's gradient boosting (`GradientBoostingRegressor` for regression). Defaults: 100 trees of depth 3, learning rate 0.1. Strong out of the box.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=222)
pgrid = {"learning_rate": [0.1, 1], "n_estimators": [10, 100], "max_depth": [1, 2, 3]}
gs = GridSearchCV(gb, pgrid, cv=5, n_jobs=6).fit(X_train, y_train)
gs.best_params_
```

Key hyperparameters: `n_estimators`, `learning_rate`, `max_depth`, and `subsample` (below 1.0 gives stochastic gradient boosting). Smaller learning rates need more trees but usually generalize better. On the spam data, 100 depth-3 trees (the defaults) score 0.968 train / **0.946** test, beating 200 stumps: slightly bigger trees capture interactions between words. scikit-learn's `HistGradientBoostingClassifier` is a much faster version that handles missing values natively.

<a id="xgboost"></a>

### XGBoost

_Concept, xgboost_

e**X**treme **G**radient **Boost**ing (Chen and Guestrin, 2016): gradient boosting with built-in **regularization** (penalties on tree complexity and leaf weights), second-order (Newton-style) optimization, efficient handling of sparse and missing data, and speed. It isn't part of scikit-learn but provides a scikit-learn-compatible API, so it works with pipelines and `GridSearchCV`. Alternatives with similar performance: LightGBM, CatBoost and scikit-learn's `HistGradientBoostingClassifier`.

```python
# pip install xgboost
from xgboost import XGBClassifier

X.columns = X.columns.str.replace("[", "lbracket", regex=False)   # see Watch out
xgb = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)                                          # spam: 0.952
pd.Series(xgb.feature_importances_, index=X.columns).nlargest(3)
# char_freq_$ 0.242, word_freq_remove 0.115, char_freq_! 0.099
```

**Controlling overfitting:** lower `learning_rate` (eta), smaller `max_depth`, `subsample` and `colsample_bytree` below 1, regularization (`reg_lambda`, `reg_alpha`), and early stopping on a validation set.

> **Watch out:** XGBoost rejects feature names containing `[`, `]` or `<` (the spam data's `char_freq_[` raises an error), so rename columns first. It's efficient **for a boosting library** and scales to massive data, but it's still slower than a linear model or a single tree, especially when heavily tuned.

<a id="catboost"></a>

### CatBoost

_Concept_

Another gradient boosting library, suggested as a next step. Its strength is native support for categorical features, so there's no need to one-hot encode them.

<a id="boosting-pros-and-cons"></a>

### Boosting pros and cons

_Concept_

**Pros:** top-tier performance and all the benefits of tree-based models. **Cons:** harder to tune (many hyperparameters), slower to train, can overfit if run too long, sensitive to noisy labels, and the highest computational cost in the course. Interpretation isn't totally gone: feature importance, partial dependence plots and SHAP all work on boosted trees.

> **Watch out:** "Can boosting be parallelized?" has a two-part answer: boosting **rounds** must run in sequence, but XGBoost, LightGBM and `HistGradientBoosting` parallelize the work **within** each tree.

See also: [Model explainability](#model-explainability)

<a id="comparing-adaboost-gradient-boosting-and-xgboost"></a>

### Comparing AdaBoost, gradient boosting and XGBoost

_Concept_

**AdaBoost** reweights the observations the previous model got wrong. **Gradient boosting** fits each new model to the previous model's residuals. **XGBoost** is gradient boosting optimized for large data with missing values.

Spam test accuracy (null 0.606): AdaBoost 0.933; gradient boosting 0.946; XGBoost 0.952; **random forest 0.953**. Boosting isn't always the winner: here a random forest ties it. Always compare against simpler baselines. Why not always boost? Simpler models are often good enough, and easier to tune, explain and maintain.

<a id="make-moons"></a>

### make_moons()

_scikit-learn_

Generates two interleaving half-circles: a toy dataset that a straight line can't separate, used to visualize boosting rounds.

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.5, random_state=42)
```

---

<a id="s31-support-vector-machines"></a>

## 31. Support vector machines

_Classifiers that find the widest possible gap between classes, then bend that gap with kernels. Maximal margin, soft margin and kernel SVMs, and the hyperparameters C and kernel. Examples use the wine data (red vs white) and the MNIST handwritten digits._

**In your capstone:** Use SVMs when accuracy matters more than interpretability on scaled, medium-sized data.

<a id="support-vector-machine-svm"></a>

### Support vector machine (SVM)

_Concept_

A classification model (support vector regressors exist but are rarely used) that separates classes with the boundary that leaves the **widest margin** around it. Part of a wider family called **discriminant models**. Exceptional performance and works in high dimensions, but a black box, and data must be scaled.

<a id="separating-hyperplane"></a>

### Separating hyperplane

_Concept_

The boundary an SVM draws between classes: a line in 2D, a plane in 3D, a hyperplane in higher dimensions. Also called the **linear discriminant** or **decision boundary**. It is **not** the support vector.

<a id="maximal-margin-classifier"></a>

### Maximal margin classifier

_Concept_

SVM part I. Of all the lines that separate two perfectly separable classes, choose the one with the **maximal margin**: the widest empty space around it. Only possible when the classes can be separated perfectly, which never happens in real life.

<a id="margin"></a>

### Margin

_Concept_

The gap between the separating hyperplane and the nearest points on each side. Wider margins tend to generalize better.

<a id="support-vectors"></a>

### Support vectors

_Concept_

The points that lie exactly on the margin (or, in a soft-margin SVM, on the wrong side of it). They alone determine where the boundary sits. Important for how SVMs are fitted, but "not too relevant for us".

<a id="soft-margin-classifier"></a>

### Soft margin classifier

_Concept_

SVM part II. Real classes overlap, so allow some points to sit inside the margin or on the wrong side, measuring how far each one strays with a **slack variable** ξᵢ. The optimization balances a wide margin against the total slack.

<a id="slack-variable"></a>

### Slack variable

_Concept_

ξᵢ, the distance from a "wrong" point to its correct margin. Zero for points on the right side of their margin.

<a id="c-svm"></a>

### C (SVM)

_scikit-learn_

The SVM's regularization hyperparameter, controlling how much slack is tolerated. **Small C**: heavy regularization, a wider margin, more misclassified training points: more bias, less variance. **Large C**: little regularization, a near-perfect fit to the training data: less bias, more variance.

```python
pgrid = {"C": np.linspace(0.0001, 2, 10)}
gcv = GridSearchCV(LinearSVC(max_iter=20000), pgrid, cv=cv).fit(X_sc, y)
gcv.best_params_
```

See also: [C (inverse regularization strength)](#c-inverse-regularization-strength)

<a id="optimization-problem-objective-and-constraints"></a>

### Optimization problem: objective and constraints

_Concept_

Almost every ML model is fitted by optimizing a loss function. An optimization problem has an **objective function** to minimize or maximize and, optionally, **constraints** it must satisfy (unconstrained if there are none). OLS is unconstrained; Lasso can be written either way; SVMs are constrained ("maximize the margin subject to every point being on the right side").

<a id="kernel"></a>

### Kernel

_Concept_

Any function that takes two vectors and returns one number. The dot product is the **linear kernel**. Kernels let an SVM measure similarity in a transformed space where the classes become separable.

<a id="kernel-trick"></a>

### Kernel trick

_Concept_

Wherever a dot product between two vectors appears in an algorithm, replace it with a kernel. This bends the model by effectively mapping the data into a higher-dimensional space without ever computing that space. Adding polynomial features does something similar but one discrete term at a time; kernelizing "curves the very axes the model is fit upon".

<a id="linear-polynomial-and-rbf-kernels"></a>

### Linear, polynomial and RBF kernels

_Concept, scikit-learn_

**Linear**: does nothing; same as not kernelizing. **Polynomial**: adjust the degree d to add curvature (degree 2 works well for NLP data). **RBF (radial basis function)**: the most adjustable, so the most common; start with this. Which kernel? "As long as it's not linear, it matters surprisingly little" or, better, grid search it.

```python
SVC(kernel="linear"); SVC(kernel="poly", degree=2); SVC(kernel="rbf")
```

<a id="mercer-kernel"></a>

### Mercer kernel

_Concept_

The mathematical condition a kernel must meet for the trick to work: there must be some mapping h with K(x, y) = h(x)·h(y). A 2nd-degree polynomial kernel maps n features to about n(n + 1)/2; the RBF kernel's h is **infinite-dimensional**, which is why regularization (C) is essential.

<a id="gamma"></a>

### gamma

_scikit-learn_

An RBF (and polynomial) kernel hyperparameter controlling how far one training point's influence reaches. High gamma = very local, wiggly boundaries (risk of overfitting). `gamma="scale"` is the sensible default.

<a id="svc-and-linearsvc"></a>

### SVC and LinearSVC

_scikit-learn_

`SVC` is the general kernel SVM. `LinearSVC` is a faster implementation for the linear kernel only, good for large datasets. Always scale first: SVMs use distances and regularization.

```python
from sklearn.svm import SVC, LinearSVC

svc = SVC(C=100, kernel="rbf", gamma="scale")
svc.fit(X_train, y_train)
accuracy_score(y_test, svc.predict(X_test))
```

<a id="mnist-digits-load-digits"></a>

### MNIST digits (load_digits)

_scikit-learn_

8×8 images of handwritten digits 0–9, each flattened into 64 pixel features. A multiclass problem where accuracy is a fair metric because every mistake is equally bad.

```python
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
plt.imshow(digits.images[42], cmap=plt.cm.gray_r)
```

<a id="svm-pros-and-cons"></a>

### SVM pros and cons

_Concept_

**Pros:** exceptional performance (historically widely used), effective with high-dimensional data, handle non-linear boundaries, fast on most datasets thanks to the kernel trick. **Cons:** complete black box, slow on very tall datasets, and must be scaled. Use them when predictive ability is all you care about.

---

<a id="s32-gradient-descent-and-optimization"></a>

## 32. Gradient descent and optimization

_How models are actually fitted when there's no formula for the answer: walk downhill on the loss function. The algorithm, the learning rate, and what goes wrong. Examples fit a slope by hand to simulated soda sales._

**In your capstone:** Explains what's happening when models fit, and why convergence warnings, scaling and learning rates matter.

<a id="optimization"></a>

### Optimization

_Concept_

Finding the minimum or maximum of an **objective function**, possibly subject to constraints. The calculus way is to set the derivative to zero and solve. With one notable exception (OLS), ML loss functions can't be solved that way, so we use iterative methods.

Maximizing f is the same as minimizing −f, so ML usually talks only about minimizing. Everyday examples: the shortest route, the cheapest delivery schedule, the best coefficients for a model.

<a id="derivative"></a>

### Derivative

_Concept_

The slope of the tangent line to a function at a point. **Positive**: the function goes up to the right, so move left to go down. **Negative**: move right. **Zero**: a minimum, maximum or inflection point. For several parameters the vector of derivatives is the **gradient**.

Example: f(x) = x² has derivative 2x, so at x = 3 the slope is 6. Setting the derivative to zero finds the flat spots: for f(x) = (x − 3)² + 1, f′(x) = 2(x − 3) = 0 gives the minimum at x = 3. But a zero derivative is necessary, not sufficient: f(x) = x³ is flat at x = 0 without a peak or valley (an inflection point).

<a id="loss-function"></a>

### Loss function

_Concept_

A function measuring the difference between y and ŷ, where **lower is better**; minimizing it means making predictions as close to the truth as possible, and the model's coefficients are the "position" being searched over. Examples: MSE (OLS), MSE + penalty (Ridge, Lasso), log loss (logistic regression), Gini/MSE (trees). Related words: **cost function** (often the average over all rows), **objective function** (whatever the algorithm optimizes).

Why isn't accuracy used for **training**? It's a step function: nudging a coefficient slightly usually changes no predictions, so its gradient is zero almost everywhere and gives gradient descent nothing to follow. Log loss is smooth, so it works. For a positive case (y = 1) log loss is −log ŷ: small when ŷ is near 1, huge when ŷ is near 0, so confident wrong answers are punished hardest.

<a id="closed-form-solution"></a>

### Closed-form solution

_Concept_

An exact formula for the answer. OLS has one: β̂ = (XᵀX)⁻¹Xᵀy, and so does ridge regression. Lasso and logistic regression don't: lasso's |β| penalty has a sharp corner at zero (no derivative there, which is also why it produces exact zeros), and log loss has no algebraic minimum. Even OLS is sometimes solved iteratively on very large data, because inverting XᵀX gets expensive with many features.

What libraries actually use: scikit-learn's `Lasso` uses **coordinate descent** (optimizing one coefficient at a time), and `LogisticRegression` defaults to **L-BFGS**, a quasi-Newton method; that's where `LogisticRegression`'s convergence warnings come from. Neural networks use gradient descent variants such as Adam.

See also: [Quasi-Newton methods (BFGS, L-BFGS)](#quasi-newton-methods-bfgs-l-bfgs)

<a id="gradient-descent"></a>

### Gradient descent

_Concept_

"Goin' downhill." Start at a random point, compute the derivative there, take a step in the **opposite direction** of the gradient, and repeat until the steps become tiny:

**new position = old position − α × gradient**

Gradient descent is how most models with no closed-form solution are fitted, including neural networks.

```python
x, alpha = 1, 1
for i in range(300):
    x = x - alpha * f_deriv(x)
```

With several coefficients, the **gradient** is the vector of derivatives with respect to each one; it points in the direction of steepest increase, so moving against it is **steepest descent**. Because each move is α × slope, the steps shrink automatically as the curve flattens near the bottom, and a slight overshoot is fine: the slope on the other side points back. Stop when the step falls below a tolerance (e.g. 10⁻⁸), the loss stops improving, or `max_iter` is reached (scikit-learn then raises a ConvergenceWarning).

Worked example: minimize f(x) = x² (f′(x) = 2x) from x₀ = 5 with α = 0.1: x₁ = 5 − 0.1 × 10 = 4, x₂ = 4 − 0.1 × 8 = 3.2; each step multiplies x by 0.8, heading towards 0.

A reusable version, applied to the deck's "food for thought" function, whose minimum has no algebraic solution:

```python
def gradient_descent(grad, x0, alpha, tol=1e-8, max_iter=10_000):
    x = x0
    for k in range(max_iter):
        step = alpha * grad(x)
        x = x - step                       # move opposite to the gradient
        if abs(step) < tol:                # stop when the move is small enough
            return x, k + 1
    return x, max_iter

f  = lambda x: -np.log(x) / (1 + x)
df = lambda x: -(1/x + 1 - np.log(x)) / (1 + x)**2
x_min, iters = gradient_descent(df, x0=1.0, alpha=5.0)   # x ≈ 3.5911, f(x) ≈ -0.2785, 155 steps
```

<a id="learning-rate-step-size"></a>

### Learning rate (step size, α)

_Concept_

How far to move each step: new position = old position − α × gradient. **Too large**: you overshoot and bounce or diverge. **Too small**: convergence takes forever, or never happens in your iteration budget. On f(x) = x² from x = 5:

```python
for alpha in [0.01, 0.1, 0.9, 1.1]:
    x, iters = gradient_descent(lambda x: 2 * x, x0=5.0, alpha=alpha, max_iter=200)
# 0.01: x = 0.088 after 200 steps (too slow) | 0.1: converged in 84 steps
# 0.9: converged in 94 (zig-zagging)        | 1.1: x ≈ 3.4e16 (diverged)
```

A practical recipe: try values on a log scale (0.001, 0.01, 0.1, 1), plot the loss against iterations, and pick the largest α that decreases the loss smoothly. α is a hyperparameter, like k in kNN.

> **Watch out:** α means different things across the course: the step size here, the regularization strength in scikit-learn (often written λ in textbooks), the significance level in statistics, and AdaBoost's model weight.

See also: [Feature scaling and gradient descent](#feature-scaling-and-gradient-descent)

<a id="feature-scaling-and-gradient-descent"></a>

### Feature scaling and gradient descent

_Concept_

Gradient descent converges much faster, and one "normal" α works, when features are **standardized**. In the soda notebook, temperatures around 45 make the gradient scale with x² (≈ 2,000), so α = 0.01 explodes (β₁ ≈ 3.8 × 10¹⁶⁵ after 100 steps) and only a tiny α (0.0001) converges; the stability limit is roughly α < 0.0005. Standardize the feature and α = 0.1 works. Poorly scaled features also create long, flat valleys in the loss surface. It's another reason for the `StandardScaler` habit.

<a id="convergence-and-tolerance"></a>

### Convergence and tolerance

_Concept, Python_

Stop when an update changes the parameter by less than a small **tolerance**, and give up after `max_iter` iterations. If you hit the limit without converging, don't trust the result: change α rather than just raising the limit.

```python
def check_update(beta_1, updated_beta_1, tolerance=0.1):
    return abs(beta_1 - updated_beta_1) < tolerance
```

<a id="gradient-descent-for-ols"></a>

### Gradient descent for OLS

_NumPy, Python_

The notebook fits β₁ in sodas = 200,000 + 1,000·temp + ε by minimizing MSE (temperatures drawn from a Poisson(45) distribution). The gradient of MSE with respect to β₁ is (2/n) Σ −xᵢ(yᵢ − (β₁xᵢ + β₀)); the vectorized NumPy line replaces a slow loop.

```python
def beta_1_gradient(x, y, beta_1, beta_0):
    return 2 * np.mean(-x * (y - (beta_1 * x + beta_0)))

def gradient_descent(x, y, beta_1=0, alpha=0.01, max_iter=100):
    for step in range(max_iter):
        new_beta_1 = beta_1 - alpha * beta_1_gradient(x, y, beta_1, 200_000)
        if abs(new_beta_1 - beta_1) < 0.1:
            return new_beta_1                  # converged
        beta_1 = new_beta_1
    print("Did not converge: don't trust beta_1")
    return beta_1

gradient_descent(df["temp"], df["sodas"], beta_1=20, alpha=0.0001)   # ≈ 1,012.7 in about 16 steps
```

The result matches the exact least-squares slope with β₀ fixed (1,012.8); the true value is 1,000, and the gap is noise in the sample. When it doesn't converge, change α rather than raising `max_iter`.

**Fitting all the coefficients at once** recovers the closed-form answer:

```python
rng = np.random.default_rng(0)
X = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
y = X @ np.array([3.0, 2.0, -1.0]) + rng.normal(scale=0.5, size=200)
beta = np.zeros(3)
for _ in range(1000):
    gradient = -2 / len(y) * X.T @ (y - X @ beta)       # gradient of the MSE
    beta -= 0.1 * gradient
beta.round(3), np.linalg.solve(X.T @ X, X.T @ y).round(3)   # identical
```

> **Watch out:** This example fixes β₀ at its true value (200,000) and only learns β₁, a deliberate simplification; fitting both gives β₀ ≈ 198,639 and β₁ ≈ 1,042.

<a id="local-vs-global-minimum"></a>

### Local vs global minimum

_Concept_

A **local minimum** is lower than its neighbours; the **global minimum** is the lowest point overall. Gradient descent can get trapped in a local minimum depending on where it starts. The fix is to check that the loss function is convex.

Gradient descent only sees the local slope, so on a non-convex loss it can stop in a local minimum. Mitigations: several random starts, momentum-based optimizers, or accepting a "good enough" minimum (common in neural networks). The starting point doesn't affect where you end up on a **convex** loss, only how long it takes.

<a id="convex-function"></a>

### Convex function

_Concept_

A bowl-shaped function: pick any two points on the curve and the straight line between them never dips below the curve (on a W-shape, the line between the two valleys passes under the middle hump). For a convex function **every local minimum is a global minimum**, so gradient descent can't get trapped. Convex: MSE for linear regression, ridge and lasso losses, and log loss for logistic regression. Non-convex: neural network losses, and some other methods such as k-means.

> **Watch out:** The "second derivative never negative" test only applies to twice-differentiable functions: |β| (the lasso penalty) is convex but has no derivative at 0.

<a id="flat-curves-vanishing-steps"></a>

### Flat curves (vanishing steps)

_Concept_

Problem 4: where the curve is very flat, the gradient is tiny, so steps shrink and progress stalls even far from the minimum.

In neural networks this is related to **vanishing gradients**. Poorly scaled features also create flat valleys; standardizing helps.

<a id="choosing"></a>

### Choosing α

_Concept_

Four approaches from the slides: (1) **manually**, by guess-and-check (also standard for neural nets); (2) **let α shrink over time** (sensible, but rarely works well); (3) **use a fancier algorithm**: IRLS (used by GLMs such as logistic regression), Adam and RMSProp (neural networks); (4) **second-order methods** such as Newton's method.

Learning-rate **schedules** (step decay, cosine decay) are widely used in deep learning alongside adaptive optimizers.

<a id="adam-rmsprop-and-irls"></a>

### Adam, RMSProp and IRLS

_Concept_

Smarter optimizers that adapt the step size automatically. **Adam** (Adaptive Moment Estimation) keeps running averages of past gradients (momentum) and of their squares, giving each parameter its own adaptive step; it's the default optimizer for most neural networks. **RMSProp** scales steps by recent gradient sizes. **Iteratively reweighted least squares (IRLS)** solves a sequence of weighted least-squares problems; statsmodels fits logistic regression this way (equivalent to Newton's method for that problem).

> **Watch out:** Adam stands for Adaptive **Moment Estimation**; "Adaptive Momentum" is a common slip.

<a id="newton-s-method"></a>

### Newton's method

_Concept_

A **second-order** method: it uses the curvature (the second derivative) as well as the slope, replacing α with 1/f″(x). Dividing by the curvature takes big steps where the function curves gently and small steps where it curves sharply; on a quadratic like x² it lands on the minimum in **one step**. With n parameters it needs the gradient ∇f and the **Hessian** ∇²f, the n × n matrix of all second derivatives: a model with a million parameters would need a trillion entries, which is why it's rarely used directly.

> **Watch out:** Newton's method converges very fast **near** the minimum of a well-behaved convex function, but it isn't guaranteed to work: it can diverge when started far away, and on non-convex functions it converges to any flat spot, including maxima and saddle points. And derivatives don't have to be estimated: automatic differentiation (PyTorch, TensorFlow, JAX) computes exact derivatives of any program.

See also: [Quasi-Newton methods (BFGS, L-BFGS)](#quasi-newton-methods-bfgs-l-bfgs)

<a id="quasi-newton-methods-bfgs-l-bfgs"></a>

### Quasi-Newton methods (BFGS, L-BFGS)

_Concept, scikit-learn_

Methods that approximate the inverse Hessian cheaply from recent gradients, getting much of Newton's speed without computing second derivatives. **L-BFGS** (limited-memory BFGS) is scikit-learn's default solver for `LogisticRegression` (`solver="lbfgs"`), so you've been using a near-Newton method all along. `scipy.optimize.minimize` uses BFGS by default for unconstrained problems.

```python
from scipy.optimize import minimize
minimize(lambda v: f(v[0]), x0=[1.0], method="L-BFGS-B", bounds=[(0.01, None)]).x   # ≈ [3.591]
```

<a id="stochastic-gradient-descent-sgd"></a>

### Stochastic gradient descent (SGD)

_Concept, scikit-learn_

A variant that estimates the gradient from one row or a small batch at a time instead of the whole dataset: noisier steps, but much faster on big data. The basis for training neural networks, and available in scikit-learn as `SGDRegressor` and `SGDClassifier`.

---

<a id="s33-unsupervised-learning-and-k-means-clustering"></a>

## 33. Unsupervised learning and k-means clustering

_No y-variable: finding structure in X itself. What clustering is for, how k-means works, and how to judge clusters when there's no right answer, using inertia, the elbow method and the silhouette score. Examples use `make_blobs`, Old Faithful geyser eruptions and the seeds and iris labs._

**In your capstone:** For segmentation questions: scale, compare k with inertia and silhouette, then profile the clusters so they mean something to the business.

<a id="unlabeled-data"></a>

### Unlabeled data

_Concept_

Data with no y-variable (no **label**). Unsupervised learning works with it. There is no clear objective, no "right answer" and so no easy way to tell how well you're doing; results need domain expertise and human interpretation.

See also: [Unsupervised learning](#unsupervised-learning)

<a id="semi-supervised-learning"></a>

### Semi-supervised learning

_Concept_

A model that uses both labelled and unlabelled data, useful when labels are expensive to get.

<a id="clustering"></a>

### Clustering

_Concept_

Grouping observations into clusters so that members of the same cluster are more similar to each other than to members of other clusters. The algorithm decides what the clusters look like; you decide what they mean.

<a id="uses-of-clustering"></a>

### Uses of clustering

_Concept_

**Market segmentation** (which customer group isn't buying, and why?), **recommender systems** (what are people in your cluster watching?), **political polling** (voter groups), discovering categories, reducing the number of classes (apples, pears → fruit), dimensionality reduction, and **pre-processing**: creating labels for supervised learning.

<a id="clustering-algorithms"></a>

### Clustering algorithms

_Concept_

There are many because "the notion of cluster cannot be precisely defined": **k-means** (mean centroids), **DBSCAN** (density), **hierarchical/agglomerative** (merging), affinity propagation, mean shift, spectral clustering, Birch and k-medoids. Clustering is exploratory, not confirmatory.

<a id="birch-and-affinity-propagation"></a>

### Birch and Affinity Propagation

_scikit-learn_

Two further scikit-learn clustering algorithms, offered as a bonus in the comparison lab. **Birch** builds a compact tree of cluster summaries in one pass over the data, so it scales to very large datasets; you can still set `n_clusters` for the final grouping. **Affinity Propagation** finds clusters by points "sending messages" to each other until a set of exemplar points emerges; like DBSCAN it **chooses the number of clusters itself**, controlled by its `damping` and `preference` settings, and it's slow on large data.

```python
from sklearn.cluster import Birch, AffinityPropagation

Birch(n_clusters=3).fit(Xs).labels_                       # scaled iris: ARI ≈ 0.66
ap = AffinityPropagation(random_state=42).fit(Xs)
len(ap.cluster_centers_indices_)                          # 9 clusters found on scaled iris
```

Left to choose for itself, Affinity Propagation finds 9 clusters on iris, a reminder that "the algorithm picks k" doesn't mean it picks the k you'd expect.

<a id="k-means"></a>

### k-means

_Concept_

Partitions the data into **k** clusters, each with a **centroid** (its mean), assigning every point to its nearest centroid so as to minimize the within-cluster sum of squares. Algorithm: (1) pick k; (2) initialize k centroids; (3) assign each point to the nearest centroid; (4) move each centroid to the centre of its cluster; (5) repeat 3–4 until the centroids stop moving. It's guaranteed to converge, but possibly to a local minimum.

k-means minimizes the within-cluster squared distance, and its loss is non-convex: different starting centroids can give different answers, which is why `n_init` runs it several times.

<a id="centroid"></a>

### Centroid

_Concept_

The centre (mean position) of a cluster. It doesn't have to be an actual data point.

<a id="centroid-initialization"></a>

### Centroid initialization

_scikit-learn_

How the starting centroids are chosen (`init`): **`"k-means++"`** (default: spreads the starting centroids far apart), `"random"`, or an array of starting points you supply (only if you already know the data well). Because results depend on the start, scikit-learn runs several initializations (`n_init`) and keeps the best.

<a id="kmeans"></a>

### KMeans

_scikit-learn_

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=42)
df = pd.DataFrame(data, columns=["x1", "x2"])
X_scaled = StandardScaler().fit_transform(df)       # very important step!
km = KMeans(n_clusters=3, random_state=42)
km.fit(X_scaled)
df["cluster"] = km.labels_
```

> **Watch out:** `KMeans()` with no arguments uses **8 clusters**; always set `n_clusters`. And since scikit-learn 1.4, `n_init="auto"` runs k-means only **once** with k-means++, so an unlucky start can land in a poor local minimum. On scaled iris, a single run with `random_state=42` splits setosa in two and merges the other species (ARI 0.43); `n_init=10` gives the clean answer (ARI ≈ 0.62). Without `random_state`, the clusters change on every run. Use `KMeans(n_clusters=3, n_init=10, random_state=42)`.

<a id="labels-and-cluster-centers"></a>

### labels* and cluster_centers*

_scikit-learn_

After fitting, `labels_` holds each row's cluster number (same as `km.predict(X)`), and `cluster_centers_` holds the centroid coordinates **in the scaled units** the model saw. Un-scale them with the scaler's `inverse_transform` to read them in original units.

```python
km.labels_
centroids = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=["x1", "x2"])
df.groupby("cluster").mean()        # same centroids, computed from the data
```

<a id="cluster-labels-are-arbitrary"></a>

### Cluster labels are arbitrary

_Concept, pandas_

Cluster numbers are just names: "cluster 0" in one run might be "cluster 2" in the next, and neither has anything to do with class 0 in your true labels. That's why the evaluation lab asks you to match clusters to species before comparing them. Compare with a **crosstab**, or use metrics that ignore the numbering (ARI, homogeneity, completeness, V-measure).

```python
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

iris = sns.load_dataset("iris")                   # or sklearn.datasets.load_iris()
X, y = iris.drop(columns="species"), iris["species"]
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit(StandardScaler().fit_transform(X)).labels_

pd.crosstab(y, labels)                            # setosa sits alone in one cluster;
                                                  # versicolor and virginica overlap
species_ind = LabelEncoder().fit_transform(y)     # 'setosa' -> 0, 'versicolor' -> 1, ...
mapping = pd.crosstab(labels, species_ind).idxmax(axis=1)    # each cluster -> its majority class
aligned = pd.Series(labels).map(mapping)          # now comparable with species_ind
```

See also: [External cluster metrics](#external-cluster-metrics), [Adjusted Rand index (ARI)](#adjusted-rand-index-ari)

<a id="scaling-before-clustering"></a>

### Scaling before clustering

_Concept_

k-means uses distances, so a feature with a large range dominates. In the slides, the red point's 5 nearest neighbours look completely different once the x-axis is rescaled. Standardize first, though in the seeds lab unscaled data happened to score better, so it's worth checking.

The seeds lab writes a function that tries every k on the raw data, `normalize`d rows, `StandardScaler` and `MinMaxScaler`, and keeps the highest silhouette.

> **Watch out:** There's no universally best scaler, so compare a few. Silhouette scores computed on differently scaled versions of the data are each measured in their own space, so comparing them across scalings is only a rough guide.

<a id="make-blobs"></a>

### make_blobs()

_scikit-learn_

Generates round, well-separated clusters: the ideal case for k-means. `make_circles()` and `make_moons()` generate the shapes k-means fails on.

<a id="inertia-within-cluster-sum-of-squares"></a>

### Inertia (within-cluster sum of squares)

_Concept, scikit-learn_

The sum of squared distances from each point to its cluster's centroid. **Low inertia = dense clusters.** Worked example from the slides: centroid (5, 7) and points (7, 10), (8, 8), (3, 5), (7, 5) give 13 + 10 + 8 + 8 = 39. Problems: it isn't interpretable, and it always falls as k rises, so you can't just minimize it.

```python
km.inertia_
```

<a id="elbow-method"></a>

### Elbow method

_Concept, matplotlib_

Plot inertia against k and look for the "elbow": the k after which adding clusters stops improving inertia much.

```python
inertia_list = [KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_
                for k in range(1, 11)]
plt.plot(range(1, 11), inertia_list, marker="o")
```

<a id="silhouette-score"></a>

### Silhouette score

_Concept, scikit-learn_

How well separated and cohesive the clusters are. For each point, **sᵢ = (bᵢ − aᵢ) / max(aᵢ, bᵢ)**, where aᵢ (**cohesion**) is its mean distance to points in its own cluster and bᵢ (**separation**) is its mean distance to points in the nearest other cluster. Average over all points. **1** = perfect clustering, **0** = just OK (overlapping), **−1** = points in the wrong clusters. Higher is better; needs k ≥ 2.

```python
from sklearn.metrics import silhouette_score

silhouette_list = [silhouette_score(X_scaled,
                                    KMeans(n_clusters=k, random_state=42).fit(X_scaled).labels_)
                   for k in range(2, 11)]
plt.plot(range(2, 11), silhouette_list, marker="o")
```

> **Watch out:** Silhouette rewards compact, round blobs, so it can mislead for density-based clusters. On the circles data, DBSCAN's correct two-ring solution scores silhouette **0.24**, lower than k-means' completely wrong solution (0.27). For non-convex clusters, judge visually or with labels. And compute it on the **same (scaled) data** the model was fitted on; passing the unscaled X is an easy slip.

<a id="choosing-the-number-of-clusters"></a>

### Choosing the number of clusters

_Concept_

There's no single right answer. Options: domain knowledge or business needs ("I need three marketing profiles"), visualization (only practical in 2–3 dimensions), the elbow method, the silhouette score, or, if the clusters feed a supervised model, tuning k for that model's performance.

> **Watch out:** The silhouette score finds the **best-separated** grouping, which isn't always the "true" number of groups. On scaled iris it prefers k = 2 (0.58) over k = 3 (0.48), because setosa is far from the other two species, which overlap. The seeds lab's automatic search also picks k = 2 for three wheat varieties, which is why silhouette alone can be untrustworthy. Combine scores with plots and domain knowledge.

<a id="k-means-assumptions"></a>

### k-means assumptions

_Concept_

k-means works best when k is right, clusters are **isotropic** (round/spherical), every variable has similar variance, and clusters are roughly the same size. These are "pretty much never met with real data", yet k-means often still works well.

<a id="k-means-sensitivities"></a>

### k-means sensitivities

_Concept_

k-means is sensitive to **outliers** (they drag centroids), **centroid initialization** (different starts, different clusters) and **non-linear shapes** (it can't find rings or crescents). Worst of all, it always returns k clusters, even when the data have no natural groups.

<a id="k-means-strengths-and-weaknesses"></a>

### k-means strengths and weaknesses

_Concept_

**Strengths:** simple, computationally efficient, intuitive results, always converges. **Weaknesses:** scale dependent, unsuitable for varying shapes and densities, evaluation is subjective, sensitive to outliers, initialization and non-linearity.

<a id="transfer-learning"></a>

### Transfer learning

_Concept_

Reusing a model trained on one task as the starting point for a related task. Using cluster labels as inputs to a later supervised model is mentioned as part of this idea.

---

<a id="s34-dbscan-and-hierarchical-clustering"></a>

## 34. DBSCAN and hierarchical clustering

_Two clustering methods that break k-means's rules: DBSCAN finds dense regions of any shape and labels outliers, and agglomerative clustering builds clusters by merging. Plus metrics for comparing clusters with known labels. Examples use NHL team statistics, `make_circles` and seven shape-testing datasets (flame, spiral, jain…)._

**In your capstone:** Use DBSCAN or HDBSCAN for irregular shapes and outliers, and agglomerative clustering when a hierarchy helps tell the story.

<a id="dbscan"></a>

### DBSCAN

_Concept_

**D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise. Clusters are areas of **high density** separated by low-density areas. It finds clusters of any contiguous shape, works out how many clusters there are, and leaves outliers unclustered.

Proposed by Ester, Kriegel, Sander and Xu (1996). Clusters grow by connecting **core points** that are within eps of each other ("reachability"); border points are reached but don't spread the cluster further. Uses: spatial data (GPS hotspots, crime locations), anomaly detection (unusual transactions as noise), irregularly shaped groups.

See also: [HDBSCAN](#hdbscan)

<a id="eps-and-min-samples"></a>

### eps and min_samples

_scikit-learn_

DBSCAN's two hyperparameters. **`eps`** (epsilon) is the search radius: the maximum distance for two points to count as neighbours. **`min_samples`** is the number of points (**including the point itself**) that must lie within eps of a point for it to be a **core point**. The defaults are `eps=0.5, min_samples=5`. Always scale features first, because eps is a distance.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(X_scaled)
dbscan.labels_
```

Results are very sensitive to eps. On the NHL practice lab (90 team-seasons, 25 standardized features, `min_samples=3`): eps = 2 gives 2 clusters and 84 of 90 points as noise; eps = 3 gives 5 clusters and 39 noise; eps = 4 gives 1 cluster and 4 noise. Having to choose both is "a huge disadvantage".

> **Watch out:** `min_samples` isn't "the minimum size of a cluster": it's the neighbourhood count (including the point itself) that makes a point a core point.

See also: [k-distance plot](#k-distance-plot)

<a id="k-distance-plot"></a>

### k-distance plot

_scikit-learn, matplotlib_

A heuristic for choosing eps: for each point, find the distance to its k-th nearest neighbour (k = min_samples), sort those distances, and plot them. The "elbow", where distances start rising steeply, is a reasonable eps: points beyond it sit in sparse regions.

```python
from sklearn.neighbors import NearestNeighbors

k = 3
dists, _ = NearestNeighbors(n_neighbors=k).fit(X_scaled).kneighbors(X_scaled)
plt.plot(np.sort(dists[:, -1]))          # look for the elbow
```

<a id="core-border-and-noise-points"></a>

### Core, border and noise points

_Concept_

**Core points** have at least `min_samples` points within `eps`. **Border points** are in a cluster (within `eps` of a core point) but don't have enough neighbours themselves. **Noise points** belong to no cluster.

```python
core = np.zeros_like(dbscan.labels_, dtype=bool)
core[dbscan.core_sample_indices_] = True
```

<a id="noise-label-1"></a>

### Noise label (−1)

_scikit-learn, Python_

DBSCAN labels noise points **−1**, so don't count it as a cluster.

```python
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
(labels == -1).sum()          # number of outliers
```

> **Watch out:** Clustering metrics treat −1 as just one more cluster. On the "jain" shape dataset, DBSCAN finds one cluster plus 100 noise points yet still scores ARI 0.97, because the noise happens to match the sparser true cluster.

<a id="dbscan-pros-and-cons"></a>

### DBSCAN pros and cons

_Concept_

**Pros:** any contiguous shape (k-means only handles round blobs); tells you how many clusters there are; finds outliers. **Cons:** needs roughly uniform density; two hyperparameters to tune; bad settings flag too many outliers; struggles when data are dense in one area and spread out elsewhere (as with the NHL data). Ideal for rings (`make_circles`).

**Verified on the course's circles data:** DBSCAN recovers both rings perfectly (adjusted Rand index 1.0), while k-means slices them into halves (ARI ≈ 0).

See also: [HDBSCAN](#hdbscan)

<a id="hdbscan"></a>

### HDBSCAN

_scikit-learn_

Hierarchical DBSCAN: handles clusters of **varying density**, DBSCAN's main weakness, and only needs `min_cluster_size` (no eps). In scikit-learn since version 1.3; noise points are still labelled −1.

```python
from sklearn.cluster import HDBSCAN
labels = HDBSCAN(min_cluster_size=10).fit_predict(X_scaled)
```

<a id="make-circles"></a>

### make_circles()

_scikit-learn_

Generates one ring inside another: the shape where DBSCAN succeeds and k-means fails.

```python
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=123)
```

<a id="hierarchical-clustering"></a>

### Hierarchical clustering

_Concept_

Builds a hierarchy of nested clusters. **Agglomerative** (bottom-up): start with every point as its own cluster and merge the closest pair repeatedly. **Divisive** (top-down): start with one cluster and split. Agglomerative is almost always preferred.

<a id="agglomerative-clustering"></a>

### Agglomerative clustering

_scikit-learn_

Merge clusters one by one until `n_clusters` remain, using a **linkage** rule to decide which clusters are closest.

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg.fit(X_scaled)
agg.labels_
```

Instead of `n_clusters`, you can set `distance_threshold` to cut the tree at a chosen height.

<a id="linkage"></a>

### Linkage

_Concept_

How the distance between two clusters is measured when merging. **Ward** (default): merge the pair that least increases within-cluster variance; it makes compact, similar-sized, k-means-like clusters and requires Euclidean distance. **Single**: minimum distance between any two points; it follows chains of nearby points, so it handles long, curved shapes, but can "chain" separate clusters together through a few bridging points. Also **complete** (maximum distance) and **average** (average distance).

On the comparison lab's spiral data, single linkage scores ARI **1.0** vs ward ≈ 0; on "pathbased", single scores 0.0 (chaining merges everything) vs ward 0.48.

<a id="dendrogram"></a>

### Dendrogram

_Concept, SciPy_

A tree diagram of the merges in agglomerative clustering, from individual points at the bottom up to a single cluster ("until All is One"). The height of each join shows how far apart the merged clusters were; cutting the tree at a height gives a set of clusters.

```python
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X_scaled, method="ward")
dendrogram(Z)
```

Long vertical gaps between joins suggest natural places to cut.

<a id="comparing-clustering-algorithms"></a>

### Comparing clustering algorithms

_Concept_

The comparison lab runs k-means, agglomerative clustering and DBSCAN on seven benchmark shape datasets with known labels. Scored with the adjusted Rand index (1 = perfect, 0 = chance), using the notebook's parameters:

| Dataset     | True k | k-means | Ward  | Single | DBSCAN |
| ----------- | ------ | ------- | ----- | ------ | ------ |
| flame       | 2      | 0.45    | 0.19  | 0.01   | 0.88   |
| aggregation | 7      | 0.76    | 0.81  | 0.80   | 0.81   |
| compound    | 6      | 0.54    | 0.55  | 0.74   | 0.96   |
| jain        | 2      | 0.32    | 0.51  | 0.26   | 0.97   |
| pathbased   | 3      | 0.46    | 0.48  | 0.00   | 0.92   |
| r15         | 15     | 0.99    | 0.98  | 0.54   | 0.79   |
| spiral      | 3      | −0.01   | −0.00 | 1.00   | 1.00   |

No algorithm wins everywhere. DBSCAN excels on irregular shapes; k-means and ward are best on round, well-separated blobs (r15); single linkage is perfect on the spiral and useless on pathbased. **Choosing a method:** round blobs and a known k → k-means; irregular shapes and outliers → DBSCAN or HDBSCAN; you need a hierarchy or dendrogram → agglomerative, with the linkage chosen to suit the shapes. Always look at the clusters, not just a score.

> **Watch out:** Set `random_state=42` on `KMeans` so results don't change on each run. k-medoids isn't in core scikit-learn (it's in the separate `scikit-learn-extra` package).

See also: [Adjusted Rand index (ARI)](#adjusted-rand-index-ari)

<a id="adjusted-rand-index-ari"></a>

### Adjusted Rand index (ARI)

_scikit-learn_

An external clustering metric: how well the cluster labels agree with the true labels, counting pairs of points placed together or apart, corrected for chance. **1 = perfect match, ≈ 0 = random, can be slightly negative.** Like V-measure, it ignores the cluster numbering.

```python
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(df["label"], AgglomerativeClustering(3, linkage="single").fit(df[["x", "y"]]).labels_)
```

<a id="external-cluster-metrics"></a>

### External cluster metrics

_Concept, scikit-learn_

When true labels happen to be available (as in the labs), you can check clusters against them. **Homogeneity**: each cluster contains only one class. **Completeness**: all members of a class end up in the same cluster. **V-measure**: their harmonic mean. **Adjusted mutual information**: agreement corrected for chance. All range up to 1 and don't care about the cluster numbering.

```python
from sklearn import metrics
metrics.homogeneity_score(y, labels)
metrics.completeness_score(y, labels)
metrics.v_measure_score(y, labels)
metrics.adjusted_mutual_info_score(y, labels)
```

On the NHL data, DBSCAN's clusters barely match team rank (homogeneity 0.118, completeness 0.103, V-measure 0.11): not every dataset has natural clusters, which is itself a useful finding.

See also: [Adjusted Rand index (ARI)](#adjusted-rand-index-ari)

<a id="normalize-vs-scalers"></a>

### normalize() vs scalers

_scikit-learn_

`sklearn.preprocessing.normalize` rescales each **row** to length 1 (useful for direction-based similarity), whereas `StandardScaler` and `MinMaxScaler` rescale each **column**. The seeds lab compares all three for k-means.

```python
from sklearn.preprocessing import normalize, MinMaxScaler
Xn = normalize(X)
Xmm = MinMaxScaler().fit_transform(X)
```

---

<a id="s35-principal-component-analysis-pca"></a>

## 35. Principal component analysis (PCA)

_Squeezing many correlated variables into a few uncorrelated ones that keep most of the information. What PCA does visually, how SVD does it mathematically, how many components to keep, and when not to use it. Examples use the wine data and the "Doug the Pug" image compression demo._

**In your capstone:** Use PCA to compress many correlated features or to visualize structure, and tune the number of components inside a pipeline.

<a id="dimensionality-reduction"></a>

### Dimensionality reduction

_Concept_

Reducing the number of variables (dimensions) while keeping as much useful information as possible. Helps when there are too many variables, some meaningless and some redundant (multicollinear). An unsupervised technique.

Two kinds: **feature selection** keeps a subset of the original features (lasso dropping features); **feature extraction** creates new features from combinations of the originals (PCA). PCA is the most common linear method; t-SNE and UMAP are non-linear methods used mainly for visualization. Examples of wide data: surveys with 200 questions, sensors with hundreds of readings, images with thousands of pixels, the 90 polynomial wine features.

<a id="principal-component-analysis-pca"></a>

### Principal component analysis (PCA)

_Concept_

Condenses your x-variables into **principal components**: new variables that are combinations of the originals, ordered by how much of the data's variability (its "information") they capture, and **uncorrelated** with each other. Keep the first few and drop the rest: perhaps z₁…z₁₀ carry as much information as x₁…x₂₀₀. Geometrically, PCA is a **rotation** of the (centred) axes to line up with the directions the data spread along. PCA ignores y entirely, which is why it's often used as a preprocessing step before a supervised model.

Benefits: faster models, less overfitting, easy 2D/3D visualization, and no multicollinearity.

> **Watch out:** PCs are **uncorrelated** (perpendicular directions), which isn't the same as independent unless the data are multivariate normal. And a component doesn't automatically keep the predictive power of the variables it combines: because PCA is unsupervised, the highest-variance direction isn't necessarily the most predictive, and occasionally a small-variance direction is exactly what separates the classes.

See also: [PCA (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#principal-component-analysis-pca)

<a id="principal-component"></a>

### Principal component

_Concept_

One of the new axes PCA creates. PC1 points in the direction of greatest variance, PC2 the greatest remaining variance at right angles to PC1, and so on. Sometimes called **rotations**, because geometrically PCA rotates the axes to line up with the data.

<a id="pca-in-scikit-learn"></a>

### PCA in scikit-learn

_scikit-learn_

Scale first, then fit and transform. The result `Z` has the same number of rows, with columns now being principal components.

```python
from sklearn.decomposition import PCA

X_sc = StandardScaler().fit_transform(X)
pc = PCA()                       # or PCA(n_components=5)
Z = pc.fit_transform(X_sc)
plt.scatter(Z[:, 0], Z[:, 1], c=wine["red_wine"], alpha=0.2, cmap="Dark2")
```

> **Watch out:** Always standardize first, or the column with the biggest units dominates PC1.

`PCA(n_components=0.9)` keeps just enough components for 90% of the variance. `pc.transform(X)` returns the **PC scores**, the data in the new coordinates.

On the wine data (6,497 wines, 12 standardized features, target `red_wine`): PC1 explains 25.3%, PC2 22.1%, PC3 13.7% (61% in three), and 8 of 12 components reach 90%. A PC1-vs-PC2 scatter separates red and white wines well, even though PCA never saw the labels.

> **Watch out:** Be clear about the target: here the model predicts `red_wine` (colour), with quality kept as a feature. PCA strictly requires **centring**; standardizing as well is needed whenever features use different units, which is almost always.

<a id="explained-variance-and-explained-variance-ratio"></a>

### Explained variance and explained variance ratio

_scikit-learn_

`explained_variance_` is the variance captured by each component; `explained_variance_ratio_` scales those to proportions that sum to 1, so you can say "the first two components explain 50% of the variance". Use `np.cumsum` to see how many components reach, say, 90%.

```python
pc.explained_variance_ratio_
np.cumsum(pc.explained_variance_ratio_)
```

Each component's variance is σ²/(n − 1), where σ is its singular value; that's exactly `explained_variance_`.

```python
np.argmax(np.cumsum(pc.explained_variance_ratio_) >= 0.9) + 1    # components needed for 90%: 8
```

> **Watch out:** `explained_variance_` holds the **variances** of the components, not the singular values (σ) themselves; the two are easy to mix up.

<a id="loadings-components"></a>

### Loadings (components\_)

_scikit-learn_

`pc.components_` holds the weights (loadings) that build each component from the original features, one row per component. Large absolute weights show which features drive a component. On the standardized wine data, PC1 is driven by **total sulfur dioxide (0.47), free sulfur dioxide (0.42) and volatile acidity (−0.39)**. This is the Vᵀ matrix from the SVD.

```python
pd.DataFrame(pc.components_[:3], columns=X.columns, index=["PC1", "PC2", "PC3"])
pd.Series(pc.components_[0], index=X.columns).sort_values(key=abs, ascending=False).head(3)
```

> **Watch out:** Read loadings from PCA (or SVD) on **standardized** data. On unscaled data, sulfur dioxide and density appear to dominate PC1 simply because of their units.

<a id="scree-plot"></a>

### Scree plot

_matplotlib_

A plot of explained variance ratio against component number. The old statistician's rule of thumb: keep the components before the "elbow". In modern practice, grid search the number of components instead.

```python
plt.plot(pc.explained_variance_ratio_, marker="o")
```

Common rules for how many components to keep: enough to explain 80–90% of the variance; the elbow of the scree plot; or, best when there's a downstream model, tune the number by cross-validation.

<a id="choosing-n-components-with-grid-search"></a>

### Choosing n_components with grid search

_scikit-learn_

Put PCA inside a pipeline and treat the number of components as a hyperparameter.

```python
pipe = Pipeline([
    ("sc", StandardScaler()),
    ("poly", PolynomialFeatures(include_bias=False)),
    ("pc", PCA()),
    ("lr", LogisticRegression(max_iter=1000)),
])
gs = GridSearchCV(pipe, {"pc__n_components": np.arange(1, 40)}, cv=5, n_jobs=4)
gs.fit(X_train, y_train)
```

In the course pipeline (scaler → 90 polynomial features → PCA → logistic regression with `penalty=None`, `max_iter=5000`), CV accuracy for predicting wine colour rises quickly then flattens: 1 PC 0.803, 2 → 0.814, 3 → 0.857, 5 → 0.937, 10 → 0.978, 20 → 0.984, best 36 PCs (0.992 CV, 0.993 test). A few components get you most of the way.

> **Watch out:** The older syntax `LogisticRegression(penalty='none')` errors in current scikit-learn; use `penalty=None` (or `C=np.inf` from 1.8), and check the documentation for your version.

<a id="singular-value-decomposition-svd"></a>

### Singular value decomposition (SVD)

_Concept, NumPy_

Every matrix can be decomposed as **X = U D Vᵀ**, like factorizing a number (12 = 2 × 2 × 3) to reveal its structure: rotation × stretching × rotation. For an n × p centred (usually standardized) data matrix: **U** (n × p) holds the left-singular vectors, **D** (p × p, diagonal) the **singular values** (non-negative, in descending order, measuring how much information each direction holds), and **Vᵀ** (p × p) the right-singular vectors, whose rows are the principal component directions. U and V are orthogonal (UᵀU = I). PCA is computed this way, and Z = XV gives the PC scores.

```python
u, s, vh = np.linalg.svd(X_sc, full_matrices=False)      # SCALED data
np.allclose(np.abs(vh), np.abs(pc.components_))           # True (signs can flip)
np.allclose(s**2 / (len(X_sc) - 1), pc.explained_variance_)   # True
```

> **Watch out:** Run the SVD on the **same standardized data** PCA was fitted on, or the vectors won't match `pc.components_`. Singular values are non-negative (they can be zero).

<a id="dimensionality-reduced-reconstruction"></a>

### Dimensionality-reduced reconstruction

_NumPy_

Keep only the first k singular values and vectors to rebuild an approximation of the data. The Doug the Pug demo does this to an image: k = 50 or 100 looks almost like the full picture with a fraction of the numbers, while k = 5 is a blur.

```python
doug = np.mean(plt.imread("doug.jpg"), axis=2)      # greyscale image as a matrix
u, s, vh = np.linalg.svd(doug)
k = 50
approx = u[:, :k] @ np.diag(s)[:k, :k] @ vh[:k, :]
plt.imshow(1 - approx, cmap="Greys")
```

<a id="pca-assumptions"></a>

### PCA assumptions

_Concept_

**Linearity**: PCA finds linear relationships (from the covariance matrix) and misses non-linear ones; methods like t-SNE handle those. **Large variance means importance**: directions with lots of spread are assumed to matter and directions with little spread are discarded.

<a id="when-to-use-pca-and-when-not"></a>

### When to use PCA (and when not)

_Concept_

**Use it** when p ≫ n (or p ≈ n), when variables are highly multicollinear (including images, which are correlated by design), or when there are simply many variables. **Don't use it** when you need to interpret the model, since components are blends of the original features, or when you only have a few features.

Summary of the trade-off: fewer, uncorrelated features (faster, less overfitting, no multicollinearity), but the components are hard to interpret and PCA ignores the target. PCA **extracts** features; it doesn't **select** them.

<a id="covariance-matrix-eigenvectors-and-eigenvalues"></a>

### Covariance matrix, eigenvectors and eigenvalues

_Concept_

Matt's "three signposts" for explaining PCA in an interview. PCA finds the **eigenvectors** of the data's **covariance matrix** (the directions of the principal components); the **eigenvalues** say how much variance each direction captures.

---

<a id="s36-recommender-systems"></a>

## 36. Recommender systems

_Suggesting items a user will like. The main families of recommenders, and an item-based collaborative filter built from MovieLens ratings with a pivot table, a sparse matrix and cosine similarity._

**In your capstone:** For "what should we recommend?" questions with ratings data, an item-based cosine-similarity recommender is a solid first build.

<a id="recommender-system"></a>

### Recommender system

_Concept_

A system that predicts which items (films, products, songs) a user will like and suggests them. The approach you choose depends on what rating data, if any, you have.

<a id="collaborative-filtering"></a>

### Collaborative filtering

_Concept_

Recommends based on **patterns in many users' ratings**, without needing to know anything about the items themselves. **Item-based**: "people who liked this film also liked…" (items are similar if the same users rate them similarly). **User-based**: find users similar to you and recommend what they liked.

<a id="content-based-knowledge-based-and-hybrid-recommenders"></a>

### Content-based, knowledge-based and hybrid recommenders

_Concept_

**Content-based** recommenders use information about the items themselves (genre, actors, keywords). **Knowledge-based** recommenders use input from the user to make informed suggestions (filters, preferences). **Hybrid** recommenders combine several approaches.

<a id="utility-ratings-matrix"></a>

### Utility (ratings) matrix

_pandas_

A table with one row per item, one column per user and ratings as values: the input for an item-based recommender. (For user-based, swap rows and columns.) Most cells are empty because each user rates only a few items.

```python
df = pd.merge(ratings, movies, on="movieId", how="left")
rating_mx = pd.pivot_table(df, index="title", columns="userId", values="rating")
```

Build it from a merge of ratings and movie titles. A **left** merge keeps every rating and attaches its title (many ratings to one movie). Always check row counts after a merge: the merged table should have exactly as many rows as `ratings`; more means duplicate keys, fewer means lost rows.

```python
df = pd.merge(ratings, movies, on="movieId", how="left")
ratings.shape, movies.shape, df.shape            # rows of df should equal rows of ratings
df = df.drop(columns=["timestamp", "movieId", "genres"])
```

See also: [.merge() (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#merge)

<a id="sparse-matrix"></a>

### Sparse matrix

_SciPy_

A data structure that stores only the non-zero values of a mostly-empty matrix, saving huge amounts of memory. Fill missing ratings with 0 first.

```python
from scipy import sparse
ratings_sparse = sparse.csr_matrix(rating_mx.fillna(0))
```

<a id="cosine-similarity"></a>

### Cosine similarity

_Concept_

Similarity between two vectors measured by the **angle** between them: cos θ = (a·b) / (‖a‖‖b‖). 1 = pointing the same way (identical taste pattern), 0 = unrelated. It ignores vector length, so a user who rates everything highly still compares sensibly. **Cosine distance** = 1 − cosine similarity.

<a id="pairwise-distances"></a>

### pairwise_distances()

_scikit-learn_

Computes the distance between every pair of rows, returning a square matrix. With `metric="cosine"` it compares every film with every other film.

```python
from sklearn.metrics.pairwise import pairwise_distances

dists = pairwise_distances(ratings_sparse, metric="cosine")
df_rec = pd.DataFrame(dists, index=rating_mx.index, columns=rating_mx.index)
df_rec["John Wick (2014)"].sort_values().head()       # closest = most similar
```

`cosine_distances(ratings_sparse)` is an identical, shorter alternative.

<a id="mean-centering-ratings"></a>

### Mean centering ratings

_Concept_

Subtracting each user's (or the overall) average rating so that a 4 from a harsh critic and a 4 from someone who rates everything 5 aren't treated the same. Mentioned as an optional step before computing similarities.

> **Watch out:** Films with very few ratings produce strange recommendations (the notebook's "What the hell?" Frozen example). Filtering out rarely rated items helps.

---

<a id="s37-bayesian-statistics"></a>

## 37. Bayesian statistics

_A one-shot introduction to the other school of statistics: treating unknown parameters as uncertain, describing that uncertainty with probability distributions, and updating them with data (prior + data → posterior). Covers Bayes' theorem, probability mass and density functions, conjugate priors, MCMC and an election-forecasting example, building on the probability and simulation sections._

**In your capstone:** Use Bayesian methods to combine prior knowledge with limited data, and to report results as direct probabilities ("P(B beats A) = 97%").

<a id="bayesian-statistics"></a>

### Bayesian statistics

_Concept_

An approach to statistics that treats unknown quantities (parameters) as **uncertain**, describes that uncertainty with probability distributions, and **updates** them as data arrives. The workflow: (1) make parametric assumptions about the process; (2) assign **prior** beliefs to the parameters, reflecting prior knowledge and how confident you are; (3) collect data; (4) combine prior and data into **posterior** estimates. Today's posterior can become tomorrow's prior as new data arrives. Named after Thomas Bayes (c. 1701–1761), an English minister and mathematician whose theorem was published after his death; Pierre-Simon Laplace developed the ideas much further.

See also: [Frequentist vs Bayesian](#frequentist-vs-bayesian), [Bayes' theorem](#bayes-theorem)

<a id="frequentist-vs-bayesian"></a>

### Frequentist vs Bayesian

_Concept_

The philosophical difference is about **parameters**. In **frequentist** statistics (everything earlier in the course: hypothesis tests, p-values, confidence intervals), a parameter such as a true average is a **fixed, unknown number**, and probability describes how often things happen over many repetitions. In **Bayesian** statistics, parameters are **random variables** with distributions, so you can assign prior information to them and make direct probability statements about them.

The usual case made for Bayesian methods: frequentism struggles with one-time events (an election), small or expensive samples, bad luck in small experiments, and can only use data, not other knowledge. The real heart of it is the last point: Bayesian methods formally include prior knowledge (expert opinion, previous studies).

> **Watch out:** Neither approach always wins. Frequentist methods are used for elections and small samples too, and with lots of data the two approaches usually give similar answers. Both are widely used.

See also: [Probability](#probability)

<a id="bayes-theorem"></a>

### Bayes' theorem

_Concept_

Derived from the multiplication rule: P(A ∩ B) = P(A)P(B | A) = P(B)P(A | B), so

**P(B | A) = P(A | B) · P(B) / P(A)**

It **flips the conditional**, turning P(A | B), which is often easy to know, into P(B | A), which is what we usually want. The classic example: a disease affects 1% of people; a test catches 99% of cases but gives false positives 5% of the time. Then P(disease | positive) = 0.99 × 0.01 / (0.99 × 0.01 + 0.05 × 0.99) ≈ **0.167**: only about 1 in 6 positives actually have the disease.

```python
p_d, sens, false_pos = 0.01, 0.99, 0.05
p_pos = sens * p_d + false_pos * (1 - p_d)          # total probability of testing positive
sens * p_d / p_pos                                   # 0.167
```

See also: [Base rate fallacy](#base-rate-fallacy), [Multiplication rule (intersections)](#multiplication-rule-intersections), [Conditional probability](#conditional-probability)

<a id="base-rate-fallacy"></a>

### Base rate fallacy

_Concept_

Ignoring how rare something is when interpreting evidence: assuming a 99%-sensitive test means a positive result is 99% likely to be right. When the base rate is low, most positives are false alarms, as in the disease example (1 in 6) and the wind farm's alarm precision (about 14%). Confusing P(positive | disease) with P(disease | positive) is known in law as the **prosecutor's fallacy**.

See also: [Base rate and false alarms](#base-rate-and-false-alarms)

<a id="prior-likelihood-and-posterior"></a>

### Prior, likelihood and posterior

_Concept_

Bayes' theorem with whole distributions, where θ stands for the parameters and x for the data:

**f(θ | x) = f(x | θ) · f(θ) / f(x)**

- **Prior f(θ):** what you believe about the parameters before seeing the data, as a distribution (e.g. "elections usually end close to 50/50"; "previous studies found an effect around 2%"). A **strong** (narrow) prior needs a lot of data to move; a **weak** (wide) prior lets the data dominate.
- **Likelihood f(x | θ):** how probable the observed data would be for each possible parameter value (e.g. a binomial likelihood for a poll). Less an assumption than a choice of how to parametrize the problem.
- **Marginal f(x):** the probability of the data, which doesn't depend on θ; it's just the constant that makes the posterior integrate to 1 (you can usually ignore it).
- **Posterior f(θ | x):** your updated beliefs after the data.

So **posterior ∝ likelihood × prior** (∝ = "proportional to": the shape is right, it just needs rescaling).

<a id="probability-mass-and-density-functions"></a>

### Probability mass and density functions

_Concept, SciPy_

Every random variable has a function describing where its probability is spread, usually written f. For a **discrete** variable, the **probability mass function (PMF)** gives actual probabilities: P(X = x) = f(x). For a **continuous** variable, the **probability density function (PDF)** gives a density, not a probability; probabilities are **areas** under the curve, and a density can even exceed 1.

```python
from scipy import stats
stats.binom.pmf(55, 100, 0.5)        # 0.0485 = P(exactly 55 heads in 100 fair flips)
stats.norm.pdf(0)                    # 0.3989 = a density at 0, not a probability
stats.norm.cdf(1) - stats.norm.cdf(-1)   # 0.683 = P(-1 < Z < 1), an area
```

See also: [Normal distribution (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#normal-gaussian-distribution)

<a id="conjugate-prior-beta-binomial"></a>

### Conjugate prior (Beta–binomial)

_Concept, SciPy_

A prior that, combined with a particular likelihood, gives a posterior in the **same family**, so the posterior has a formula and no simulation is needed. The standard example: a **Beta(a, b)** prior on a proportion plus binomial data (s successes, f failures) gives a **Beta(a + s, b + f)** posterior. The **Beta distribution** lives between 0 and 1, which makes it a natural prior for a rate or probability; Beta(1, 1) is flat, and Beta(50, 50) is concentrated near 0.5.

A poll of 100 voters with 55 for candidate A:

```python
for name, a, b in [("flat prior", 1, 1), ("'close to 50/50' prior", 50, 50)]:
    post = stats.beta(a + 55, b + 45)
    lo, hi = post.ppf([0.025, 0.975])
    print(f"{name}: mean {post.mean():.3f}, 95% interval {lo:.3f}-{hi:.3f}, P(A wins) {1 - post.cdf(0.5):.2f}")
# flat prior:             mean 0.549, interval 0.452-0.644, P(A wins) 0.84
# 'close to 50/50' prior: mean 0.525, interval 0.456-0.594, P(A wins) 0.76
```

The prior tempers an over-confident reading of one poll: the estimate shrinks from 0.549 towards 0.5, and the chance that A is ahead falls from 84% to 76%.

<a id="credible-interval"></a>

### Credible interval

_Concept_

A Bayesian interval that is a direct probability statement about the parameter: "there's a 95% probability the true support is between 45.2% and 64.4%". A frequentist **confidence interval** can't be read that way, because the true value is fixed; its 95% describes the method's long-run success rate over repeated samples. With lots of data and weak priors the two intervals are often numerically similar.

> **Watch out:** Frequentists report uncertainty too, not just a point estimate (standard errors, confidence intervals); the difference is the interpretation.

<a id="grid-approximation"></a>

### Grid approximation

_NumPy, SciPy_

Computing a posterior without a formula: evaluate likelihood × prior on a fine grid of parameter values, then divide by the total so it sums to 1. It shows that the "ignored denominator" is just this rescaling, and reproduces the conjugate answer exactly.

```python
grid = np.linspace(0, 1, 1001)
unnormalized = stats.binom.pmf(55, 100, grid) * stats.beta.pdf(grid, 50, 50)
posterior = unnormalized / unnormalized.sum()
np.sum(grid * posterior)             # 0.525, same as the formula
```

Grids work for one or two parameters; with many parameters the number of grid points explodes, which is where MCMC comes in.

<a id="markov-chain-monte-carlo-mcmc"></a>

### Markov Chain Monte Carlo (MCMC)

_Concept_

A family of algorithms that draw random samples whose distribution matches the posterior, **without ever computing f(x)**. Except in simple conjugate cases, posteriors can't be calculated by hand, so in practice you sample. With thousands of posterior samples you can compute anything: the mean, credible intervals, the probability a parameter exceeds a threshold. It builds directly on the Monte Carlo ideas from the simulation section.

See also: [Monte Carlo simulation](#monte-carlo-simulation), [PyMC](#pymc)

<a id="pymc"></a>

### PyMC

_Python_

The leading Python library for Bayesian modelling with MCMC (install with `pip install pymc`). You write the prior and likelihood, and PyMC samples the posterior. The poll model:

```python
import pymc as pm

with pm.Model():
    p = pm.Beta("p", alpha=50, beta=50)                    # prior: close to 50/50
    pm.Binomial("votes", n=100, p=p, observed=55)          # likelihood: 55 of 100
    trace = pm.sample(2000, random_seed=42)
trace.posterior["p"].mean().item()                          # ≈ 0.525, matching the formula
```

Alternatives include Stan (via CmdStanPy) and NumPyro.

> **Watch out:** Older tutorials refer to "PyMC3". The library was renamed **PyMC** from version 4, and some of its API changed, so follow the current documentation. On some restricted or single-core environments sampling can fail while detecting CPU cores; pass `chains=2, cores=1` to `pm.sample`.

<a id="bayesian-election-forecasting"></a>

### Bayesian election forecasting

_Concept_

Elections happen once, which makes them a natural fit for Bayesian methods. Nate Silver's forecasts (formerly at FiveThirtyEight, now in his Silver Bulletin newsletter) are the deck's example. His **prior** uses a fact people forget early in an election cycle: elections usually end up close to 50/50, which stops a forecast overreacting to one dramatic early poll. Poll data form the **likelihood**, and polls are graded by quality, acting like weights (another prior assumption). Schematically: f(p | x) ∝ f(x | p, θ) · f(p) · f(θ), where p is the probability candidate A wins and θ the poll-quality ratings.

> **Watch out:** Judging a single probabilistic forecast is hard. In 2016 his model gave Trump about a 30% chance, higher than most rivals: the result was a surprise, but not proof the forecast was "wrong". FiveThirtyEight was closed by ABC News in 2025, and "Twitter" is now X.

<a id="bayesian-a-b-testing"></a>

### Bayesian A/B testing

_NumPy_

Comparing two conversion rates by sampling from each posterior. Page A converts 120 of 1,000 visitors and page B 150 of 1,000; with flat Beta(1, 1) priors:

```python
rng = np.random.default_rng(42)
a = rng.beta(1 + 120, 1 + 880, 100_000)        # posterior samples for page A's rate
b = rng.beta(1 + 150, 1 + 850, 100_000)        # posterior samples for page B's rate
(b > a).mean()                                 # P(B is better) ≈ 0.975
np.mean(b / a - 1)                             # expected relative lift ≈ 26%
```

The output answers the business question directly ("97.5% probability B converts better"), which a p-value doesn't.

See also: [A/B test (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#a-b-test)

---

<a id="s38-capstone-playbook"></a>

## 38. Capstone playbook

_How to use this glossary for your capstone: running the data science lifecycle end to end on your own, specific question. Each stage lists what to do, the entries to read, and the mistakes that are easiest to make. Four tested code templates (classification, regression, time series and clustering) give you a leak-free starting point to adapt to your data._

<a id="using-this-glossary-for-your-capstone"></a>

### Using this glossary for your capstone

_Concept_

Your capstone answers one specific question by following the **data science lifecycle**: frame the problem → get and understand the data → choose a validation strategy → build baselines → train and tune candidate models → evaluate honestly → interpret and explain → communicate the value → plan deployment and next steps. The stages below map each step to the sections of this glossary. It's a loop, not a line: findings at any stage can send you back to reframe the question or rethink the data, and that's normal.

A strong capstone shows **judgement** as much as technique: why this question, why this metric, why this model, how you know it isn't leaking or overfitting, and what the result means for the stakeholder.

See also: [Data science lifecycle](#data-science-lifecycle), [Modeling workflow (end to end)](#modeling-workflow-end-to-end)

<a id="stage-1-frame-the-question"></a>

### Stage 1: Frame the question

_Concept_

Write a **specific, measurable** problem statement before touching a model: what you'll predict or discover, for whom, the decision it supports, and how you'll know it worked. Decide the **problem type** (regression, classification, time series forecasting, clustering, recommendation) and the **success metrics**, both technical (e.g. recall ≥ 0.8) and business (e.g. fewer missed failures). Estimate the **value** (ROI) and check the **ethics**: whose data is it, and who's affected by mistakes?

Read: [Problem framing](#problem-framing), [Problem statement](#problem-statement), [SMART goals](#smart-goals), [Stakeholder](#stakeholder), [Success metrics and hypotheses](#success-metrics-and-hypotheses), [Framing the problem type early](#framing-the-problem-type-early), [Feasibility assessment](#feasibility-assessment), [Return on investment (ROI)](#return-on-investment-roi), [Ethical, legal and strategic implications](#ethical-legal-and-strategic-implications), [Framing checklist](#framing-checklist).

> **Watch out:** "Predict X" isn't a problem statement until it says why it matters and what counts as success. And with a rare outcome, a success target of "high accuracy" is meaningless: see [Base rate and false alarms](#base-rate-and-false-alarms).

<a id="stage-2-get-and-understand-the-data"></a>

### Stage 2: Get and understand the data

_Concept_

Identify the **target** column and each feature's type, clean it, and explore it (the Unit 2 EDA toolkit). Check class balance, missing values, outliers, correlations, and whether rows are independent or ordered in time. For every feature, ask: **would I know this value at the moment I make the prediction?** If not, it's leakage.

Read: [Types of variables](#types-of-variables), [Data leakage (in model evaluation)](#data-leakage-in-model-evaluation), [Imbalanced classes](#imbalanced-classes), [No multicollinearity (M)](#no-multicollinearity-m), [Time series](#time-series), and the [Unit 2 glossary](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD) for cleaning and EDA.

> **Watch out:** Columns that are consequences of the target (like `accept` and `enroll` when predicting applications, or `casual` + `registered` when predicting total rentals) make a model look brilliant and useless at the same time. Drop them before modelling.

<a id="stage-3-choose-the-split-and-validation-strategy"></a>

### Stage 3: Choose the split and validation strategy

_Concept_

Split **before** any fitting, so the test set represents genuinely unseen data. Use a stratified split for imbalanced classification, a chronological split for anything time-ordered, and group-aware splits when the same customer or patient appears in several rows. Tune with cross-validation on the training data only; look at the test set **once**, at the end.

Read: [Train-test split](#train-test-split), [test_size, stratify and shuffle](#test-size-stratify-and-shuffle), [k-fold cross-validation](#k-fold-cross-validation), [Other cross-validation schemes](#other-cross-validation-schemes), [Three-way split (validation set)](#three-way-split-validation-set), [Time series train/test split](#time-series-train-test-split).

> **Watch out:** Each time you change a model because of its test score, the test set leaks a little into your choices. Make decisions on cross-validation scores.

<a id="stage-4-build-baselines"></a>

### Stage 4: Build baselines

_Concept_

Every result needs a comparison. Before any real model, compute what you'd get with no model at all: the mean (regression), the majority class (classification) or the last value or last season (forecasting). Then build one simple, interpretable model (linear or logistic regression, a shallow tree). Report every later model against both.

Read: [Null model (baseline) for regression](#null-model-baseline-for-regression), [Null accuracy](#null-accuracy), [DummyClassifier](#dummyclassifier), [Naive and seasonal naive forecasts](#naive-and-seasonal-naive-forecasts).

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
DummyClassifier(strategy="most_frequent").fit(X_train, y_train).score(X_test, y_test)
DummyRegressor(strategy="mean").fit(X_train, y_train)
```

<a id="stage-5-choose-candidate-models"></a>

### Stage 5: Choose candidate models

_Concept_

Pick two to four candidates suited to your problem type and data size, ranging from interpretable to powerful (for tabular classification: logistic regression → random forest → gradient boosting). Use the [Choosing an approach by question type](#choosing-an-approach-by-question-type) and [Model selection cheat sheet](#model-selection-cheat-sheet) tables below. More complex isn't automatically better: with limited data a simpler model often generalizes best.

Read: [Bias–variance trade-off](#bias-variance-trade-off), [Fitting sine exercise](#fitting-sine-exercise), [Which model is best?](#which-model-is-best).

<a id="choosing-an-approach-by-question-type"></a>

### Choosing an approach by question type

_Concept_

| Your question looks like…                       | Problem type              | Start with                             | Then try                                                          | Headline metrics                                   | Baseline               |
| ----------------------------------------------- | ------------------------- | -------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------- | ---------------------- |
| "How much / how many…?"                         | Regression                | Linear regression                      | Ridge/Lasso, random forest, gradient boosting                     | RMSE or MAE, R²                                    | Predict the mean       |
| "Will it happen / which category…?"             | Classification            | Logistic regression                    | kNN, random forest, gradient boosting, SVM                        | F1, recall/precision, ROC AUC                      | Majority class         |
| "…and the event is rare"                        | Imbalanced classification | Logistic regression with class weights | Resampling in a pipeline, boosting                                | Recall, precision, F1, balanced accuracy, PR curve | Majority class         |
| "What will it be next week/month?"              | Time series forecasting   | Seasonal naive, ARIMA/SARIMA           | ARIMA with exogenous predictors, VAR, tree models on lag features | MAE (or RMSE), AIC for model choice                | Naive / seasonal naive |
| "How volatile/risky will it be?"                | Volatility forecasting    | GARCH(1, 1)                            | Other GARCH orders                                                | Forecast variance vs realized                      | Constant variance      |
| "What natural groups exist?"                    | Clustering                | k-means on scaled data                 | DBSCAN/HDBSCAN, agglomerative                                     | Silhouette, inertia (elbow), cluster profiles      | —                      |
| "Can I simplify these many features?"           | Dimensionality reduction  | PCA                                    | PCA inside a supervised pipeline                                  | Explained variance; downstream CV score            | All features           |
| "What should we recommend?"                     | Recommendation            | Item-based collaborative filtering     | Content-based, hybrid                                             | Sense-checked recommendations, hit rate            | Most popular items     |
| "Which version is better, and how sure are we?" | Experiment / Bayesian     | Bayesian A/B test                      | Frequentist test (Unit 2)                                         | P(B better), credible interval                     | —                      |
| "What happens if…?"                             | Simulation                | Monte Carlo                            | Discrete event, agent-based, system dynamics                      | Distribution of outcomes (percentiles)             | Current state          |

See also: [Regression vs classification tasks](#regression-vs-classification-tasks), [Choosing a classification metric](#choosing-a-classification-metric), [Choosing a regression metric](#choosing-a-regression-metric)

<a id="model-selection-cheat-sheet"></a>

### Model selection cheat sheet

_Concept, scikit-learn_

| Model                       | Task                     | Scale features?                  | Key hyperparameters                                | Interpretability            | Watch for                               |
| --------------------------- | ------------------------ | -------------------------------- | -------------------------------------------------- | --------------------------- | --------------------------------------- |
| Linear regression           | Regression               | No (yes to compare coefficients) | —                                                  | High                        | Assumptions (LINEM), outliers, leakage  |
| Ridge / Lasso / ElasticNet  | Regression               | **Yes**                          | `alpha`, `l1_ratio`                                | High                        | Choose alpha by CV                      |
| Logistic regression         | Classification           | Recommended                      | `C`, penalty                                       | High (odds ratios)          | Default L2 penalty, perfect separation  |
| kNN                         | Both                     | **Yes**                          | `n_neighbors`, `weights`, `p`                      | Medium                      | Slow on big data, many features         |
| Decision tree               | Both                     | No                               | `max_depth`, `min_samples_leaf`, `ccp_alpha`       | High                        | Overfits; unstable                      |
| Random forest               | Both                     | No                               | `n_estimators`, `max_features`, `min_samples_leaf` | Medium (importances)        | Regressor default uses all features     |
| Gradient boosting / XGBoost | Both                     | No                               | `learning_rate`, `n_estimators`, `max_depth`       | Low–medium (SHAP)           | Overfits if run too long                |
| SVM (`SVC`)                 | Classification           | **Yes**                          | `C`, `kernel`, `gamma`                             | Low                         | Slow on very large data                 |
| k-means                     | Clustering               | **Yes**                          | `n_clusters`, `n_init`                             | Medium (centroids)          | Round, similar-sized clusters only      |
| DBSCAN / HDBSCAN            | Clustering               | **Yes**                          | `eps`, `min_samples` / `min_cluster_size`          | Medium                      | Sensitive to eps; varying density       |
| Agglomerative               | Clustering               | **Yes**                          | `n_clusters`, `linkage`                            | Medium (dendrogram)         | Slow on large data                      |
| PCA                         | Dimensionality reduction | **Yes**                          | `n_components`                                     | Low (components are blends) | Ignores the target                      |
| ARIMA / SARIMA              | Forecasting              | —                                | `order`, `seasonal_order`                          | Medium                      | Needs stationarity; chronological split |
| GARCH                       | Volatility               | Rescale returns                  | `p`, `q`                                           | Medium                      | Needs the `arch` package                |
| VAR                         | Multivariate forecasting | —                                | `maxlags`, `ic`                                    | Medium                      | Parameters grow fast                    |

See also: [Scaling for distance-based models](#scaling-for-distance-based-models), [Standardize before regularizing](#standardize-before-regularizing)

<a id="stage-6-build-a-pipeline-and-tune"></a>

### Stage 6: Build a pipeline and tune

_Concept, scikit-learn_

Put **all** preprocessing (imputation, scaling, encoding, resampling) and the model into one pipeline, then tune it with `GridSearchCV` (or `RandomizedSearchCV` for big grids) on the training data. Choose `scoring` to match your headline metric. Check whether the best values sit at the edge of your grid, and treat tiny CV differences as ties.

Read: [Pipelines (in tuning)](#pipelines-in-tuning), [Pipeline with ColumnTransformer](#pipeline-with-columntransformer), [Pipeline + GridSearchCV](#pipeline-gridsearchcv), [step\_\_parameter naming](#step-parameter-naming), [Choosing a scoring metric for tuning](#choosing-a-scoring-metric-for-tuning), [Reading grid search results](#reading-grid-search-results), [imbalanced-learn pipeline](#imbalanced-learn-pipeline).

> **Watch out:** Fitting a scaler, imputer, encoder or resampler on all the data before splitting is the most common source of leakage. Inside a pipeline it can't happen.

<a id="stage-7-evaluate-honestly"></a>

### Stage 7: Evaluate honestly

_Concept_

Evaluate the chosen model **once** on the test set, with the metrics you committed to in Stage 1, next to the baselines. For classifiers, show the confusion matrix and consider the threshold; for regressors, show the residual plot and errors in real units; for forecasts, compare with naive baselines and check that residuals look like white noise. Compare models on the **same scale** and the same data.

Read: [Reading train vs test scores](#reading-train-vs-test-scores), [Confusion matrix](#confusion-matrix), [Classification threshold](#classification-threshold), [Precision–recall curve](#precision-recall-curve), [Residual plot](#residual-plot), [Comparing models on the same scale](#comparing-models-on-the-same-scale), [Ljung–Box test](#ljung-box-test), [Resampling is a trade-off](#resampling-is-a-trade-off).

> **Watch out:** A result that looks too good to be true (99% accuracy, R² of 0.98) usually is. Check for leakage, perfect separation, duplicates, or a time-series split that let the future in.

<a id="stage-8-interpret-and-explain"></a>

### Stage 8: Interpret and explain

_Concept_

Explain **what drives the predictions** (global: permutation importance, coefficients, partial dependence) and **why a particular case was predicted as it was** (local: SHAP). Translate effects into stakeholder language with units ("each extra $1,000 of radio spend is associated with about 189 more units sold"), and remember these are associations, not causes.

Read: [Interpreting coefficients](#interpreting-coefficients), [Interpreting logistic regression coefficients](#interpreting-logistic-regression-coefficients), [Permutation importance](#permutation-importance), [Partial dependence plot (PDP)](#partial-dependence-plot-pdp), [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations), [Global vs local explanations](#global-vs-local-explanations), [Proxy variable](#proxy-variable).

<a id="stage-9-communicate-the-value"></a>

### Stage 9: Communicate the value

_Concept_

Tell the story the stakeholder needs: the question and why it matters → the data and approach (briefly) → results against the baseline **in business terms** → what drives the predictions → limitations and risks → recommendation and next steps. Quantify value where you can (savings, risk-adjusted ROI, with a range), and be explicit about what the model **can't** do.

A presentation outline that works: (1) problem and stakeholder; (2) data and key EDA findings; (3) approach and validation strategy; (4) results vs baseline; (5) interpretation; (6) value and recommendations; (7) limitations, ethics and next steps.

Read: [Return on investment (ROI)](#return-on-investment-roi), [Risk-adjusted ROI](#risk-adjusted-roi), [Sensitivity analysis](#sensitivity-analysis), [Why explainability matters](#why-explainability-matters).

<a id="stage-10-deploy-monitor-and-next-steps"></a>

### Stage 10: Deploy, monitor and next steps

_Concept, Python_

After choosing a model, **refit it on all the training data** (or all data, for forecasts), save the whole pipeline, and describe how it would be used and monitored: what triggers retraining, which metrics to track, what happens when the data drift.

```python
import joblib
joblib.dump(best.best_estimator_, "capstone_model.joblib")     # saves preprocessing + model
model = joblib.load("capstone_model.joblib")
model.predict(new_data)                                          # same columns as training X
```

Read: [Refitting for production](#refitting-for-production), [Data science lifecycle](#data-science-lifecycle), [Pipeline memory and debugging](#pipeline-memory-and-debugging).

> **Watch out:** A saved model should be reloaded with the same library versions it was trained with; record them (see [Reproducibility habits](#reproducibility-habits)).

<a id="capstone-template-classification"></a>

### Capstone template: classification

_scikit-learn_

A leak-free starting point for tabular classification (tested on the Titanic data: baseline F1 0.0; tuned random forest CV F1 0.741, test F1 0.71, ROC AUC 0.83). Replace the file name and target, and make sure `y` is 0/1 with 1 as the class you care about.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

df = pd.read_csv("data/your_data.csv")
target = "your_target"                               # 0/1, where 1 = the class you care about
X, y = df.drop(columns=[target]), df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("baseline F1:", f1_score(y_test, baseline.predict(X_test)))

num_cols = X.select_dtypes(include=["number", "bool"]).columns   # booleans count as numeric
cat_cols = X.columns.difference(num_cols)
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
candidates = {
    "logistic regression": (LogisticRegression(max_iter=1000),
                            {"model__C": np.logspace(-3, 2, 6)}),
    "random forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                      {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 5]}),
}
searches = {}
for name, (model, grid) in candidates.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    searches[name] = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1).fit(X_train, y_train)
    print(f"{name}: CV F1 {searches[name].best_score_:.3f} with {searches[name].best_params_}")

best = max(searches.values(), key=lambda gs: gs.best_score_)      # choose on CV, not test
y_pred = best.predict(X_test)                                      # the one final test check
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best.predict_proba(X_test)[:, 1]))
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test)

imp = permutation_importance(best, X_test, y_test, scoring="f1", n_repeats=10, random_state=42)
pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False).head(10)
```

> **Watch out:** `select_dtypes("number")` excludes boolean columns; if they fall into the text group, the imputer can fail with "could not convert string to float". Including `"bool"` with the numbers avoids it. Change `scoring` (e.g. to `"recall"` or `"roc_auc"`) to match your headline metric, and add an `imblearn` sampler or `class_weight="balanced"` if the positive class is rare.

<a id="capstone-template-regression"></a>

### Capstone template: regression

_scikit-learn_

The same structure for a numeric target (tested on scikit-learn's diabetes data: baseline RMSE 73.2; tuned ridge test RMSE 53.5, MAE 43.0, R² 0.46). It continues from the classification template's imports; for mixed column types, reuse that template's `ColumnTransformer` as the first pipeline step.

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X, y = df.drop(columns=[target]), df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
print("baseline RMSE:", metrics.root_mean_squared_error(y_test, baseline.predict(X_test)))

candidates = {
    "ridge": (Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
              {"model__alpha": np.logspace(-2, 3, 12)}),
    "random forest": (Pipeline([("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
                      {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 5, 10]}),
}
searches = {name: GridSearchCV(pipe, grid, cv=5, scoring="neg_root_mean_squared_error",
                               n_jobs=-1).fit(X_train, y_train)
            for name, (pipe, grid) in candidates.items()}
for name, gs in searches.items():
    print(f"{name}: CV RMSE {-gs.best_score_:.1f} with {gs.best_params_}")   # flip the sign back

best = max(searches.values(), key=lambda gs: gs.best_score_)
y_pred = best.predict(X_test)
print("RMSE", metrics.root_mean_squared_error(y_test, y_pred),
      "MAE", metrics.mean_absolute_error(y_test, y_pred), "R²", metrics.r2_score(y_test, y_pred))
plt.scatter(y_pred, y_test - y_pred, s=10); plt.axhline(0, color="orange")   # residual plot
```

> **Watch out:** If you log-transform the target, back-transform predictions with `np.exp` before computing RMSE or MAE, so every model is compared in the same units.

<a id="capstone-template-time-series-forecasting"></a>

### Capstone template: time series forecasting

_statsmodels_

For a regularly spaced series with a datetime index (tested on monthly CO₂ data: naive MAE 2.42, seasonal naive 1.88, SARIMA 0.33, Ljung–Box p = 0.97). Set `season` to your cycle length (12 for monthly data, 4 for quarterly, 7 for daily data with a weekly pattern). It uses NumPy and pandas from the classification template's imports.

```python
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

y = df.set_index(pd.to_datetime(df["date"]))["value"].resample("MS").mean()   # monthly, regular
season, h = 12, 24                                       # cycle length; hold out the last 24 periods
train, test = y.iloc[:-h], y.iloc[-h:]                   # chronological split: never shuffle

mae = lambda forecast: np.mean(np.abs(test.values - np.asarray(forecast)))
print("naive MAE:", mae(np.repeat(train.iloc[-1], h)))
print("seasonal naive MAE:", mae(np.tile(train.iloc[-season:].values, h // season)))

results = []
for p, q in itertools.product(range(3), range(3)):       # choose orders by AIC on training data
    fit = ARIMA(train, order=(p, 1, q), seasonal_order=(0, 1, 1, season)).fit()
    results.append(((p, 1, q), fit.aic))
order = min(results, key=lambda r: r[1])[0]
fit = ARIMA(train, order=order, seasonal_order=(0, 1, 1, season)).fit()
print("SARIMA MAE:", mae(fit.forecast(h)))
print("Ljung–Box p:", acorr_ljungbox(fit.resid[season + 1:], lags=[season])["lb_pvalue"].iloc[0])

final = ARIMA(y, order=order, seasonal_order=(0, 1, 1, season)).fit()   # refit on everything
final.forecast(12)                                                         # the real forecast
```

> **Watch out:** Check stationarity and look at the ACF/PACF before trusting any order; take logs first if the swings grow with the level, and back-transform before computing MAE.

<a id="capstone-template-clustering"></a>

### Capstone template: clustering

_scikit-learn_

Segmentation without labels (tested on iris: silhouette prefers k = 2, the elbow suggests 3; three clusters of 53, 50 and 47 give interpretable profiles). The output that matters is the **profile table**, which turns cluster numbers into segments a stakeholder can act on. It reuses `StandardScaler` and pandas from the classification template's imports.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

features = df.select_dtypes("number")
X_sc = StandardScaler().fit_transform(features)

scores = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_sc)
    scores.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X_sc, km.labels_)})
pd.DataFrame(scores)                                   # look for an elbow and a good silhouette

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_sc)
segments = features.assign(cluster=km.labels_)
segments["cluster"].value_counts()                     # cluster sizes
segments.groupby("cluster").mean().round(2)            # the profiles: name each segment
```

> **Watch out:** Scores only guide the choice of k; combine them with plots and domain knowledge, and try DBSCAN or agglomerative clustering if the groups aren't round blobs.

<a id="common-capstone-mistakes-checklist"></a>

### Common capstone mistakes checklist

_Concept_

Before you submit, check each of these; they're the easiest mistakes to make and the first things a reviewer looks for:

1. **Leakage:** a feature that wouldn't be known at prediction time, or preprocessing fitted before the split. ([Data leakage (in model evaluation)](#data-leakage-in-model-evaluation))
2. **Tuning on the test set:** choosing k, alpha or a model by test score instead of cross-validation. ([Three-way split (validation set)](#three-way-split-validation-set))
3. **No baseline:** reporting a score without the null model next to it. ([Null accuracy](#null-accuracy), [Null model (baseline) for regression](#null-model-baseline-for-regression))
4. **Accuracy on imbalanced data:** use recall, precision, F1, balanced accuracy or PR AUC. ([Imbalanced classes](#imbalanced-classes))
5. **Shuffling time series:** split chronologically and compare with naive forecasts. ([Time series train/test split](#time-series-train-test-split))
6. **Unscaled distance or penalty models:** kNN, SVM, k-means, PCA, ridge and lasso need scaling. ([Scaling for distance-based models](#scaling-for-distance-based-models))
7. **Evaluating on the wrong data:** passing unscaled test data to a model trained on scaled data (pipelines prevent it).
8. **Comparing metrics on different scales:** log-target vs raw-target R², or training vs test scores. ([Comparing models on the same scale](#comparing-models-on-the-same-scale))
9. **Over-reading coefficients:** comparing unscaled coefficients, ignoring multicollinearity, or saying "times more likely" for odds ratios. ([Coefficients are not importance](#coefficients-are-not-importance), [Interpreting logistic regression coefficients](#interpreting-logistic-regression-coefficients))
10. **Causal language for associations:** a model shows what predicts, not what causes. ([Correlation does not imply causation (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#correlation-does-not-imply-causation))
11. **Ignoring the threshold:** 0.5 is a default, not a decision. ([Classification threshold](#classification-threshold))
12. **Untested AI-generated code:** test it against inputs with known answers. ([Testing AI-generated code](#testing-ai-generated-code))
13. **Results that can't be reproduced:** no random seeds, no recorded versions. ([Reproducibility habits](#reproducibility-habits))

<a id="library-version-changes-to-know"></a>

### Library version changes to know

_Concept, Python_

Libraries evolve: parameters are renamed, defaults change and old functions are removed, and tutorials online often use older syntax. When code from a tutorial or older notebook fails, check the documentation for the version you have installed. The changes you're most likely to meet in this unit:

| Library      | Older code                                      | Current code                                                                | When it changed                       |
| ------------ | ----------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------- |
| scikit-learn | `LogisticRegression(penalty='none')`            | `penalty=None`; from 1.8, `C=np.inf`                                        | 1.2; `penalty` deprecated in 1.8      |
| scikit-learn | `penalty='l1'` / `'l2'`                         | `l1_ratio=1` / `l1_ratio=0`                                                 | Deprecated 1.8 (removal planned 1.10) |
| scikit-learn | `LogisticRegression(multi_class=...)`           | Default multinomial, or `OneVsRestClassifier`                               | Deprecated 1.5, since removed         |
| scikit-learn | `mean_squared_error(..., squared=False)`        | `root_mean_squared_error(...)`                                              | Added 1.4; `squared` since removed    |
| scikit-learn | `plot_roc_curve`, `plot_confusion_matrix`       | `RocCurveDisplay.from_estimator`, `ConfusionMatrixDisplay.from_estimator`   | Removed 1.2                           |
| scikit-learn | `base_estimator=` (bagging, AdaBoost)           | `estimator=`                                                                | Renamed 1.2, old name removed 1.4     |
| scikit-learn | `AdaBoostClassifier(algorithm="SAMME.R")`       | Omit `algorithm` (SAMME is used)                                            | Deprecated 1.4, since removed         |
| scikit-learn | `KMeans(n_init=10)` was the default             | `n_init="auto"` (one run with k-means++)                                    | 1.4: set `n_init=10` for stability    |
| scikit-learn | Random forest `n_estimators=10` default         | 100                                                                         | 0.22                                  |
| scikit-learn | (not available)                                 | `HDBSCAN` in `sklearn.cluster`                                              | Added 1.3                             |
| pandas       | `resample("M")`, `"Q"`, `"Y"`, `freq="Q-DEC"`   | `"ME"`, `"QE"`, `"YE"`, `"QE-DEC"`                                          | Deprecated 2.2, errors in 3.0         |
| pandas       | `series[0]` on a labelled index                 | `series.iloc[0]`                                                            | Deprecated 2.1, `KeyError` in 3.0     |
| pandas       | `get_dummies` gave 0/1 integers                 | Gives `True`/`False`; pass `dtype=int`                                      | 2.0                                   |
| pandas       | `str.replace(".", "_")` treated "." as a regex  | Literal by default (`regex=False`)                                          | 2.0                                   |
| pandas       | Text columns stored as `object`                 | Dedicated `str` dtype by default                                            | 3.0                                   |
| statsmodels  | `from statsmodels.tsa.arima_model import ARIMA` | `from statsmodels.tsa.arima.model import ARIMA`                             | Old class now only raises an error    |
| statsmodels  | `adfuller(...)` returns a tuple                 | Warns that its return format will change; p-value is still the second value | 0.15                                  |
| PyMC         | `import pymc3 as pm`                            | `import pymc as pm`                                                         | Renamed at version 4                  |
| Google GenAI | `google-generativeai` package                   | `google-genai` (`from google import genai`)                                 | 2025                                  |

```python
import sklearn, pandas, statsmodels
print(sklearn.__version__, pandas.__version__, statsmodels.__version__)   # what you're running
```

> **Watch out:** Deprecation warnings are an early warning that code will break in a future version. Read them, and update the code rather than silencing them.

<a id="reproducibility-habits"></a>

### Reproducibility habits

_Python_

Make your capstone rerunnable by someone else (or you, in three months): set `random_state` on splits, models and searches; restart the kernel and run the notebook top to bottom before submitting; keep data loading, cleaning, modelling and evaluation in clear sections; record library versions; and save the fitted pipeline, not just the model.

```python
# pip freeze > requirements.txt     (in a terminal: records exact versions)
import sklearn, pandas, numpy
print({"sklearn": sklearn.__version__, "pandas": pandas.__version__, "numpy": numpy.__version__})
```

See also: [Random seed (Unit 2)](https://github.com/CharleyHananiaGA/DSB-Unit-02/blob/main/Readme.MD#random-seed), [Library version changes to know](#library-version-changes-to-know)

---

<a id="a-z-index"></a>

## A–Z index

**A**

- [Accuracy](#accuracy)
- [ACF plot](#acf-plot)
- [AdaBoost](#adaboost)
- [AdaBoost from scratch](#adaboost-from-scratch)
- [AdaBoostClassifier](#adaboostclassifier)
- [Adam, RMSProp and IRLS](#adam-rmsprop-and-irls)
- [ADASYN](#adasyn)
- [Addition rule (unions)](#addition-rule-unions)
- [Adjusted Rand index (ARI)](#adjusted-rand-index-ari)
- [Adjusted R²](#adjusted-r2)
- [Agent-based modelling (ABM)](#agent-based-modelling-abm)
- [Agglomerative clustering](#agglomerative-clustering)
- [AI, machine learning and deep learning](#ai-machine-learning-and-deep-learning)
- [AIC (Akaike Information Criterion)](#aic-akaike-information-criterion)
- [alpha (regularization strength)](#alpha-regularization-strength)
- [Analytical model vs simulation](#analytical-model-vs-simulation)
- [ARCH(p)](#arch-p)
- [ARIMA with exog (GLS in Python)](#arima-with-exog-gls-in-python)
- [ARIMA(p, d, q)](#arima-p-d-q)
- [ARMA(p, q)](#arma-p-q)
- [AUC (ROC AUC)](#auc-roc-auc)
- [Augmented Dickey–Fuller (ADF) test](#augmented-dickey-fuller-adf-test)
- [Autocorrelation](#autocorrelation)
- [Autoregression (AR)](#autoregression-ar)

**B**

- [Bagging (bootstrap aggregating)](#bagging-bootstrap-aggregating)
- [Bagging vs boosting](#bagging-vs-boosting)
- [BaggingClassifier](#baggingclassifier)
- [Balanced accuracy](#balanced-accuracy)
- [Balanced model](#balanced-model)
- [Base rate and false alarms](#base-rate-and-false-alarms)
- [Base rate fallacy](#base-rate-fallacy)
- [Baseline costs and projected savings](#baseline-costs-and-projected-savings)
- [Bayes' theorem](#bayes-theorem)
- [Bayesian A/B testing](#bayesian-a-b-testing)
- [Bayesian election forecasting](#bayesian-election-forecasting)
- [Bayesian statistics](#bayesian-statistics)
- [Benefits of simulation](#benefits-of-simulation)
- [best*params*, best*score* and best*estimator*](#best-params-best-score-and-best-estimator)
- [Bias](#bias)
- [Bias–variance decomposition](#bias-variance-decomposition)
- [Bias–variance trade-off](#bias-variance-trade-off)
- [Big data considerations](#big-data-considerations)
- [Binary and multiclass classification](#binary-and-multiclass-classification)
- [Birch and Affinity Propagation](#birch-and-affinity-propagation)
- [Black-box model](#black-box-model)
- [Boosting](#boosting)
- [Boosting pros and cons](#boosting-pros-and-cons)
- [Bootstrap sample](#bootstrap-sample)
- [Bootstrapping](#bootstrapping)
- [Build, buy or partner](#build-buy-or-partner)
- [Bullseye diagram](#bullseye-diagram)

**C**

- [C (inverse regularization strength)](#c-inverse-regularization-strength)
- [C (SVM)](#c-svm)
- [Calibration](#calibration)
- [Calling an LLM from Python](#calling-an-llm-from-python)
- [Capstone template: classification](#capstone-template-classification)
- [Capstone template: clustering](#capstone-template-clustering)
- [Capstone template: regression](#capstone-template-regression)
- [Capstone template: time series forecasting](#capstone-template-time-series-forecasting)
- [CART algorithm](#cart-algorithm)
- [CatBoost](#catboost)
- [Categorical features in kNN](#categorical-features-in-knn)
- [ccp_alpha (cost-complexity pruning)](#ccp-alpha-cost-complexity-pruning)
- [Centroid](#centroid)
- [Centroid initialization](#centroid-initialization)
- [Changing the threshold for imbalanced data](#changing-the-threshold-for-imbalanced-data)
- [Chat interface vs API](#chat-interface-vs-api)
- [Choosing a classification metric](#choosing-a-classification-metric)
- [Choosing a regression metric](#choosing-a-regression-metric)
- [Choosing a scoring metric for tuning](#choosing-a-scoring-metric-for-tuning)
- [Choosing a simulation approach](#choosing-a-simulation-approach)
- [Choosing an approach by question type](#choosing-an-approach-by-question-type)
- [Choosing k](#choosing-k)
- [Choosing n_components with grid search](#choosing-n-components-with-grid-search)
- [Choosing p, d and q](#choosing-p-d-and-q)
- [Choosing the number of clusters](#choosing-the-number-of-clusters)
- [Choosing α](#choosing)
- [Class weights](#class-weights)
- [Classical decomposition](#classical-decomposition)
- [Classification](#classification)
- [Classification threshold](#classification-threshold)
- [classification_report()](#classification-report)
- [Closed-form solution](#closed-form-solution)
- [Cluster labels are arbitrary](#cluster-labels-are-arbitrary)
- [Clustering](#clustering)
- [Clustering algorithms](#clustering-algorithms)
- [.coef* and .intercept*](#coef-and-intercept)
- [Coefficients are not importance](#coefficients-are-not-importance)
- [Coffee shop queue simulation](#coffee-shop-queue-simulation)
- [Collaborative filtering](#collaborative-filtering)
- [Common capstone mistakes checklist](#common-capstone-mistakes-checklist)
- [Comparing AdaBoost, gradient boosting and XGBoost](#comparing-adaboost-gradient-boosting-and-xgboost)
- [Comparing clustering algorithms](#comparing-clustering-algorithms)
- [Comparing models on the same scale](#comparing-models-on-the-same-scale)
- [Complement (Aᶜ)](#complement-a)
- [Complement rule](#complement-rule)
- [Conditional heteroscedasticity](#conditional-heteroscedasticity)
- [Conditional probability](#conditional-probability)
- [Confusion matrix](#confusion-matrix)
- [ConfusionMatrixDisplay](#confusionmatrixdisplay)
- [Conjugate prior (Beta–binomial)](#conjugate-prior-beta-binomial)
- [Content-based, knowledge-based and hybrid recommenders](#content-based-knowledge-based-and-hybrid-recommenders)
- [Context window](#context-window)
- [Convergence and tolerance](#convergence-and-tolerance)
- [Convex function](#convex-function)
- [Core, border and noise points](#core-border-and-noise-points)
- [Cosine similarity](#cosine-similarity)
- [Counterfactual explanation](#counterfactual-explanation)
- [Covariance matrix, eigenvectors and eigenvalues](#covariance-matrix-eigenvectors-and-eigenvalues)
- [Credible interval](#credible-interval)
- [Cross-validation pros and cons](#cross-validation-pros-and-cons)
- [cross_val_score()](#cross-val-score)
- [cv*results*](#cv-results)

**D**

- [Data leakage (in model evaluation)](#data-leakage-in-model-evaluation)
- [Data minimization and anonymization](#data-minimization-and-anonymization)
- [Data privacy with GenAI tools](#data-privacy-with-genai-tools)
- [Data science lifecycle](#data-science-lifecycle)
- [Date-based slicing](#date-based-slicing)
- [Datetime index](#datetime-index)
- [DBSCAN](#dbscan)
- [DBSCAN pros and cons](#dbscan-pros-and-cons)
- [De Morgan's laws](#de-morgan-s-laws)
- [Decision boundary](#decision-boundary)
- [Decision stump](#decision-stump)
- [Decision tree](#decision-tree)
- [Decision tree pros and cons](#decision-tree-pros-and-cons)
- [decision_function()](#decision-function)
- [DecisionTreeClassifier and DecisionTreeRegressor](#decisiontreeclassifier-and-decisiontreeregressor)
- [Dendrogram](#dendrogram)
- [Derivative](#derivative)
- [Design matrix](#design-matrix)
- [Differencing and .diff()](#differencing-and-diff)
- [Dimensionality reduction](#dimensionality-reduction)
- [Dimensionality-reduced reconstruction](#dimensionality-reduced-reconstruction)
- [Discrete event simulation (DES)](#discrete-event-simulation-des)
- [Disjoint (mutually exclusive) events](#disjoint-mutually-exclusive-events)
- [Distance metric (the four rules)](#distance-metric-the-four-rules)
- [Distance metrics (Euclidean, Manhattan, Minkowski)](#distance-metrics-euclidean-manhattan-minkowski)
- [Dummy variables in regression](#dummy-variables-in-regression)
- [DummyClassifier](#dummyclassifier)
- [Durbin–Watson statistic](#durbin-watson-statistic)

**E**

- [ElasticNet](#elasticnet)
- [Elbow method](#elbow-method)
- [Embedding](#embedding)
- [Emergence](#emergence)
- [Ensemble model](#ensemble-model)
- [Entropy and cross-entropy](#entropy-and-cross-entropy)
- [eps and min_samples](#eps-and-min-samples)
- [Equal variance / homoscedasticity (E)](#equal-variance-homoscedasticity-e)
- [Error term (ε)](#error-term)
- [Ethical concerns with GenAI](#ethical-concerns-with-genai)
- [Ethical, legal and strategic implications](#ethical-legal-and-strategic-implications)
- [Euclidean distance](#euclidean-distance)
- [Evaluating a split with Gini](#evaluating-a-split-with-gini)
- [Event](#event)
- [Exogenous and endogenous variables](#exogenous-and-endogenous-variables)
- [Experiment](#experiment)
- [Explained variance and explained variance ratio](#explained-variance-and-explained-variance-ratio)
- [export_text()](#export-text)
- [External cluster metrics](#external-cluster-metrics)
- [Extrapolation](#extrapolation)

**F**

- [F-beta score](#f-beta-score)
- [F1 score](#f1-score)
- [False positive rate (FPR)](#false-positive-rate-fpr)
- [Feasibility assessment](#feasibility-assessment)
- [Feature importance](#feature-importance)
- [Feature importance in ensembles](#feature-importance-in-ensembles)
- [Feature scale in linear regression](#feature-scale-in-linear-regression)
- [Feature scaling and gradient descent](#feature-scaling-and-gradient-descent)
- [Feedback loop](#feedback-loop)
- [fit_regularized() in statsmodels](#fit-regularized-in-statsmodels)
- [fit_resample()](#fit-resample)
- [Fitting sine exercise](#fitting-sine-exercise)
- [Five key considerations for framing](#five-key-considerations-for-framing)
- [Flat curves (vanishing steps)](#flat-curves-vanishing-steps)
- [Fold](#fold)
- [Four pillars of simulation](#four-pillars-of-simulation)
- [Framing checklist](#framing-checklist)
- [Framing the problem type early](#framing-the-problem-type-early)
- [Frequentist vs Bayesian](#frequentist-vs-bayesian)
- [From coefficients to probability](#from-coefficients-to-probability)

**G**

- [Gambler's fallacy](#gambler-s-fallacy)
- [gamma](#gamma)
- [GARCH(p, q)](#garch-p-q)
- [GenAI tools for data scientists](#genai-tools-for-data-scientists)
- [Generalization](#generalization)
- [Generalized least squares (GLS)](#generalized-least-squares-gls)
- [Generation mode vs analysis mode](#generation-mode-vs-analysis-mode)
- [Generative AI (GenAI)](#generative-ai-genai)
- [Gini impurity](#gini-impurity)
- [Global vs local explanations](#global-vs-local-explanations)
- [Gradient boosting](#gradient-boosting)
- [Gradient descent](#gradient-descent)
- [Gradient descent for OLS](#gradient-descent-for-ols)
- [GradientBoostingClassifier](#gradientboostingclassifier)
- [Greedy algorithm](#greedy-algorithm)
- [Grid approximation](#grid-approximation)
- [Grid search](#grid-search)
- [GridSearchCV](#gridsearchcv)

**H**

- [Hallucination](#hallucination)
- [Hat notation (ŷ, β̂)](#hat-notation)
- [HDBSCAN](#hdbscan)
- [Hierarchical clustering](#hierarchical-clustering)
- [Holdout method](#holdout-method)
- [How GenAI output goes wrong](#how-genai-output-goes-wrong)
- [How LLMs are trained](#how-llms-are-trained)
- [Hyperparameter](#hyperparameter)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Hyperparameter vs parameter](#hyperparameter-vs-parameter)

**I**

- [Imbalanced classes](#imbalanced-classes)
- [Imbalanced data checklist](#imbalanced-data-checklist)
- [imbalanced-learn pipeline](#imbalanced-learn-pipeline)
- [Independence (I)](#independence-i)
- [Individual conditional expectation (ICE)](#individual-conditional-expectation-ice)
- [Inertia (within-cluster sum of squares)](#inertia-within-cluster-sum-of-squares)
- [Influence/interest grid](#influence-interest-grid)
- [Information criteria (AIC, BIC, Mallows's Cp)](#information-criteria-aic-bic-mallows-s-cp)
- [Ingredients of a supervised model](#ingredients-of-a-supervised-model)
- [Interaction term](#interaction-term)
- [Intercept (β₀)](#intercept)
- [Interpretability](#interpretability)
- [Interpreting coefficients](#interpreting-coefficients)
- [Interpreting logistic regression coefficients](#interpreting-logistic-regression-coefficients)
- [Intersection (A ∩ B)](#intersection-a-b)
- [Irreducible error](#irreducible-error)
- [itertools.product()](#itertools-product)

**J**

- [Johnson & Johnson earnings (running example)](#johnson-and-johnson-earnings-running-example)

**K**

- [k-distance plot](#k-distance-plot)
- [k-fold cross-validation](#k-fold-cross-validation)
- [k-means](#k-means)
- [k-means assumptions](#k-means-assumptions)
- [k-means sensitivities](#k-means-sensitivities)
- [k-means strengths and weaknesses](#k-means-strengths-and-weaknesses)
- [k-nearest neighbors (kNN)](#k-nearest-neighbors-knn)
- [Kernel](#kernel)
- [Kernel trick](#kernel-trick)
- [KernelExplainer and force plot](#kernelexplainer-and-force-plot)
- [Key components of a simulation](#key-components-of-a-simulation)
- [Key probability facts (Kolmogorov's axioms)](#key-probability-facts-kolmogorov-s-axioms)
- [KFold](#kfold)
- [KMeans](#kmeans)
- [KNeighborsClassifier](#kneighborsclassifier)
- [kNN strengths and weaknesses](#knn-strengths-and-weaknesses)

**L**

- [labels* and cluster_centers*](#labels-and-cluster-centers)
- [Lag (backshift) and .shift()](#lag-backshift-and-shift)
- [Large language model (LLM)](#large-language-model-llm)
- [Lasso regression (L1)](#lasso-regression-l1)
- [Law of Large Numbers (LLN)](#law-of-large-numbers-lln)
- [Lazy learner](#lazy-learner)
- [Learning rate (step size, α)](#learning-rate-step-size)
- [learning_rate and n_estimators](#learning-rate-and-n-estimators)
- [Leave-one-out cross-validation (LOOCV)](#leave-one-out-cross-validation-loocv)
- [Library version changes to know](#library-version-changes-to-know)
- [Limitations of simulation](#limitations-of-simulation)
- [Line of best fit](#line-of-best-fit)
- [Linear probability model](#linear-probability-model)
- [Linear regression](#linear-regression)
- [Linear, polynomial and RBF kernels](#linear-polynomial-and-rbf-kernels)
- [Linearity (L)](#linearity-l)
- [LinearRegression](#linearregression)
- [LINEM assumptions](#linem-assumptions)
- [Link function and the logit](#link-function-and-the-logit)
- [Linkage](#linkage)
- [Ljung–Box test](#ljung-box-test)
- [Loadings (components\_)](#loadings-components)
- [Local vs global minimum](#local-vs-global-minimum)
- [Log loss (binary cross-entropy)](#log-loss-binary-cross-entropy)
- [Log returns](#log-returns)
- [Log-odds](#log-odds)
- [Log-transforming the target](#log-transforming-the-target)
- [Logistic (sigmoid) function](#logistic-sigmoid-function)
- [Logistic regression](#logistic-regression)
- [Logistic regression assumptions](#logistic-regression-assumptions)
- [Logistic regression vs decision trees](#logistic-regression-vs-decision-trees)
- [LogisticRegressionCV](#logisticregressioncv)
- [Loss function](#loss-function)

**M**

- [MACD chart](#macd-chart)
- [Machine learning](#machine-learning)
- [Macro, weighted and micro averages](#macro-weighted-and-micro-averages)
- [Majority and minority class](#majority-and-minority-class)
- [make_blobs()](#make-blobs)
- [make_circles()](#make-circles)
- [make_classification()](#make-classification)
- [make_imbalance()](#make-imbalance)
- [make_moons()](#make-moons)
- [make_pipeline()](#make-pipeline)
- [Margin](#margin)
- [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc)
- [Matrix form of OLS](#matrix-form-of-ols)
- [max_features](#max-features)
- [Maximal margin classifier](#maximal-margin-classifier)
- [Mean absolute error (MAE)](#mean-absolute-error-mae)
- [Mean centering ratings](#mean-centering-ratings)
- [Mean squared error (MSE)](#mean-squared-error-mse)
- [Mercer kernel](#mercer-kernel)
- [Misclassification rate](#misclassification-rate)
- [MNIST digits (load_digits)](#mnist-digits-load-digits)
- [Model complexity](#model-complexity)
- [Model explainability](#model-explainability)
- [Model selection cheat sheet](#model-selection-cheat-sheet)
- [Model-specific vs model-agnostic methods](#model-specific-vs-model-agnostic-methods)
- [Modeling workflow (end to end)](#modeling-workflow-end-to-end)
- [Modelling with GenAI: what to check](#modelling-with-genai-what-to-check)
- [Monte Carlo simulation](#monte-carlo-simulation)
- [Moving average model (MA)](#moving-average-model-ma)
- [Multi-year ROI](#multi-year-roi)
- [Multimodal LLM](#multimodal-llm)
- [Multinomial logistic regression](#multinomial-logistic-regression)
- [Multiple linear regression](#multiple-linear-regression)
- [Multiplication rule (intersections)](#multiplication-rule-intersections)

**N**

- [n_jobs and verbose](#n-jobs-and-verbose)
- [Naive and seasonal naive forecasts](#naive-and-seasonal-naive-forecasts)
- [Newton's method](#newton-s-method)
- [No multicollinearity (M)](#no-multicollinearity-m)
- [Noise label (−1)](#noise-label-1)
- [Non-parametric model](#non-parametric-model)
- [Normality of residuals (N)](#normality-of-residuals-n)
- [normalize() vs scalers](#normalize-vs-scalers)
- [np.logspace()](#np-logspace)
- [np.polyfit() and np.polyval()](#np-polyfit-and-np-polyval)
- [np.random.choice()](#np-random-choice)
- [np.random.default_rng()](#np-random-default-rng)
- [np.random.randint() and np.random.uniform()](#np-random-randint-and-np-random-uniform)
- [Null accuracy](#null-accuracy)
- [Null model (baseline) for regression](#null-model-baseline-for-regression)

**O**

- [Observation weights](#observation-weights)
- [Odds](#odds)
- [Odds ratio](#odds-ratio)
- [One-time vs recurring costs](#one-time-vs-recurring-costs)
- [One-vs-rest (OvR)](#one-vs-rest-ovr)
- [Optimization](#optimization)
- [Optimization problem: objective and constraints](#optimization-problem-objective-and-constraints)
- [Ordinary least squares (OLS)](#ordinary-least-squares-ols)
- [Other cross-validation schemes](#other-cross-validation-schemes)
- [Other forecasting models](#other-forecasting-models)
- [Out-of-bag (OOB) observations and error](#out-of-bag-oob-observations-and-error)
- [Out-of-sample data](#out-of-sample-data)
- [Outcome](#outcome)
- [Overfitting](#overfitting)
- [Oversampling](#oversampling)

**P**

- [PACF plot](#pacf-plot)
- [pairwise_distances()](#pairwise-distances)
- [Parallelizable](#parallelizable)
- [param_grid](#param-grid)
- [Parsing unstructured data with GenAI](#parsing-unstructured-data-with-genai)
- [Partial dependence plot (PDP)](#partial-dependence-plot-pdp)
- [Partitioning the feature space](#partitioning-the-feature-space)
- [Payback period and NPV](#payback-period-and-npv)
- [PCA assumptions](#pca-assumptions)
- [PCA in scikit-learn](#pca-in-scikit-learn)
- [penalty and l1_ratio in LogisticRegression](#penalty-and-l1-ratio-in-logisticregression)
- [Penalty term](#penalty-term)
- [Perfect separation](#perfect-separation)
- [Permutation importance](#permutation-importance)
- [Pipeline + GridSearchCV](#pipeline-gridsearchcv)
- [Pipeline memory and debugging](#pipeline-memory-and-debugging)
- [Pipeline with ColumnTransformer](#pipeline-with-columntransformer)
- [Pipelines (in tuning)](#pipelines-in-tuning)
- [Plotting the logistic curve](#plotting-the-logistic-curve)
- [Poisson arrivals and exponential gaps](#poisson-arrivals-and-exponential-gaps)
- [Polynomial regression](#polynomial-regression)
- [PolynomialFeatures](#polynomialfeatures)
- [pos_label](#pos-label)
- [Precision](#precision)
- [Precision–recall curve](#precision-recall-curve)
- [Precision–recall trade-off](#precision-recall-trade-off)
- [.predict_proba()](#predict-proba)
- [Predictive maintenance](#predictive-maintenance)
- [Predictor and response (and their many names)](#predictor-and-response-and-their-many-names)
- [Principal component](#principal-component)
- [Principal component analysis (PCA)](#principal-component-analysis-pca)
- [Prior, likelihood and posterior](#prior-likelihood-and-posterior)
- [Probability](#probability)
- [Probability mass and density functions](#probability-mass-and-density-functions)
- [Probability notation P(A)](#probability-notation-p-a)
- [Problem framing](#problem-framing)
- [Problem statement](#problem-statement)
- [Project sponsor and KPI](#project-sponsor-and-kpi)
- [Prompting for code](#prompting-for-code)
- [Proxy variable](#proxy-variable)
- [PyMC](#pymc)

**Q**

- [Quasi-Newton methods (BFGS, L-BFGS)](#quasi-newton-methods-bfgs-l-bfgs)
- [Q–Q plot](#q-q-plot)

**R**

- [Random forest](#random-forest)
- [Random forest pros and cons](#random-forest-pros-and-cons)
- [RandomForestClassifier and RandomForestRegressor](#randomforestclassifier-and-randomforestregressor)
- [RandomizedSearchCV](#randomizedsearchcv)
- [RandomOverSampler](#randomoversampler)
- [Rank deficiency](#rank-deficiency)
- [Reading a statsmodels results object](#reading-a-statsmodels-results-object)
- [Reading ACF and PACF to choose p and q](#reading-acf-and-pacf-to-choose-p-and-q)
- [Reading grid search results](#reading-grid-search-results)
- [Reading train vs test scores](#reading-train-vs-test-scores)
- [Recall (sensitivity, true positive rate)](#recall-sensitivity-true-positive-rate)
- [Recommender system](#recommender-system)
- [Refitting for production](#refitting-for-production)
- [Regression vs classification tasks](#regression-vs-classification-tasks)
- [Regularization](#regularization)
- [Regularization is on by default](#regularization-is-on-by-default)
- [Regularizing categorical variables](#regularizing-categorical-variables)
- [Reinforcement learning](#reinforcement-learning)
- [Reproducibility habits](#reproducibility-habits)
- [.resample() for time series](#resample-for-time-series)
- [Resampling is a trade-off](#resampling-is-a-trade-off)
- [Residual plot](#residual-plot)
- [Return on investment (ROI)](#return-on-investment-roi)
- [Ridge and Lasso trace plot](#ridge-and-lasso-trace-plot)
- [Ridge regression (L2)](#ridge-regression-l2)
- [RidgeCV and LassoCV](#ridgecv-and-lassocv)
- [Risk-adjusted ROI](#risk-adjusted-roi)
- [ROC curve](#roc-curve)
- [Rolling statistics](#rolling-statistics)
- [Root mean squared error (RMSE)](#root-mean-squared-error-rmse)
- [Root node, internal node, leaf and branch](#root-node-internal-node-leaf-and-branch)
- [R² (coefficient of determination)](#r2-coefficient-of-determination)

**S**

- [Sample space](#sample-space)
- [Sampling bias](#sampling-bias)
- [Sampling with and without replacement](#sampling-with-and-without-replacement)
- [SARIMA](#sarima)
- [Scaling before clustering](#scaling-before-clustering)
- [Scaling for distance-based models](#scaling-for-distance-based-models)
- [.score()](#score)
- [scoring parameter](#scoring-parameter)
- [Scree plot](#scree-plot)
- [Season-based train/test split](#season-based-train-test-split)
- [Seasonal differencing](#seasonal-differencing)
- [Seasonal-mean forecast](#seasonal-mean-forecast)
- [seasonal_decompose()](#seasonal-decompose)
- [Seasonality](#seasonality)
- [Semi-supervised learning](#semi-supervised-learning)
- [Sensitivity analysis](#sensitivity-analysis)
- [Separating hyperplane](#separating-hyperplane)
- [Set and element](#set-and-element)
- [SHAP (SHapley Additive exPlanations)](#shap-shapley-additive-explanations)
- [Shapley value](#shapley-value)
- [Shrinkage and selection](#shrinkage-and-selection)
- [Signal and noise](#signal-and-noise)
- [Signum and indicator functions](#signum-and-indicator-functions)
- [Silhouette score](#silhouette-score)
- [Simple linear regression](#simple-linear-regression)
- [Simpson's paradox](#simpson-s-paradox)
- [SimPy](#simpy)
- [Simulation](#simulation)
- [Simulation (Monte Carlo)](#simulation-monte-carlo)
- [Singular value decomposition (SVD)](#singular-value-decomposition-svd)
- [sklearn.metrics](#sklearn-metrics)
- [Slack variable](#slack-variable)
- [Slope / coefficient (β₁)](#slope-coefficient)
- [SMART goals](#smart-goals)
- [SMOTE](#smote)
- [SMOTENC and SMOTEN](#smotenc-and-smoten)
- [Soft margin classifier](#soft-margin-classifier)
- [solver and max_iter](#solver-and-max-iter)
- [Sparse matrix](#sparse-matrix)
- [Specificity (true negative rate)](#specificity-true-negative-rate)
- [Splitting criterion](#splitting-criterion)
- [Stacking](#stacking)
- [Stage 10: Deploy, monitor and next steps](#stage-10-deploy-monitor-and-next-steps)
- [Stage 1: Frame the question](#stage-1-frame-the-question)
- [Stage 2: Get and understand the data](#stage-2-get-and-understand-the-data)
- [Stage 3: Choose the split and validation strategy](#stage-3-choose-the-split-and-validation-strategy)
- [Stage 4: Build baselines](#stage-4-build-baselines)
- [Stage 5: Choose candidate models](#stage-5-choose-candidate-models)
- [Stage 6: Build a pipeline and tune](#stage-6-build-a-pipeline-and-tune)
- [Stage 7: Evaluate honestly](#stage-7-evaluate-honestly)
- [Stage 8: Interpret and explain](#stage-8-interpret-and-explain)
- [Stage 9: Communicate the value](#stage-9-communicate-the-value)
- [Stakeholder](#stakeholder)
- [Standardize before regularizing](#standardize-before-regularizing)
- [Stationarity](#stationarity)
- [Statistical independence](#statistical-independence)
- [Statistical modeling](#statistical-modeling)
- [statsmodels macrodata](#statsmodels-macrodata)
- [statsmodels OLS](#statsmodels-ols)
- [step\_\_parameter naming](#step-parameter-naming)
- [Stochastic gradient descent (SGD)](#stochastic-gradient-descent-sgd)
- [Stocks and flows](#stocks-and-flows)
- [Stopping criteria](#stopping-criteria)
- [Stratified split for imbalanced data](#stratified-split-for-imbalanced-data)
- [StratifiedKFold](#stratifiedkfold)
- [Success metrics and hypotheses](#success-metrics-and-hypotheses)
- [Sum of squared errors (SSE)](#sum-of-squared-errors-sse)
- [Supervised learning](#supervised-learning)
- [Support vector machine (SVM)](#support-vector-machine-svm)
- [Support vectors](#support-vectors)
- [SVC and LinearSVC](#svc-and-linearsvc)
- [SVM pros and cons](#svm-pros-and-cons)
- [Synthetic data generation](#synthetic-data-generation)
- [System dynamics (SD)](#system-dynamics-sd)

**T**

- [Technical debt](#technical-debt)
- [Temperature and sampling](#temperature-and-sampling)
- [test_size, stratify and shuffle](#test-size-stratify-and-shuffle)
- [Testing AI-generated code](#testing-ai-generated-code)
- [The cost of grid searching](#the-cost-of-grid-searching)
- [The modeling workflow in scikit-learn](#the-modeling-workflow-in-scikit-learn)
- [Three-way split (validation set)](#three-way-split-validation-set)
- [Time handling in simulations](#time-handling-in-simulations)
- [Time series](#time-series)
- [Time series train/test split](#time-series-train-test-split)
- [Token](#token)
- [Total sum of squares (SST)](#total-sum-of-squares-sst)
- [Train-test ratio rule of thumb](#train-test-ratio-rule-of-thumb)
- [Train-test split](#train-test-split)
- [Training vs test error curve](#training-vs-test-error-curve)
- [Transfer learning](#transfer-learning)
- [Transformer and attention](#transformer-and-attention)
- [Transparency](#transparency)
- [Trend](#trend)
- [True/false positives and negatives](#true-false-positives-and-negatives)
- [Tuning tree depth](#tuning-tree-depth)
- [Type I and Type II errors](#type-i-and-type-ii-errors)
- [Types of variables](#types-of-variables)

**U**

- [Underfitting](#underfitting)
- [Undersampling](#undersampling)
- [Union (A ∪ B)](#union-a-b)
- [Univariate vs multivariate time series](#univariate-vs-multivariate-time-series)
- [Unlabeled data](#unlabeled-data)
- [Unsupervised learning](#unsupervised-learning)
- [Uses of clustering](#uses-of-clustering)
- [Using this glossary for your capstone](#using-this-glossary-for-your-capstone)
- [Utility (ratings) matrix](#utility-ratings-matrix)
- [Utilization](#utilization)

**V**

- [Validating GenAI classifications](#validating-genai-classifications)
- [Variance (model variance)](#variance-model-variance)
- [Variance inflation factor (VIF)](#variance-inflation-factor-vif)
- [Vector autoregression (VAR)](#vector-autoregression-var)
- [Vectorized simulation](#vectorized-simulation)
- [Verification and validation](#verification-and-validation)
- [Visualizing a decision tree (plot_tree)](#visualizing-a-decision-tree-plot-tree)
- [Volatility clustering](#volatility-clustering)
- [Voronoi diagram](#voronoi-diagram)
- [VotingClassifier](#votingclassifier)

**W**

- [Weak learner](#weak-learner)
- [When imbalance matters](#when-imbalance-matters)
- [When to use PCA (and when not)](#when-to-use-pca-and-when-not)
- [Where GenAI helps in the data science lifecycle](#where-genai-helps-in-the-data-science-lifecycle)
- [Which model is best?](#which-model-is-best)
- [while loop](#while-loop)
- [White noise](#white-noise)
- [White-box model](#white-box-model)
- [Why data science projects fail](#why-data-science-projects-fail)
- [Why explainability matters](#why-explainability-matters)
- [Why lasso gives exact zeros](#why-lasso-gives-exact-zeros)
- [Why linear models overfit](#why-linear-models-overfit)
- [Why not linear regression for classification?](#why-not-linear-regression-for-classification)
- [Wisdom of the crowd](#wisdom-of-the-crowd)

**X**

- [X is 2D, y is 1D](#x-is-2d-y-is-1d)
- [XGBoost](#xgboost)
