# 📘 Machine Learning Models — Complete Guide

![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-blue)
![Python](https://img.shields.io/badge/Language-Python-yellow)
![Status](https://img.shields.io/badge/Status-Learning%20%26%20Practice-success)

A comprehensive reference guide to Machine Learning algorithms, covering their concepts, working principles, applications, advantages, limitations, evaluation metrics, and Python libraries.

## 👨‍💻 Author

**Vijay Kumar Prajapat**

- **GitHub:** [Vijay7300](https://github.com/Vijay7300)
- **Portfolio:** [Visit My GitHub Profile](https://github.com/Vijay7300)
- **Last Updated:** 20 October 2026

---

## 📑 Table of Contents

1. [Introduction to Machine Learning](#-1-introduction-to-machine-learning)
2. [Types of Machine Learning](#-2-types-of-machine-learning)
3. [Supervised Learning](#-3-supervised-learning)
4. [Unsupervised Learning](#-4-unsupervised-learning)
5. [Semi-Supervised Learning](#-5-semi-supervised-learning)
6. [Reinforcement Learning](#-6-reinforcement-learning)
7. [Ensemble Learning](#-7-ensemble-learning)
8. [Dimensionality Reduction](#-8-dimensionality-reduction)
9. [Feature Selection](#-9-feature-selection)
10. [Anomaly Detection](#-10-anomaly-detection)
11. [Time-Series Forecasting](#-11-time-series-forecasting)
12. [Recommendation Systems](#-12-recommendation-systems)
13. [Deep Learning](#-13-deep-learning)
14. [Model Evaluation Metrics](#-14-model-evaluation-metrics)
15. [Model Selection and Hyperparameter Tuning](#-15-model-selection-and-hyperparameter-tuning)
16. [Data Preprocessing](#-16-data-preprocessing)
17. [Model Comparison](#-17-model-comparison)
18. [Python Libraries](#-18-python-libraries)
19. [Learning Roadmap](#-19-learning-roadmap)
20. [References](#-20-references)

---

## 📌 1. Introduction to Machine Learning

Machine Learning (ML) is a branch of Artificial Intelligence that enables computers to learn patterns from data and make predictions or decisions without explicitly programming every rule.

### Main applications

- Healthcare: Disease prediction and medical image analysis.
- Finance: Fraud detection and credit risk assessment.
- E-commerce: Recommendation systems and customer segmentation.
- Transportation: Autonomous driving and traffic prediction.
- Agriculture: Crop monitoring and obstacle detection.
- Business: Sales forecasting and customer churn prediction.
- Natural Language Processing: Sentiment analysis and text classification.
- Computer Vision: Object detection, image classification, and segmentation.

### Basic Machine Learning workflow

1. Collect data.
2. Explore and understand the data.
3. Clean missing values and handle inconsistencies.
4. Perform feature engineering.
5. Split data into training, validation, and testing sets.
6. Select an appropriate algorithm.
7. Train the model.
8. Evaluate performance.
9. Tune hyperparameters.
10. Deploy and monitor the model.

---

## 🧠 2. Types of Machine Learning

Machine Learning is commonly divided into four major learning paradigms.

| Type | Training data | Main objective | Example |
|---|---|---|---|
| Supervised Learning | Labelled | Predict outputs | House-price prediction |
| Unsupervised Learning | Unlabelled | Discover patterns | Customer segmentation |
| Semi-Supervised Learning | Few labelled and many unlabelled samples | Improve learning with limited labels | Image classification |
| Reinforcement Learning | Rewards and environment feedback | Learn optimal actions | Autonomous navigation |

---

## 📈 3. Supervised Learning

Supervised learning learns a mapping between input features \(X\) and known target values \(y\).

It has two main tasks: regression and classification.

### 3.1 Regression Algorithms

Regression predicts a continuous numerical value.

#### 1. Linear Regression

**Concept:** Models a linear relationship between input features and a continuous target.

Equation:

\[
y=\beta_0+\beta_1x_1+\cdots+\beta_nx_n+\epsilon
\]

**Applications:** House prices, sales forecasting, and demand estimation.

**Advantages:**
- Simple and interpretable.
- Fast to train.
- Provides a useful baseline.

**Limitations:**
- Cannot naturally represent complex nonlinear relationships without feature transformations.
- Sensitive to influential observations.
- Statistical inference relies on assumptions that must be checked.

**Python:** `sklearn.linear_model.LinearRegression`

#### 2. Polynomial Regression

**Concept:** Extends linear regression by adding polynomial features.

\[
y=\beta_0+\beta_1x+\beta_2x^2+\cdots+\beta_dx^d
\]

**Applications:** Nonlinear trends and growth curves.

**Advantages:** Models curved relationships.

**Limitations:** High polynomial degrees can overfit.

**Python:** `PolynomialFeatures` + `LinearRegression`

#### 3. Ridge Regression

**Concept:** Linear regression with L2 regularization.

\[
L(\beta)=MSE+\lambda\sum_j\beta_j^2
\]

**Applications:** Prediction with correlated features.

**Advantages:** Reduces coefficient variance and helps control overfitting.

**Limitations:** Usually does not set coefficients exactly to zero.

**Python:** `sklearn.linear_model.Ridge`

#### 4. Lasso Regression

**Concept:** Linear regression with L1 regularization.

\[
L(\beta)=MSE+\lambda\sum_j|\beta_j|
\]

**Applications:** Prediction and sparse feature selection.

**Advantages:** Can shrink some coefficients to zero.

**Limitations:** May select one feature arbitrarily among strongly correlated predictors.

**Python:** `sklearn.linear_model.Lasso`

#### 5. Elastic Net Regression

**Concept:** Combines L1 and L2 regularization.

**Applications:** High-dimensional data with correlated predictors.

**Advantages:** Balances sparsity and coefficient stability.

**Limitations:** Requires tuning regularization parameters.

**Python:** `sklearn.linear_model.ElasticNet`

#### 6. Decision Tree Regression

**Concept:** Divides feature space into regions and predicts a value for each leaf.

**Applications:** Price prediction and nonlinear tabular data.

**Advantages:** Captures nonlinear relationships and feature interactions.

**Limitations:** Deep trees can overfit and be unstable.

**Python:** `sklearn.tree.DecisionTreeRegressor`

#### 7. Random Forest Regression

**Concept:** Averages predictions from multiple randomized decision trees.

**Applications:** Laptop-price prediction, property valuation, and demand estimation.

**Advantages:** Handles nonlinear patterns and often works well on tabular data.

**Limitations:** Less interpretable than a single tree and can require substantial memory.

**Python:** `sklearn.ensemble.RandomForestRegressor`

#### 8. Support Vector Regression (SVR)

**Concept:** Fits a function while allowing errors within an epsilon-insensitive margin.

**Applications:** Small- to medium-sized nonlinear regression tasks.

**Advantages:** Supports nonlinear prediction through kernels.

**Limitations:** Requires scaling and can be computationally expensive on large datasets.

**Python:** `sklearn.svm.SVR`

#### 9. K-Nearest Neighbors Regression

**Concept:** Predicts a target using the target values of nearby training examples.

**Applications:** Similarity-based estimation.

**Advantages:** Simple and non-parametric.

**Limitations:** Prediction can be slow and sensitive to feature scaling.

**Python:** `sklearn.neighbors.KNeighborsRegressor`

#### 10. Gradient Boosting Regression

**Concept:** Builds trees sequentially, with each new tree improving the current ensemble.

**Applications:** Forecasting, pricing, and risk estimation.

**Advantages:** Models complex nonlinear patterns.

**Limitations:** Requires hyperparameter tuning and can overfit.

**Python:** `sklearn.ensemble.GradientBoostingRegressor`

#### 11. XGBoost Regression

**Concept:** Uses optimized gradient-boosted decision trees with regularization.

**Applications:** Structured datasets and predictive analytics.

**Advantages:** Strong performance on many tabular prediction tasks.

**Limitations:** Requires careful tuning and appropriate validation.

**Python:** `xgboost.XGBRegressor`

#### 12. LightGBM Regression

**Concept:** A gradient-boosting framework designed for efficient tree learning.

**Applications:** Large structured datasets.

**Advantages:** Efficient training on many tabular problems.

**Limitations:** Can overfit small datasets if configured poorly.

**Python:** `lightgbm.LGBMRegressor`

#### 13. CatBoost Regression

**Concept:** Gradient boosting with built-in techniques for handling categorical features.

**Applications:** Datasets with many categorical variables.

**Advantages:** Often reduces the need for extensive categorical encoding.

**Limitations:** Still requires validation and tuning.

**Python:** `catboost.CatBoostRegressor`

#### Other regression models

- Bayesian Regression
- Huber Regression
- Quantile Regression
- Stochastic Gradient Descent Regression
- Gaussian Process Regression
- Partial Least Squares Regression
- Kernel Ridge Regression
- Generalized Linear Models

### 3.2 Classification Algorithms

Classification predicts a discrete class or category.

#### 1. Logistic Regression

**Concept:** Estimates class probabilities using the logistic or softmax function.

For binary classification:

\[
P(y=1\mid x)=\frac{1}{1+e^{-(w^Tx+b)}}
\]

**Applications:** Spam detection, churn prediction, and credit-risk classification.

**Advantages:** Interpretable, efficient, and provides probabilities.

**Limitations:** A linear decision boundary may be insufficient for complex patterns.

**Python:** `sklearn.linear_model.LogisticRegression`

#### 2. K-Nearest Neighbors (KNN)

**Concept:** Assigns a class based on the labels of nearby training samples.

**Applications:** Pattern recognition and similarity-based classification.

**Advantages:** Simple and easy to understand.

**Limitations:** Sensitive to scaling, irrelevant features, and the choice of \(k\).

**Python:** `sklearn.neighbors.KNeighborsClassifier`

#### 3. Naive Bayes

**Concept:** Uses Bayes' theorem with a conditional independence assumption.

\[
P(C\mid X)\propto P(X\mid C)P(C)
\]

**Applications:** Spam filtering, document classification, and sentiment analysis.

**Advantages:** Fast and effective for many text-classification tasks.

**Limitations:** The independence assumption may not hold in real data.

**Variants:**
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Complement Naive Bayes
- Categorical Naive Bayes

**Python:** `sklearn.naive_bayes`

#### 4. Support Vector Machine (SVM)

**Concept:** Finds a decision boundary that maximizes the margin between classes. Kernels can represent nonlinear boundaries.

**Applications:** Text classification, image recognition, and small-to-medium datasets.

**Advantages:** Effective in high-dimensional feature spaces.

**Limitations:** Scaling and kernel selection matter; large datasets may be expensive.

**Python:** `sklearn.svm.SVC`

#### 5. Decision Tree Classifier

**Concept:** Uses feature-based rules to recursively divide data into classes.

**Applications:** Medical decision support and customer classification.

**Advantages:** Interpretable rules and little need for feature scaling.

**Limitations:** Can overfit and change significantly with small data changes.

**Python:** `sklearn.tree.DecisionTreeClassifier`

#### 6. Random Forest Classifier

**Concept:** Combines predictions from multiple decision trees using voting or probability averaging.

**Applications:** Fraud detection, risk classification, and customer analytics.

**Advantages:** Captures nonlinear patterns and reduces the variance of individual trees.

**Limitations:** Larger models can be slower and less interpretable.

**Python:** `sklearn.ensemble.RandomForestClassifier`

#### 7. AdaBoost

**Concept:** Sequentially trains weak learners while increasing emphasis on previously misclassified samples.

**Applications:** Binary classification and boosting baselines.

**Advantages:** Combines simple learners into a stronger predictor.

**Limitations:** Can be sensitive to noisy observations.

**Python:** `sklearn.ensemble.AdaBoostClassifier`

#### 8. Gradient Boosting Classifier

**Concept:** Builds an additive model of weak learners that optimizes a differentiable loss function.

**Applications:** Structured-data classification.

**Advantages:** Can learn complex feature relationships.

**Limitations:** Requires tuning and can overfit.

**Python:** `sklearn.ensemble.GradientBoostingClassifier`

#### 9. XGBoost Classifier

**Concept:** Optimized gradient boosting with regularization and efficient training.

**Applications:** Fraud detection, churn prediction, and classification competitions.

**Advantages:** Flexible and powerful for tabular data.

**Limitations:** Can require considerable tuning.

**Python:** `xgboost.XGBClassifier`

#### 10. LightGBM Classifier

**Concept:** Efficient gradient-boosted tree learning.

**Applications:** Large-scale structured classification.

**Advantages:** Fast training for suitable datasets.

**Limitations:** Requires careful parameter selection and validation.

**Python:** `lightgbm.LGBMClassifier`

#### 11. CatBoost Classifier

**Concept:** Gradient boosting with specialized categorical-feature handling.

**Applications:** Customer churn, credit risk, and mixed-type tabular data.

**Advantages:** Convenient categorical-feature support.

**Limitations:** Model complexity and tuning still matter.

**Python:** `catboost.CatBoostClassifier`

#### Other classification models

- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Perceptron
- Passive-Aggressive Classifier
- Stochastic Gradient Descent Classifier
- Gaussian Process Classifier
- Multilayer Perceptron Classifier

---

## 🔍 4. Unsupervised Learning

Unsupervised learning discovers structure in data without requiring target labels.

### 4.1 Clustering Algorithms

#### 1. K-Means Clustering

**Concept:** Groups observations into \(K\) clusters by minimizing within-cluster squared distances.

**Applications:** Customer segmentation and grouping similar samples.

**Advantages:** Simple and efficient.

**Limitations:** Requires selecting \(K\), is sensitive to initialization, and works best with reasonably compact clusters.

**Python:** `sklearn.cluster.KMeans`

#### 2. Hierarchical Clustering

**Concept:** Builds a hierarchy of clusters using agglomerative merging or divisive splitting.

**Applications:** Biological analysis and customer segmentation.

**Advantages:** Produces a hierarchy that can be visualized as a dendrogram.

**Limitations:** Can be expensive for large datasets.

**Python:** `sklearn.cluster.AgglomerativeClustering`

#### 3. DBSCAN

**Concept:** Groups dense regions and identifies points in low-density regions as noise.

**Applications:** Geographic data, spatial clustering, and anomaly exploration.

**Advantages:** Does not require a predefined number of clusters and can find irregular shapes.

**Limitations:** Sensitive to density parameters and varying-density data.

**Python:** `sklearn.cluster.DBSCAN`

#### 4. HDBSCAN

**Concept:** Finds stable clusters across a hierarchy of density levels.

**Applications:** Complex datasets with varying cluster densities.

**Advantages:** Can handle varying densities better than standard DBSCAN in many situations.

**Limitations:** Parameter selection and interpretation require care.

**Python:** `hdbscan.HDBSCAN`

#### 5. Mean Shift

**Concept:** Moves candidate points toward local density maxima.

**Applications:** Image segmentation and density-based grouping.

**Advantages:** Does not require specifying the number of clusters in advance.

**Limitations:** Can be computationally expensive.

**Python:** `sklearn.cluster.MeanShift`

#### 6. Gaussian Mixture Model (GMM)

**Concept:** Represents data as a mixture of Gaussian probability distributions.

**Applications:** Soft clustering and probabilistic segmentation.

**Advantages:** Provides probabilistic cluster memberships.

**Limitations:** Distributional assumptions and the number of components affect performance.

**Python:** `sklearn.mixture.GaussianMixture`

#### Other clustering methods

- Spectral Clustering
- BIRCH
- OPTICS
- Affinity Propagation
- Fuzzy C-Means
- Mini-Batch K-Means

### 4.2 Association Rule Learning

Association methods discover items that frequently occur together.

**Apriori:** Finds frequent itemsets and association rules.

**FP-Growth:** Mines frequent patterns using an FP-tree structure.

**Applications:** Market-basket analysis, product bundling, and shopping behavior.

Common measures:

- Support
- Confidence
- Lift

---

## 🏷️ 5. Semi-Supervised Learning

Semi-supervised learning uses a small amount of labelled data together with a larger amount of unlabelled data.

### Common approaches

- **Self-Training:** A model generates pseudo-labels for selected unlabelled samples.
- **Label Propagation:** Labels spread through a similarity graph.
- **Label Spreading:** A graph-based approach that regularizes label propagation.
- **Consistency Regularization:** Encourages predictions to remain consistent under suitable input perturbations.

**Applications:** Image classification, speech recognition, and medical datasets where labelling is expensive.

**Python:** `sklearn.semi_supervised`

---

## 🎮 6. Reinforcement Learning

Reinforcement Learning (RL) trains an agent to select actions in an environment to maximize cumulative reward.

### Core components

- **Agent:** Learner or decision-maker.
- **Environment:** The system with which the agent interacts.
- **State:** Current representation of the environment.
- **Action:** A decision made by the agent.
- **Reward:** Feedback received after an action.
- **Policy:** A strategy for selecting actions.
- **Value function:** Expected future return from a state or state-action pair.

### Important RL algorithms

| Algorithm | Type | Description |
|---|---|---|
| Q-Learning | Model-free, value-based | Learns action values off-policy |
| SARSA | Model-free, value-based | Learns action values on-policy |
| DQN | Deep value-based | Uses a neural network to approximate action values |
| Policy Gradient | Policy-based | Optimizes policy parameters |
| REINFORCE | Policy-based | Uses sampled returns to update a policy |
| Actor-Critic | Hybrid | Combines a policy learner and value estimator |
| A2C | Actor-Critic | Uses synchronized advantage-based updates |
| PPO | Policy optimization | Uses a clipped or constrained update objective |
| DDPG | Off-policy actor-critic | Supports continuous action spaces |
| SAC | Off-policy actor-critic | Uses entropy-regularized policy learning |

**Applications:** Robotics, game-playing, resource allocation, and autonomous navigation.

**Python libraries:** Gymnasium, Stable-Baselines3, PyTorch, and TensorFlow.

---

## 🌳 7. Ensemble Learning

Ensemble learning combines several models to improve predictive performance or robustness.

### 7.1 Bagging

Trains models on bootstrap samples and combines their predictions.

**Example:** Random Forest.

**Main benefit:** Often reduces prediction variance.

### 7.2 Boosting

Trains models sequentially, with later learners focusing on errors or optimizing a loss function.

**Examples:**
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

**Main benefit:** Can build strong predictors from weak learners.

### 7.3 Voting

Combines predictions from different models.

- Hard voting: Selects the most common predicted class.
- Soft voting: Averages class probabilities.

### 7.4 Stacking

Uses predictions from base models as inputs to a meta-model.

**Applications:** Tabular prediction, competitions, and model ensembles.

**Important:** Use out-of-fold predictions when training the meta-model to avoid information leakage.

---

## 📉 8. Dimensionality Reduction

Dimensionality reduction transforms high-dimensional data into a smaller representation while attempting to preserve relevant information.

### 1. Principal Component Analysis (PCA)

Finds orthogonal directions that capture the maximum possible variance.

**Applications:** Feature compression, visualization, and noise reduction.

**Python:** `sklearn.decomposition.PCA`

### 2. Linear Discriminant Analysis (LDA)

Finds projections that emphasize separation between labelled classes.

**Applications:** Supervised feature extraction and classification.

**Python:** `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`

**Difference:** PCA is unsupervised; LDA uses class labels.

### 3. Independent Component Analysis (ICA)

Separates signals into statistically independent components under suitable assumptions.

**Applications:** Blind source separation and signal processing.

**Python:** `sklearn.decomposition.FastICA`

### 4. t-SNE

Creates a low-dimensional representation that emphasizes local neighborhood structure.

**Applications:** Visualizing high-dimensional embeddings.

**Limitation:** Distances between separated clusters can be misleading; it is generally not a default predictive feature transformation.

**Python:** `sklearn.manifold.TSNE`

### 5. UMAP

Creates a low-dimensional embedding intended to preserve important neighborhood relationships.

**Applications:** Exploratory visualization and embedding analysis.

**Limitation:** Results depend on parameters and data characteristics.

**Python:** `umap-learn`

### Other methods

- Truncated SVD
- Kernel PCA
- Non-Negative Matrix Factorization (NMF)
- Multidimensional Scaling (MDS)
- Isomap
- Locally Linear Embedding (LLE)
- Random Projection

---

## 🎯 9. Feature Selection

Feature selection chooses a subset of the original features that may improve efficiency, interpretability, and generalization.

### 9.1 Filter Methods

Select features using statistical properties independent of a specific predictive model.

Examples:
- Correlation analysis
- Variance Threshold
- Chi-Square test
- ANOVA F-test
- Mutual Information

### 9.2 Wrapper Methods

Evaluate subsets of features using model performance.

Examples:
- Forward Selection
- Backward Elimination
- Recursive Feature Elimination (RFE)
- Sequential Feature Selection

### 9.3 Embedded Methods

Select or penalize features during model training.

Examples:
- Lasso Regression
- Tree-Based Feature Importance
- SelectFromModel

**Important:** Fit feature selection only on training data, preferably inside a cross-validation pipeline.

---

## 🚨 10. Anomaly Detection

Anomaly detection identifies observations that differ substantially from expected patterns.

### 1. Isolation Forest

Isolates unusual observations using randomized trees.

**Applications:** Fraud detection and equipment monitoring.

**Python:** `sklearn.ensemble.IsolationForest`

### 2. Local Outlier Factor (LOF)

Compares the local density of a sample with the density of its neighbors.

**Applications:** Local outlier detection.

**Python:** `sklearn.neighbors.LocalOutlierFactor`

### 3. One-Class SVM

Learns a boundary around typical observations.

**Applications:** Novelty detection.

**Python:** `sklearn.svm.OneClassSVM`

### 4. Elliptic Envelope

Estimates a robust covariance structure and identifies observations outside the expected distribution.

**Applications:** Approximately elliptical, unimodal data.

**Python:** `sklearn.covariance.EllipticEnvelope`

### 5. Autoencoders

Learn to reconstruct normal data; high reconstruction error can indicate anomalies.

**Applications:** Industrial monitoring and complex sensor data.

**Python:** PyTorch or TensorFlow/Keras.

---

## ⏳ 11. Time-Series Forecasting

Time-series models predict future observations from temporally ordered data.

### 1. AR (Autoregressive Model)

Predicts a value using previous observations.

### 2. MA (Moving Average Model)

Models a series using current and past error terms.

### 3. ARMA

Combines autoregressive and moving-average components for suitable stationary series.

### 4. ARIMA

Adds differencing to model non-stationary series.

### 5. SARIMA

Extends ARIMA with seasonal components.

### 6. Exponential Smoothing

Uses weighted averages, with more recent observations generally receiving greater weight.

### 7. Prophet

A forecasting framework that models trend, seasonality, and specified holidays.

### 8. ML-Based Forecasting

Regression models such as Random Forest, XGBoost, and neural networks can forecast using lagged values and engineered temporal features.

**Applications:** Sales, demand, weather-related forecasting, and inventory planning.

**Python libraries:** `statsmodels`, `sktime`, `Prophet`, `scikit-learn`, and `darts`.

**Important:** Preserve chronological order during validation. Avoid randomly splitting time-series observations when that would leak future information into training.

---

## 🛍️ 12. Recommendation Systems

Recommendation systems suggest items that may be relevant to a user.

### 1. Content-Based Filtering

Recommends items based on their attributes and a user's preferences.

### 2. Collaborative Filtering

Uses similarities in user-item interactions.

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering

### 3. Matrix Factorization

Learns compact user and item representations from interaction data.

Examples:
- Singular Value Decomposition (SVD)
- Alternating Least Squares (ALS)

### 4. Hybrid Recommendation

Combines content-based and collaborative approaches.

**Applications:** Movies, music, e-commerce, and online learning platforms.

**Evaluation:** Precision@K, Recall@K, MAP, NDCG, and appropriate ranking or rating-prediction metrics.

---

## 🧬 13. Deep Learning

Deep Learning uses multilayer neural networks to learn complex representations from data.

### 13.1 Artificial Neural Network (ANN / MLP)

Uses interconnected neurons and nonlinear activation functions.

**Applications:** Tabular classification and regression.

**Libraries:** TensorFlow/Keras and PyTorch.

### 13.2 Convolutional Neural Network (CNN)

Learns spatial features through convolutional filters.

**Applications:** Image classification, object detection, segmentation, and medical imaging.

### 13.3 Recurrent Neural Network (RNN)

Processes sequential information using recurrent connections.

**Applications:** Sequence modelling and time-dependent data.

### 13.4 Long Short-Term Memory (LSTM)

Uses gated memory mechanisms to learn longer-term dependencies.

**Applications:** Sequence modelling, forecasting, and text processing.

### 13.5 Gated Recurrent Unit (GRU)

Uses a simpler gated architecture than LSTM.

**Applications:** Sequential prediction and time-series modelling.

### 13.6 Autoencoders and Variational Autoencoders

Learn compressed representations; variational autoencoders additionally model a latent probability distribution.

**Applications:** Compression, representation learning, and anomaly detection.

### 13.7 Transformers

Use attention mechanisms to model relationships between sequence elements.

**Applications:** Large language models, translation, vision, and multimodal learning.

### 13.8 Generative Adversarial Networks (GANs)

Train a generator and discriminator in competition.

**Applications:** Synthetic image generation and data augmentation.

### 13.9 Diffusion Models

Learn to generate data through a denoising process.

**Applications:** Image synthesis and other generative tasks.

**Libraries:** PyTorch, TensorFlow/Keras, Hugging Face Transformers, and Diffusers.

---

## 📊 14. Model Evaluation Metrics

Choose evaluation metrics according to the task and the cost of different errors.

### 14.1 Regression Metrics

| Metric | Meaning |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Average squared prediction error |
| RMSE | Square root of MSE |
| \(R^2\) | Relative goodness of fit against a mean-target baseline |
| Adjusted \(R^2\) | Adjusts \(R^2\) for the number of predictors |
| MAPE | Average absolute percentage error, with limitations near zero targets |

### 14.2 Classification Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of correctly classified samples |
| Precision | Fraction of predicted positives that are positive |
| Recall | Fraction of actual positives correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ranking performance across ROC thresholds |
| PR-AUC | Precision-recall performance, especially useful with class imbalance |
| Log Loss | Quality of predicted class probabilities |
| Confusion Matrix | Counts of true and false predictions by class |

Formulas:

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

\[
F1=2\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

### 14.3 Clustering Metrics

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index (ARI), when reference labels are available
- Normalized Mutual Information (NMI), when reference labels are available

### 14.4 Forecasting Metrics

- MAE
- RMSE
- MASE
- sMAPE
- MAPE, when appropriate

### 14.5 Reinforcement Learning Metrics

- Average episodic return
- Success rate
- Episode length
- Sample efficiency
- Stability across multiple training runs

---

## ⚙️ 15. Model Selection and Hyperparameter Tuning

### 15.1 Train, Validation, and Test Sets

- **Training set:** Fits model parameters.
- **Validation set:** Helps choose models and tune hyperparameters.
- **Test set:** Provides a final estimate of generalization performance.

### 15.2 Cross-Validation

Common approaches:

- K-Fold Cross-Validation
- Stratified K-Fold
- Leave-One-Out Cross-Validation
- Group K-Fold
- Time-Series Split

Choose the strategy according to class balance, grouping, and temporal structure.

### 15.3 Hyperparameter Tuning

**Grid Search:** Tests specified combinations of parameters.

**Random Search:** Samples parameter combinations from specified distributions.

**Bayesian Optimization:** Uses a surrogate model to guide promising experiments.

**Successive Halving:** Allocates resources progressively to promising candidates.

Python tools:
- `GridSearchCV`
- `RandomizedSearchCV`
- Optuna
- Hyperopt

### 15.4 Overfitting and Underfitting

**Overfitting:** A model learns training-specific patterns that do not generalize well.

Possible solutions:
- Regularization
- Cross-validation
- Early stopping
- Simpler models
- More representative training data
- Appropriate data augmentation

**Underfitting:** A model cannot capture the relevant patterns.

Possible solutions:
- Increase model capacity
- Improve feature engineering
- Reduce excessive regularization
- Train sufficiently

---

## 🧹 16. Data Preprocessing

Data preprocessing is essential for building reliable models.

### 16.1 Missing Values

- Mean or median imputation
- Most-frequent imputation
- KNN imputation
- Model-based imputation

### 16.2 Feature Scaling

**Standardization:** Transforms features to have approximately zero mean and unit variance.

**Min-Max Scaling:** Maps values to a specified range.

Scaling is especially important for distance-based methods, SVMs, and many linear optimization methods. Tree-based models generally do not require it.

### 16.3 Categorical Encoding

- One-Hot Encoding
- Ordinal Encoding
- Target Encoding, with leakage-safe implementation
- Model-specific categorical handling

### 16.4 Handling Imbalanced Data

- Class weights
- Random oversampling
- Random undersampling
- SMOTE, applied only to training data within the validation pipeline
- Threshold tuning

### 16.5 Feature Engineering

- Log and power transformations
- Polynomial features
- Interaction features
- Date and time features
- Lagged features for time series
- Domain-specific features

### 16.6 Preventing Data Leakage

Never use information from validation or test data to fit preprocessing transformations or select features.

Fit preprocessing steps on training data only, preferably by using a pipeline.

---

## ⚖️ 17. Model Comparison

| Model | Regression | Classification | Interpretability | Scaling usually needed |
|---|---|---|---|---|
| Linear Regression | Yes | No | High | Sometimes |
| Logistic Regression | No | Yes | High | Usually helpful |
| KNN | Yes | Yes | Moderate | Yes |
| Naive Bayes | No | Yes | Moderate | Depends on variant |
| SVM / SVR | Yes | Yes | Low to moderate | Yes |
| Decision Tree | Yes | Yes | High for small trees | No |
| Random Forest | Yes | Yes | Moderate to low | No |
| Gradient Boosting | Yes | Yes | Moderate to low | Usually no |
| XGBoost | Yes | Yes | Moderate to low | Usually no |
| K-Means | No | No | Moderate | Usually yes |
| PCA | No | No | Moderate | Often yes |
| Isolation Forest | No | No | Low to moderate | Usually no |

*These are general tendencies, not guarantees. Model suitability depends on the dataset, objective, data quality, and evaluation strategy.*

### How to choose a model

- Start with a simple baseline.
- Use linear models when interpretability and a linear relationship are important.
- Try tree ensembles for complex tabular datasets.
- Use SVM or KNN when dataset size and feature geometry make them suitable.
- Use clustering methods when labels are unavailable and grouping is the objective.
- Use CNNs for spatial image patterns and sequence models or Transformers for suitable sequential tasks.
- Compare candidates with the same leakage-safe validation procedure.
- Select based on the metric that reflects the real-world objective, not accuracy alone.

---

## 🐍 18. Python Libraries

| Library | Purpose |
|---|---|
| NumPy | Numerical computing |
| Pandas | Data manipulation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Classical machine learning |
| XGBoost | Gradient boosting |
| LightGBM | Efficient gradient boosting |
| CatBoost | Gradient boosting with categorical-feature support |
| Statsmodels | Statistical modelling and time series |
| TensorFlow / Keras | Deep learning |
| PyTorch | Deep learning and research |
| Gymnasium | Reinforcement learning environments |
| Stable-Baselines3 | Reinforcement learning algorithms |
| Hugging Face Transformers | Transformer-based models |
| UMAP-learn | Nonlinear dimensionality reduction |

### Installation

Install the core packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Optional libraries can be installed when needed:

```bash
pip install xgboost lightgbm catboost statsmodels
pip install tensorflow torch
pip install umap-learn hdbscan
pip install gymnasium stable-baselines3
pip install transformers
```

---

## 🛣️ 19. Learning Roadmap

A suggested progression for studying Machine Learning:

### Beginner
- Python, NumPy, and Pandas
- Data preprocessing and visualization
- Linear and Logistic Regression
- KNN and Naive Bayes
- Train-test split and basic metrics

### Intermediate
- Decision Trees and Random Forests
- SVM
- Gradient Boosting and XGBoost
- Feature engineering and regularization
- Cross-validation and hyperparameter tuning
- PCA, LDA, and clustering

### Advanced
- Anomaly detection
- Time-series forecasting
- Recommendation systems
- Neural networks and deep learning
- Reinforcement learning
- Model deployment and monitoring

### Suggested projects
- House-price prediction
- Laptop-price prediction
- Customer churn classification
- Customer segmentation
- Spam detection
- Fraud detection
- Sales forecasting
- Image classification
- Object detection
- Autonomous navigation simulation

---

## 📚 20. References

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-learn Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/en/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/learn)
- [Hugging Face Documentation](https://huggingface.co/docs)

---

## ⭐ Final Note

This repository is intended as a learning and reference resource for Machine Learning concepts and algorithms. The best model is not necessarily the most complex one; it is the model that generalizes well, meets the task's requirements, and performs reliably on unseen data.

**Author:** Vijay Kumar Prajapat  
**Last Updated:** 20 October 2026

If you find this guide useful, consider starring the repository and contributing improvements.
