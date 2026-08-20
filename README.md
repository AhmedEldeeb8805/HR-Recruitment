# HR Recruitment

A Machine Learning project for analyzing recruitment data and predicting whether a candidate is likely to be looking for a new job opportunity.

The project follows an end-to-end Machine Learning workflow, including data cleaning, exploratory data analysis, feature preprocessing, model training, evaluation, candidate ranking, and business-oriented insights.

## Project Overview

Recruitment teams work with large amounts of candidate information, making it useful to identify patterns associated with job-search behavior.

This project uses candidate-related features to build classification models that predict the `target` variable, where:

* `0` → Candidate is not looking for a new job
* `1` → Candidate is looking for a new job

Two classification models are developed and compared:

* Logistic Regression
* Random Forest Classifier

The project also includes a recruitment analytics dashboard and a candidate-ranking analysis based on predicted job-change probability.

## Dataset

The project uses the `aug_train.csv` dataset.

### Dataset Size

* Original records: **19,158**
* Features: **14**
* Records after missing-value handling: **18,014**

### Main Features

The dataset contains information related to:

* City development index
* Gender
* Relevant experience
* University enrollment
* Education level
* Major discipline
* Professional experience
* Company size
* Company type
* Previous job change
* Training hours

The target variable represents whether the candidate is looking for a new job opportunity.

## Machine Learning Workflow

### 1. Data Loading & Exploration

The dataset is loaded using Pandas and inspected using:

* `info()`
* `describe()`
* Missing-value analysis
* Unique-value analysis
* Feature distributions

### 2. Missing-Value Handling

Missing values are handled using different strategies depending on the feature.

For example:

* `company_size` → filled using the mode and converted into numerical categories
* `gender` → filled using the mode
* `company_type` → filled using the mode
* `major_discipline` → filled using the mode
* Remaining incomplete records → removed using `dropna()`

This results in **18,014 records** available for modeling.

### 3. Feature Preprocessing

The following columns are removed from the modeling features:

```text
enrollee_id
city
gender
```

Categorical variables are transformed using **One-Hot Encoding** with `pd.get_dummies()`.

The following features are encoded:

```text
major_discipline
education_level
enrolled_university
relevent_experience
company_type
experience
```

The `company_size` feature is mapped from categorical ranges to numerical values.

`last_new_job` is also converted into a numerical representation.

After preprocessing, the dataset contains **49 features**.

### 4. Train/Test Split

The data is split into training and testing sets using:

```python
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This creates an **80/20 train-test split** while preserving the target distribution.

### 5. Feature Scaling

Standardization is applied to:

```text
city_development_index
company_size
last_new_job
training_hours
```

The mean and standard deviation are calculated from the training data and then applied to both training and testing data.

## Models

### Logistic Regression

The first baseline model is Logistic Regression with:

```python
LogisticRegression(
    max_iter=10000,
    random_state=42,
    class_weight="balanced"
)
```

### Random Forest

The second model is a Random Forest Classifier configured with:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)
```

The `class_weight="balanced"` option is used to account for the imbalance between the two target classes.

## Model Performance

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Results

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     71.00% |     43.92% |     65.72% |     52.65% |
| **Random Forest**   | **75.83%** | **50.54%** | **68.44%** | **58.15%** |

Based on these results, **Random Forest performed better across all reported metrics**.

It achieved:

* +4.83 percentage points in Accuracy
* +6.62 percentage points in Precision
* +2.72 percentage points in Recall
* +5.50 percentage points in F1 Score

compared with Logistic Regression.

## Recruitment Analytics

In addition to model evaluation, the project includes a recruitment analytics dashboard that visualizes:

* Candidate job-search status
* Job-search rate by city development index
* Job-search rate by experience
* Job-search rate by education level
* Model performance comparison
* Confusion matrices

These visualizations help connect the Machine Learning results with recruitment-related analysis.

## Candidate Ranking

The Random Forest model is also used to calculate a predicted probability of job change for candidates in the test set.

The project generates a **Top 10 candidate ranking** based on the predicted probability.

The highest predicted probability among the displayed top candidates is approximately **89.17%**.

This provides a practical way to prioritize candidates who may have a higher probability of looking for a new opportunity.

## Feature Importance

The project uses the Random Forest model's `feature_importances_` to identify the most influential features in the trained model.

A feature-importance visualization is included in the notebook to help understand which variables contribute most to the model's predictions.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
HR-Recruitment/
│
├── HR_Recruitment.ipynb
├── aug_train.csv
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AhmedEldeeb8805/HR-Recruitment.git
```

Navigate to the project directory:

```bash
cd HR-Recruitment
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
HR_Recruitment.ipynb
```

Run the notebook cells sequentially to reproduce the data preprocessing, analysis, model training, evaluation, dashboard, and candidate ranking.

## Key Machine Learning Concepts

This project demonstrates practical experience with:

* Exploratory Data Analysis
* Missing-Value Handling
* Categorical Data Encoding
* Feature Engineering
* Feature Scaling
* Train/Test Splitting
* Classification
* Logistic Regression
* Random Forest
* Class Imbalance Handling
* Model Evaluation
* Confusion Matrices
* Feature Importance
* Probability-Based Candidate Ranking
* Data Visualization

## Future Improvements

Potential improvements for a future version include:

* Hyperparameter tuning
* Cross-validation
* Testing additional classification algorithms
* More advanced feature engineering
* Improving class-imbalance handling
* Model explainability using SHAP
* Saving the trained model for inference
* Deploying the model as a web application or API

## Author

**Ahmed Eldeeb**

GitHub: [AhmedEldeeb8805](https://github.com/AhmedEldeeb8805)

---

⭐ If you found this project useful, feel free to explore the notebook and the complete Machine Learning workflow.
