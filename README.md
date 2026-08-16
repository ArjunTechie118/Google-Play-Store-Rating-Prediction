# Google Play Store Rating Prediction using Machine Learning

An end-to-end machine learning project that predicts Google Play Store app ratings using regression models. This project demonstrates the complete data science workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, preprocessing, model building, hyperparameter tuning, cross-validation, and model evaluation using Scikit-learn pipelines.

---

## Project Objectives

- Perform comprehensive data preprocessing on a real-world dataset.
- Explore relationships between app features and ratings through EDA.
- Build a reusable machine learning pipeline.
- Compare regression models for rating prediction.
- Improve model performance using hyperparameter tuning.
- Evaluate model performance using multiple regression metrics.

---

## Dataset

The dataset contains information about Android applications published on the Google Play Store.

### Features

- App
- Category
- Rating (Target Variable)
- Reviews
- Size
- Installs
- Type
- Price
- Genres
- Content Rating
- Android Version
- Current Version
- Last Updated

---

# Project Workflow

## 1. Data Cleaning

The raw dataset was cleaned before model training.

Performed operations:

- Removed duplicate records
- Handled missing values
- Converted data types
- Converted `Installs` into numeric format
- Converted `Price` from string to float
- Removed unnecessary columns
- Prepared categorical and numerical features

---

## 2. Exploratory Data Analysis (EDA)

Several visualizations were created to understand the dataset before training.

Visualizations include:

- Rating Distribution
- Reviews Distribution
- Log Reviews Distribution
- Category Count Plot
- Scatter Plot (Log Reviews vs Rating)
- Scatter Plot (Log Installs vs Rating)
- Boxplots
- Correlation Heatmap
- Pair Plot

### Key Observations

- Most applications have ratings between **4.0 and 4.5**.
- Reviews and installs are heavily right-skewed.
- Log transformation significantly reduced skewness.
- Family is the largest application category.
- Numerical features show relatively weak linear correlation with ratings.

---

## 3. Feature Engineering

The following feature engineering techniques were applied.

### Created Log_Reviews

Large review counts were highly skewed. Applying a logarithmic transformation compressed extreme values and produced a more balanced distribution.

### Created Log_Installs

The install count also contained extreme values. Log transformation reduced skewness and improved feature representation.

### Converted Price into Float

Price values were originally stored as strings (e.g. "$2.99"). They were converted into numerical values so they could be used by machine learning algorithms.

---

## 4. Feature Selection

Different combinations of features were tested to identify the most useful predictors.

### Removed Features

- Reviews
- Current Ver
- Android Ver
- Last Updated

### Final Features

### Numerical Features

- Log_Installs
- Size
- Price

### Categorical Features

- Category
- Type
- Genres

---

## 5. Data Preprocessing

A complete Scikit-learn preprocessing pipeline was implemented.

### Numerical Pipeline

- Median Imputation
- StandardScaler

### Categorical Pipeline

- OneHotEncoder

Both pipelines were combined using **ColumnTransformer**, creating a reproducible preprocessing workflow.

---

## 6. Model Building

Two regression models were explored.

### Linear Regression

Linear Regression was trained as a baseline model. However, it produced a poor R² score, indicating that the available features do not have a strong linear relationship with app ratings.

### Random Forest Regressor

Random Forest Regressor was then implemented to capture non-linear relationships within the data.

---

## 7. Hyperparameter Tuning

Model performance was improved using **GridSearchCV**.

### Tuned Parameters

- Number of Estimators
- Maximum Depth
- Minimum Samples Split
- Minimum Samples Leaf

### Best Parameters

```python
{
    'regressor__max_depth': 10,
    'regressor__min_samples_leaf': 4,
    'regressor__min_samples_split': 10,
    'regressor__n_estimators': 100
}
```

---

# Model Performance

| Metric | Score |
|---------|-------|
| Train R² | **0.189** |
| Test R² | **0.089** |
| Cross Validation R² | **0.082** |
| Mean Absolute Error (MAE) | **0.297** |
| Mean Squared Error (MSE) | **0.198** |

---

## Conclusion

Although the tuned Random Forest model outperformed the baseline Linear Regression model, the overall R² score remained relatively low. This indicates that the available features explain only a limited portion of the variation in app ratings. The project demonstrates the complete machine learning workflow and highlights the importance of feature engineering, preprocessing, and model evaluation when working with real-world datasets.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git
- GitHub

---

# Repository Structure

```
Google-Play-Store-ML-Pipeline/
│
├── google_playstore_rating_prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── googleplaystore.csv
│
└── images/
```

---

# Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Engineering
- Feature Selection
- Machine Learning Pipelines
- ColumnTransformer
- OneHotEncoder
- StandardScaler
- Train-Test Split
- Random Forest Regression
- Hyperparameter Tuning
- Cross Validation
- Model Evaluation
- Git & GitHub

---

# Key Takeaways & Real-World Insights
While the Random Forest model successfully learned patterns within the structured metadata (achieving an $\text{MAE} = 0.297$), the low $R^2$ score ($0.089$) highlights a fundamental domain insight: quantitative app metrics (size, category, number of installs) alone cannot reliably predict subjective user satisfaction. 
App ratings are driven heavily by qualitative factors—such as user experience, recent bug fixes, and customer support responsiveness—which cannot be captured by metadata alone.

---

# Future Improvements

Potential improvements include:

- Exploring advanced regression models such as XGBoost and LightGBM.
- Engineering additional predictive features.
- Performing feature importance analysis.
- Deploying the trained model as a web application using Streamlit or Flask.

---

# Author

**Arjun Dadhich**

Computer Science Engineering Student

Interested in Data Science, Machine Learning, Data Engineering, and building practical machine learning solutions.
