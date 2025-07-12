# Life Expectancy Analysis 

Predicting life expectancy using traditional Machine Learning and Deep Learning models, wrapped in a clean Streamlit app for real-time prediction. 

## 🧠 Objectives

- To analyze and visualize global health indicators that impact life expectancy.
- To build and evaluate multiple Machine Learning and Deep Learning models to accurately predict life expectancy.
- To deploy an interactive Streamlit app for real-time prediction.
- Predicting the variables that actually affect the life expectancy

## 🔧 Tools & Technologies

- **Python** (`CSV`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `SciPy`, `Scikit-Learn`, `XGBoost`, `Tensorflow` & `Scikit-Learn`)
- **Streamlit** – For building an interactive web app  
- **Jupyter Notebook** – For code development and EDA  
- **Git & GitHub** – For version control and collaboration 

## 📂 Dataset

- The dataset includes various socio-economic and health indicators for countries across years (like GDP, schooling, mortality rates, immunization rates, etc.)
- Due to privacy and size constraints, the full dataset is **not included** in this repository.

➡️ [Download dataset here](https://www.kaggle.com/datasets/nailasrivastava/life-expectancy-analysis)

## 💻 How to Run

1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Open the notebooks: `EDA_Life_Expect.ipynb`, `ML_Life_Expect.ipynb` and `DL_Life_Expect.ipynb`  
4. Run all the cells and enjoy the visuals!

## 🤖 Models and their Features

1. Feature Engineering: Outlier detection, Imputation, Normalization
2. Multicollinearity check (VIF) and PCA
3. Machine Learning: Linear Regression, Random Forest, XGBoost (with GridSearchCV)
4. Evaluation Metrics: R² Score, MAE, RMSE, Loss Curves, Validation Loss Tracking
5. Deep Learning: Multi-layered Neural Network using Keras & Tensorflow

## 🧠 Model Evaluation Summary

| Model           | R² Score | MAE   | RMSE  |
|-----------------|----------|-------|-------|
| Linear Reg.     | 0.78     | 2.34  | 2.91  |
| Random Forest   | 0.93     | 1.22  | 1.65  |
| XGBoost         | 0.935    | 1.19  | 1.62  |
| Neural Network  | 0.91     | 1.35  | 1.75  |

➡️ *XGBoost and Random Forest gave the best balance between interpretability and accuracy.*

## 📊 Key Analysis

- Strong negative correlation between infant mortality, HIV/AIDS prevalence, and life expectancy.
- Positive impact of GDP, schooling, and health expenditure on life expectancy.
- Clear regional clusters in healthcare access and outcomes.

## 📈 Sample Visualizations

- **Correlation Heatmap**- To identify multicollinearity and relationships between features.
- **Distribution Plots**- To understand the skewness, central tendencies, and spread of key variables.
- **Outlier Detection via Boxplots**- Helped shape data preprocessing.
- **Hypothesis Testing (t-tests)**- Checked significance of means across grouped data.
- **Predicted vs Actual Plot**- To evaluate model accuracy.
- **Training & Validation Loss Curves**- For visualizing model convergence and overfitting signs.

## 🌍 Advanced Visuals

- Neural Network Training Curves
- Dimensionality Reduction via PCA
- Feature Importance (Random Forest & XGBoost)
- Streamlit Live Inference Interface

## 📝 Key Takeaways

* Socio-economic features like **Schooling**, **GDP**, and **Income Composition** have a **strong positive impact** on life expectancy.
* Health-related indicators such as **Adult Mortality**, **HIV/AIDS**, and **Infant Deaths** show a **strong negative correlation**.
* **Multicollinearity** was found between some features (e.g., infant deaths & under-five deaths), and was addressed using VIF filtering and PCA.
* Ensemble methods like **Random Forest** and **XGBoost** performed **exceptionally well** with minimal tuning.
* **Deep Learning** showed competitive performance after early stopping and validation tuning.
* Visualisations helped uncover **non-obvious relationships**, such as expenditure outliers in some countries with low life expectancy.
* **PCA** was tested for dimensionality reduction but slightly reduced model accuracy.
* Our **Streamlit app** makes the entire model pipeline **accessible for real-time prediction**.
* Deep Learning models showed competitive performance compared to ensemble methods, but simpler models like Random Forest held up surprisingly well.
* PCA didn’t improve prediction but raw engineered features worked best.
* Model deployment adds a practical layer to any predictive analysis project.

## 🧩 What’s Next?

1. Implement SHAP or LIME for model explainability
2. Add time-series trend analysis (multi-year tracking)
3. Extend app with batch predictions via CSV upload

## 🙌 Acknowledgments

Huge thanks to the open-source community, and Streamlit for making deployment ridiculously fun.
This project was completed as part of a 12-week Upskilling Sprint (Data Analyst Internship).

Massive gratitude to:
* Deeksha Russell and Duan Wang who collected the dataset from the WHO and the United Nations website.

## 📂 Project structure

```plaintext
Life-Expectancy-Analysis/
│
├── app.py                                              # Streamlit application script
├── life_expectancy_model.keras                         # Trained DL model
├── scaler.pkl                                          # Feature scaler object
├── requirements.txt                                    # Dependencies list
├── Life_Expectancy_Data.csv                            # Raw dataset
├── life_expectancy_cleaned_dataset.csv                 # Cleaned dataset
├── new_df.csv                                          # Dataset for ML and Deep Learning analysis 
├── EDA_Life_Expect.ipynb                               # EDA notebook
├── ML_Life_Expect.ipynb                                # Machine Learning notebook
├── DL_Life_Expect.ipynb                                # Deep Learning notebook
├── plots/                                              # Folder for all generated plots
├── README.md                                           # You're reading this now 
└── .gitignore

