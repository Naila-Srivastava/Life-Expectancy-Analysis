# 📈 Life Expectancy Analysis 

This project uses Python, data visualization, and Streamlit to provide an interactive and insightful web-based dashboard that lets users explore the underlying patterns in life expectancy data.

## 🚀 Demo

👉 [Click here to view the live Streamlit app](https://your-streamlit-link.streamlit.app)  


## 🔍 Features in the App

- Sidebar filters: Year, Country, Status (Developed/Developing)
- Interactive line/bar charts
- Correlation heatmaps
- Country-level comparison
- Key metrics summary (mean life expectancy, outliers, etc.)
- Downloadable insights (coming soon 🔄)

## 🧪 To Run Locally

* git clone [https://github.com/Naila-Srivastava/Life-Expectancy-Analysis.git]
* cd `life-expectancy-analysis`
* pip `install -r Model requirements.txt`
* streamlit run `app.py`

## 📂 Model structure in this file

```plaintext
Life_Expectancy_Analysis/
│
├── models/
│   ├── life_expectancy_model.keras      # Trained Deep Learning (Keras) Model
│   └── scaler.pkl                       # The fitted scaler used for transforming features before training
│   └── Model requirements.txt           # Contains required packages and libraries to deploy the app
│   └── Model_README.md                  # The file you're reading rn
│   └── Model.png                        # Image of the trained model
├── app.py                               # App deployment script
