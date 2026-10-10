
# 🌍 AirGuard AI

*AirGuard AI* is a Python-based air quality monitoring and prediction application designed to analyze historical air pollution data, forecast pollution trends, and identify potential PM2.5 pollution spikes using machine learning. 🌱📊

---

🚀 Live Demo

Live Dashboard: https://airguard-ai-ally.onrender.com

## ✨ Features

- 🌍 Analyze historical air quality data for Jaipur, Rajasthan
- 📊 Explore air quality data through an interactive dashboard
- 📈 Forecast air quality trends using a SARIMA time-series model
- 🤖 Predict potential PM2.5 pollution spikes using Logistic Regression
- 🎯 Identify potential PM2.5 threshold exceedances of 60 µg/m³ within the following 24 hours
- 📉 Compare forecasting model results
- 📋 View saved model evaluation metrics
- 📊 Visualize forecasting results and prediction outputs
- 🧹 Clean and preprocess raw air quality data
- 🔍 Inspect timestamps and missing observations
- 💾 Store generated predictions and evaluation results
- 🖥️ Interactive web interface built using Streamlit
- ⚡ Modular scripts for data processing, training, and evaluation

---

## 🛠️ Technologies Used

- 💻 **Language:** Python
- 🌐 **Framework:** Streamlit
- 🐼 **Data Analysis:** Pandas, NumPy
- 🤖 **Machine Learning:** Scikit-learn
- 📈 **Time-Series Forecasting:** Statsmodels (SARIMA)
- 📊 **Data Visualization:** Matplotlib
- 💾 **Model Serialization:** Joblib
- 🧑‍💻 **IDE:** Visual Studio Code or another Python-compatible editor
- ⚙️ **Environment:** Python virtual environment
- 🌿 **Version Control:** Git and GitHub

---

## 📂 Project Structure

```text
AirGuard-AI/
|
|-- 📁 data/
|   |-- 📁 processed/
|   |   |-- 📄 jaipur_air_quality_hourly.csv
|   |
|   |-- 📁 raw/                 # Excluded from GitHub
|
|-- 📁 outputs/
|   |-- 📄 forecast_comparison_predictions.csv
|   |-- 📄 forecast_model_comparison.csv
|   |-- 📄 forecast_model_comparison.png
|   |-- 📄 sarima_forecast_plot.png
|   |-- 📄 sarima_metrics.csv
|   |-- 📄 sarima_test_forecast.csv
|   |-- 📄 spike_model.joblib
|   |-- 📄 spike_model_metrics.json
|   |-- 📄 spike_test_predictions.csv
|   |
|   |-- 📄 sarima_model.pkl   # Excluded from GitHub due to size
|
|-- 📁 pages/
|   |-- 📄 2_Model_Insights.py
|   |-- 📄 3_Live_Model_Predictions.py
|
|-- 📄 app.py
|-- 📄 check_data.py
|-- 📄 clean_data.py
|-- 📄 diagnose_timestamps.py
|-- 📄 download_data.py
|-- 📄 evaluate_forecast.py
|-- 📄 find_best_window.py
|-- 📄 find_training_period.py
|-- 📄 get_sensors.py
|-- 📄 test_setup.py
|-- 📄 train_sarima.py
|-- 📄 train_spike_model.py
|-- 📄 .gitignore
|-- 📄 README.md
|-- 📄 requirements.txt       # To be added
```

*Note: The raw JSON datasets and the large SARIMA model file are excluded from GitHub. The SARIMA model must be generated or made available separately before running features that depend on it.*

---

## 🚀 Getting Started

### 📋 Prerequisites

Make sure you have the following installed:

- 🐍 Python
- 📦 pip (Python package installer)
- 🌿 Git
- 🧑‍💻 Visual Studio Code or another Python-compatible editor

---

## 📥 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/1601devanshi/AirGuard-AI.git
```

### 2️⃣ Open the Project

- 📂 Open the terminal or command prompt
- 📁 Navigate to the cloned `AirGuard-AI` folder
- 🧪 Create a Python virtual environment
- ⏳ Activate the virtual environment
- 📦 Install the required Python dependencies

Create the virtual environment:

```bash
python -m venv .venv
```

For Windows, activate it using:

```bat
.venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install streamlit pandas numpy scikit-learn statsmodels matplotlib joblib
```

### 3️⃣ Run the Application

Start the Streamlit dashboard using:

```bash
streamlit run app.py
```

- 🌐 Open the local URL displayed in the terminal
- 📊 Explore the available dashboard pages
- 📈 View the forecasting and model insights sections

**Note:** The full application may require the processed dataset and trained model files. The SARIMA model artifact is not included in the repository, so make sure it is generated or provided before using dependent features.

---

## 🔔 How to Use

- 🌐 Open the AirGuard AI dashboard
- 📊 Explore the available air quality data and visualizations
- 📈 Review the generated forecasting results
- 🤖 Explore the pollution spike prediction functionality
- 🔍 Open the model insights page
- 📋 Review the saved evaluation metrics and model comparisons
- 📉 Examine the forecast plots and prediction outputs
- 🧹 Run the data cleaning scripts when preparing the dataset
- 🧪 Run the training scripts when you need to regenerate the models

*The exact options available depend on the application's current implementation and the required data and model files.*

---

## 🎯 Use Cases

AirGuard AI can support educational and analytical exploration of:

- 🌫️ Historical PM2.5 pollution trends
- 🌁 Historical PM10 pollution trends
- 📈 Short-term air quality forecasting experiments
- 🤖 Potential pollution spike identification
- 📊 Comparison of forecasting approaches
- 🔍 Analysis of missing air quality observations
- 🧪 Machine learning experimentation
- 🌍 Environmental data analysis
- 🎓 Academic projects involving data science and machine learning

*AirGuard AI is an analytical project and is not an official air quality monitoring or public warning service.*

---

## 🏗️ Build the Project

To prepare the environment and install the main dependencies:

```bash
python -m venv .venv
```

For Windows:

```bat
.venv\Scripts\activate
```

Install the packages:

```bash
pip install streamlit pandas numpy scikit-learn statsmodels matplotlib joblib
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📦 Model Artifacts

The project generates and uses model artifacts for forecasting and pollution spike detection.

- 📈 **SARIMA model:** `outputs/sarima_model.pkl`
- 🤖 **Spike detection model:** `outputs/spike_model.joblib`
- 📊 **Forecast metrics:** `outputs/sarima_metrics.csv`
- 📋 **Spike model metrics:** `outputs/spike_model_metrics.json`

The SARIMA model file is excluded from GitHub because it exceeds GitHub's standard 100 MB individual-file limit.

To regenerate the models, use the corresponding training scripts after ensuring that the required data and dependencies are available:

```bash
python train_sarima.py
python train_spike_model.py
```

*Model training time and successful execution depend on the available data, Python environment, and system resources.*

---

## 🧪 Testing

The project includes a setup-checking script:

```bash
python test_setup.py
```

Additional data inspection and evaluation scripts include:

```bash
python check_data.py
python diagnose_timestamps.py
python evaluate_forecast.py
```

These scripts can be used to inspect data and review evaluation outputs. A complete automated test suite has not been confirmed.

---

## 🔮 Future Enhancements

- 🌐 Deploy the Streamlit dashboard for public access
- 📡 Integrate a live air quality data API
- 📍 Add location-based air quality monitoring
- 📊 Improve forecasting model comparisons
- 🤖 Experiment with additional machine learning algorithms
- 🔔 Add configurable pollution threshold alerts
- 📈 Improve model validation and evaluation
- 📥 Add downloadable forecast reports
- 🧪 Add automated tests
- ⚙️ Add a verified `requirements.txt` for reproducible setup
- 🖼️ Add screenshots of the actual dashboard and model insights page

---

## 🤝 Contributing

Contributions and suggestions are welcome! ❤️

1. 🍴 Fork the repository.
2. 🌿 Create a new branch:

   ```bash
   git checkout -b feature/your-feature
   ```

3. ✨ Make your changes.
4. 💾 Commit your changes:

   ```bash
   git add .
   git commit -m "Describe your changes"
   ```

5. 🚀 Push your branch:

   ```bash
   git push origin feature/your-feature
   ```

6. 🔀 Open a Pull Request on GitHub.

Please make sure your changes are tested and follow the existing project structure.

---

## 👨‍💻 Author

# Devanshi Sharma

## 🌐 GitHub: [1601devanshi](https://github.com/1601devanshi)

## ⭐ Support

If you find AirGuard AI interesting or useful, consider giving the repository a ⭐ on GitHub. Your support is appreciated! ❤️

---

🌍 **AirGuard AI**

*Turning air quality data into actionable insights.* 🌱📊

