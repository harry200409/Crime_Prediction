# Crime Case Status Prediction

##  Overview
This is a **machine learning-based web application** that predicts whether a crime case will be **closed** or **remain open**.  
The system analyzes various parameters such as **location, crime type, victim details, and police deployment** to make predictions about case closure probability.

---

##  Features
-  Predictive analysis of crime case status  
-  User-friendly web interface  
-  Real-time predictions  
-  Handles multiple input parameters:
  - City
  - Crime Code
  - Crime Description
  - Victim Age and Gender
  - Weapon Used
  - Crime Domain
  - Number of Police Deployed

---

##  Tech Stack
- **Python 3.x**
- **Flask** (Web Framework)
- **XGBoost** (Machine Learning Model)
- **scikit-learn**
- **NumPy & Pandas**
- **HTML/CSS**

---

##  Project Structure

```text
├── app.py # Main Flask application
├── model.pkl # Trained ML model
├── crime_dataset_india.xls # Dataset file
├── requirements.txt # Project dependencies
└── templates/
└── index.html # Web interface template
```


---

##  Installation

1️. **Clone the repository**
```bash
git clone https://github.com/yourusername/crime-case-status-prediction.git
cd crime-case-status-prediction
```
2. **Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate    # On Windows
source .venv/bin/activate # On Mac/Linux
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Dependencies

```bash
Flask==3.1.0
XGBoost==3.0.0
NumPy==2.2.4
Pandas==2.2.3
scikit-learn==1.6.1
gunicorn==23.0.0
```
- (Additional dependencies are listed in requirements.txt)

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```bash
http://localhost:5000
```

3. Enter the required case information in the form.
4. Submit to receive the prediction.

## Model Information

- The application uses an XGBoost classifier model trained on Indian crime data.
- It processes both numerical and categorical inputs using label encoding for categorical variables.

##  Development Notes

- Built using Flask for the backend API.
- A clean, responsive HTML/CSS interface for the frontend.
- The ML model is serialized using pickle and loaded when the application starts.
