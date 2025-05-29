# Crime Case Prediction

A machine learning web application that predicts whether a crime case will be closed or remain open based on various input parameters. The application uses an XGBoost model for predictions and Flask for the web interface.

## Features

- Predicts crime case status (Closed/Not Closed)
- User-friendly web interface
- Handles multiple input parameters:
  - City
  - Crime Code
  - Crime Description
  - Victim Age
  - Victim Gender
  - Weapon Used
  - Crime Domain
  - Number of Police Deployed

## Technologies Used

- Python
- Flask (Web Framework)
- XGBoost (Machine Learning Model)
- scikit-learn (For Label Encoding)
- NumPy (For Numerical Operations)
- HTML/CSS (Frontend)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`
3. Fill in the required information in the form
4. Click the submit button to get the prediction

## Project Structure

```
├── app.py                 # Flask application file
├── model.pkl              # Trained XGBoost model
├── requirements.txt       # Project dependencies
├── templates/
│   └── index.html         # HTML template for the web interface
```

## Model Information

The project uses an XGBoost classifier model that has been trained on crime data. The model takes various features as input and predicts whether a case will be closed or remain open. The input features are preprocessed using Label Encoding for categorical variables before making predictions.
