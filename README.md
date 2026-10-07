# SkyNest Airline Booking & Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-150458.svg)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-purple.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A customer booking intelligence project that combines airline booking operations, predictive modeling, and a premium flight-booking dashboard experience.

---

## ✈️ Project Overview

This repository blends two use cases into one airline-focused system:

1. A machine learning model that predicts whether a customer will complete a flight booking.
2. A polished booking dashboard where users can search flights, choose seats, review pricing, and complete a real-time-like booking flow.

The app is designed to mimic a modern airline operations dashboard with premium styling, live seat data, booking records, revenue tracking, and flight scheduling.

---

## 📌 What’s Included

- Flight operations dashboard with premium airline branding
- Real-time-like booking flow with route filtering and fare controls
- Seat selection and passenger information form
- Ancillary pricing for baggage, preferred seats, meals, and travel protection
- Booking confirmation and revenue tracking
- Flight tracking and schedule overview
- Conversion prediction using a trained Random Forest model
- Analytics and feature importance reporting

---

## 🧠 Machine Learning Component

The model is trained on the historical booking dataset in `customer_booking.csv` to predict whether a customer completes a booking.

### Model used
- Random Forest Classifier
- OneHotEncoder for categorical variables
- Pipeline-based preprocessing

### Business objective
- Understand which factors influence conversion
- Estimate customer intent from search behavior
- Highlight high-impact booking signals such as purchase lead time, route, trip type, and ancillaries

### Example model outcomes
- Accuracy: around 85%
- ROC-AUC: around 0.79
- Key drivers include purchase lead time, trip duration, route, and ancillary choices

---

## 🗂️ Dataset Overview

The raw dataset contains flight-search interaction data and conversion outcomes.

- Total rows: 50,000
- Target variable: `booking_complete`
- Predictors include:
  - `num_passengers`
  - `sales_channel`
  - `trip_type`
  - `purchase_lead`
  - `length_of_stay`
  - `flight_hour`
  - `flight_day`
  - `route`
  - `booking_origin`
  - `wants_extra_baggage`
  - `wants_preferred_seat`
  - `wants_in_flight_meals`
  - `flight_duration`

---

## 🚀 Features in the Dashboard

### Booking experience
- Search flights by route and fare cap
- Pick a flight from the active schedule
- Select seat numbers for passengers
- Review dynamic total pricing including extras
- Confirm booking with instant success feedback

### Operations monitoring
- Flight network overview
- Active route and schedule tracking
- Revenue and booking totals
- Occupancy and seat status
- Flight status board

### Prediction tools
- Real-time conversion scoring for customer searches
- Model analytics and feature importance charts
- Data quality monitoring

---

## 📁 Repository Structure

```text
.
├── app.py                     # Streamlit airline booking dashboard
├── booking_prediction.ipynb   # ML analysis notebook
├── customer_booking.csv       # Booking dataset
├── dashboard.html             # HTML mockup/dashboard concept
├── feature_importance.csv     # Feature importance output
├── model_metrics.csv          # Model evaluation metrics
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── LICENSE                    # License file (if present)
```

---

## 🛠️ Install & Run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Launch the dashboard

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🧪 Jupyter Notebook

To reproduce the analysis:

```bash
jupyter notebook booking_prediction.ipynb
```

Run the cells in order to explore the dataset, train the model, and generate visualizations.

---

## 🎯 Use Cases

This project can be used for:
- airline marketing optimization
- booking conversion analysis
- customer intent prediction
- flight operations and revenue tracking
- travel product and ancillaries recommendation

---

## 📜 License

This project is open-source and available under the MIT License.
