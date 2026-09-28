# Airbnb data analysis and price prediction

This project explores Airbnb listing data to understand pricing, availability, room types, and location-based trends. A machine learning model was also developed to predict Airbnb listing prices.

## Objectives

* Clean and explore Airbnb listing data
* Analyze pricing and availability trends
* Identify factors related to listing prices
* Build a machine learning model for price prediction
* Create a functional web app for making predictions

## Tools used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Streamlit

## Machine learning

A CatBoost regression model was trained on Amsterdam Airbnb listings (Inside
Airbnb dataset) using listing features such as host details, location,
capacity, availability, reviews, amenities count, room type (one-hot) and a
smoothed target-encoded neighbourhood value (24 features total). The target
was square-root transformed during training, so predictions are squared back
to currency at serving time. The model was evaluated using regression metrics
and then integrated into a Streamlit application.

## Web app

A functional Streamlit app (`app.py`) was created to allow users to enter
listing information and receive a predicted Airbnb nightly price from the
trained machine learning model. It loads, from this same folder:

* `airbnb_catboost_model.joblib` — trained CatBoostRegressor
* `neighbourhood_encoding_lookup.pkl` — neighbourhood → encoded value map
* `global_mean_price.pkl` — fallback for unseen neighbourhoods

## Project workflow

```text
Data cleaning
     ↓
Exploratory data analysis
     ↓
Feature engineering
     ↓
Model training
     ↓
Model evaluation
     ↓
Streamlit app
     ↓
Price prediction
```

## Project status

Completed data analysis, machine learning model, and functional prediction app.

## Run this project

The notebook (`AirBnB.ipynb`) loads the Inside Airbnb Amsterdam listings
from URL on first run, so no data file is stored in this repo.

```bash
pip install -r requirements.txt
streamlit run app.py
jupyter notebook AirBnB.ipynb
```
