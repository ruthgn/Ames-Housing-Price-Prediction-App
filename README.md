# Ames Housing Price Prediction App

This repository contains all the data and code used to build a machine learning web app to predict house prices in Ames, Iowa. 

Access the app → https://share.streamlit.io/ruthgn/ames-housing-price-prediction/main/ames-house-ml-app.py

The prediction model running on the web app ranks in the top 8% of Kaggle's [House Price Prediction Competition leaderboard](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) and top 1% of [Housing Prices Competition for Kaggle Learn Users leaderboard](https://www.kaggle.com/c/home-data-for-ml-course/overview) (as of 10/29/2021). Kaggle notebook outlining the model building process is available [here](https://www.kaggle.com/ruthgn/house-prices-top-8-featengineering-xgb-optuna/notebook).


Files
-----
* [The Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (train.csv and test.csv).
* Python object serialization of the trained prediction model running on the app (ames_house_xgb_model.pkl).
* List of all packages in the environment where this project was built and run (requirements.txt).
* Sample CSV files for inputing bulk entries on the app (sample_test.csv).
* Data dictionary containing all variable description and details (data_description.txt).
* Data dictionary containing labels representing different levels in categorical features and their human-readable counterparts (level_dictionary.csv).
* Code file of the app (ames-house-prediction-app.py).


Acknowledgment
-----
The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset.
