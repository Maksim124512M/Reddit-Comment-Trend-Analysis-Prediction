# Reddit Comment Trend Analysis & Prediction

Analyze historical Reddit comment activity, visualize yearly trends, and predict future comment volumes using Linear Regression.

## Dataset
The project uses the dataset: `the-reddit-dataset-dataset-comments.csv`  - https://www.kaggle.com/datasets/pavellexyr/the-reddit-dataset-dataset <br>
It contains Reddit comments with timestamps, sentiment scores, and text content.

## Project Structure
├── data_preparing_and_visualization.py   *# Data cleaning and preparation*  <br>
├── prediction.py  *# Trend prediction and plotting*  <br>
├── the-reddit-dataset-dataset-comments.csv  <br>
└── README.md  *# Project description*  <br>

## Features
- Clean and prepare the dataset (handle missing values, convert timestamps)
- Perform EDA: count comments per year, plot yearly trends
- Predict future comment activity using Linear Regression
- Visualize results with annotated bar plots

## Technologies
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
