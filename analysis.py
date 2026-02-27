#Imports
import numpy as np
import pandas as pd

#Read data
df = pd.read_csv("Dataset/nyc_housing_base.csv")

#Drop rows with missing values. 236 rows were dropped which is only about 0.7% of the dataset
df = df.dropna()

#Inspect data
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print((df["sale_price"] == 0).sum())
print((df["bldgarea"] == 0).sum())

print("Sale price skew:", df["sale_price"].skew())

#Take log transform of sale price to reduce skewness and help with regression
df["log_sale_price"] = np.log(df["sale_price"])

#Compare skew of sale price before and after log transform
print("Original sale price skew:", df["sale_price"].skew())
print("Log sale price skew:", df["log_sale_price"].skew())

#Feature engineering - create price per square foot feature
df["price_per_sqft"] = df["sale_price"] / df["bldgarea"]