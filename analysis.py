#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#Feature Engineering - Take log transform of sale price to reduce skewness and help with regression
df["log_sale_price"] = np.log(df["sale_price"])

#Compare skew of sale price before and after log transform
print("Original sale price skew:", df["sale_price"].skew())
print("Log sale price skew:", df["log_sale_price"].skew())

#Feature Engineering - create price per square foot feature
df["price_per_sqft"] = df["sale_price"] / df["bldgarea"]

#Research Question 1: Does sale price differ significantly by borough?
#Compute median sale price by borough and create bar chart
borough_median = df.groupby("borough_y")["sale_price"].median() / 1000000

plt.bar(borough_median.index, borough_median.values)

plt.xlabel("Borough")
plt.ylabel("Median Sale Price (Millions $)")
plt.title("Median Sale Price by Borough")

plt.savefig("figures/median_price_by_borough.png")
plt.show()

#Compute average sale price by borough and create bar chart
borough_avg = df.groupby("borough_y")["sale_price"].mean() / 1000000

plt.bar(borough_avg.index, borough_avg.values)

plt.xlabel("Borough")
plt.ylabel("Average Sale Price (Millions $)")
plt.title("Average Sale Price by Borough")

plt.savefig("figures/average_price_by_borough.png")
plt.show()

#Research Question 2: Does the number of residential units affect sale price per square foot?



#Research Question 3: Is zip code a determinant of sale price?



#Research Question 4: Does building age affect sale price?



#Research Question 5: Can we accurately predict the sale price using building characteristics?