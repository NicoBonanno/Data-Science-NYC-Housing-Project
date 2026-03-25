#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#Read data
df = pd.read_csv("Dataset/nyc_housing_base.csv")

#Drop rows with missing values. 236 rows were dropped which is only about 0.7% of the dataset
df = df.dropna()

#Convert data types
df["zip_code"] = df["zip_code"].astype(int)
df["yearbuilt"] = df["yearbuilt"].astype(int)
df["unitsres"] = df["unitsres"].astype(int)
df["unitstotal"] = df["unitstotal"].astype(int)
df["numfloors"] = df["numfloors"].astype(int)
df["landuse"] = df["landuse"].astype(int)
df["building_age"] = df["building_age"].astype(int)

#Inspect data
pd.set_option("display.max_columns", None)
print(df.head())

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print((df["sale_price"] == 0).sum())
print((df["bldgarea"] == 0).sum())

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
#Get top 20 zip codes by number of sales for readability
top_zips = df["zip_code"].value_counts().head(20).index
df_top = df[df["zip_code"].isin(top_zips)]

#Use numpy to calculate median and mean sale price per zip code
zip_stats = df_top.groupby("zip_code")["sale_price"].agg(
    median=lambda x: np.median(x),
    mean=lambda x: np.mean(x)
).reset_index()

#Compute median sale price by zip code and create bar chart
zip_median = zip_stats.sort_values("median", ascending=False)
plt.bar(zip_median["zip_code"].astype(str), zip_median["median"] / 1000000)
plt.xlabel("Zip Code")
plt.ylabel("Median Sale Price (Millions $)")
plt.title("Median Sale Price by Zip Code (Top 20 by Volume)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/median_price_by_zipcode.png")
plt.show()

#Compute average sale price by zip code and create bar chart
zip_mean = zip_stats.sort_values("mean", ascending=False)
plt.bar(zip_mean["zip_code"].astype(str), zip_mean["mean"] / 1000000)
plt.xlabel("Zip Code")
plt.ylabel("Average Sale Price (Millions $)")
plt.title("Average Sale Price by Zip Code (Top 20 by Volume)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/average_price_by_zipcode.png")
plt.show()

#Research Question 4: Does building age affect sale price?



#Research Question 5: How are building area and residential units associated with sale price?
#Prepare data
df_reg = df.copy()

#Remove outliers in sale price and building area (top and bottom 5%)
price_lower = df_reg["sale_price"].quantile(0.05)
price_upper = df_reg["sale_price"].quantile(0.95)
df_reg = df_reg[
    (df_reg["sale_price"] >= price_lower) & (df_reg["sale_price"] <= price_upper)
].copy()

area_lower = df_reg["bldgarea"].quantile(0.05)
area_upper = df_reg["bldgarea"].quantile(0.95)
df_reg = df_reg[
    (df_reg["bldgarea"] >= area_lower) & (df_reg["bldgarea"] <= area_upper)
].copy()

#Create log-transformed variables to reduce skewness
df_reg["log_price"] = np.log(df_reg["sale_price"])
df_reg["log_bldgarea"] = np.log(df_reg["bldgarea"])

#Cap very extreme residential unit values for cleaner visualization
df_reg["unitsres_capped"] = df_reg["unitsres"].clip(
    upper=df_reg["unitsres"].quantile(0.99)
)

#Set features
features = ["log_bldgarea", "unitsres_capped"]

#Set target and predictors
X = df_reg[features]
y = df_reg["log_price"]

#Standardize predictors so coefficients are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Fit model on full filtered dataset
model = LinearRegression()
model.fit(X_scaled, y)

#Print intercept and coefficients
print("Intercept:", model.intercept_)

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nRegression Coefficients:")
print(coefficients)

#Plot building area trend
df_area = df_reg.copy()

df_area["bin"] = pd.qcut(df_area["log_bldgarea"], q=20, duplicates="drop")
grouped = df_area.groupby("bin")["log_price"].median()

centers = np.array([interval.mid for interval in grouped.index], dtype=float)

plt.figure()
plt.plot(centers, grouped.values, marker="o")
plt.xlabel("Log Building Area")
plt.ylabel("Log Sale Price")
plt.title("Trend of Log Sale Price vs Log Building Area")
plt.savefig("figures/trend_log_price_vs_area.png")
plt.show()

#Plot residential units trend
df_units = df_reg.copy()

df_units["bin"] = pd.qcut(df_units["unitsres_capped"], q=20, duplicates="drop")
grouped = df_units.groupby("bin", observed=False)["log_price"].median()

centers = np.array([interval.mid for interval in grouped.index], dtype=float)

plt.figure()
plt.plot(centers, grouped.values, marker="o")
plt.xlabel("Residential Units")
plt.ylabel("Log Sale Price")
plt.title("Trend of Log Sale Price vs Residential Units")
plt.savefig("figures/trend_log_price_vs_unitsres.png")
plt.show()