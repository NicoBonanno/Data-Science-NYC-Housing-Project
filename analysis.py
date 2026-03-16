#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

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



#Research Question 5: Can we accurately predict the sale price using building characteristics?
# Remove extreme outliers for price, price per square foot and building area(top and bottom 1%)
lower = df["sale_price"].quantile(0.01)
upper = df["sale_price"].quantile(0.99)
df = df[(df["sale_price"] >= lower) & (df["sale_price"] <= upper)]

lower = df["price_per_sqft"].quantile(0.01)
upper = df["price_per_sqft"].quantile(0.99)
df = df[(df["price_per_sqft"] >= lower) & (df["price_per_sqft"] <= upper)]

lower = df["bldgarea"].quantile(0.01)
upper = df["bldgarea"].quantile(0.99)
df = df[(df["bldgarea"] >= lower) & (df["bldgarea"] <= upper)]

#Define features
features = [
    "bldgarea",
    "lotarea",
    "building_age",
    "unitsres",
    "numfloors",
    "resarea",
    "comarea",
    "price_per_sqft",
    "borough_x",
    "zip_code",
]

X = df[features]
y = df["sale_price"]

#Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Compute R² and RMSE for performance
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

#Plot the results
# Convert to millions for plotting
actual_m = y_test / 1_000_000
pred_m = y_pred / 1_000_000

plt.scatter(actual_m, pred_m, alpha=0.4, label="Predicted Prices")

# Perfect prediction line
plt.plot(
    [actual_m.min(), actual_m.max()],
    [actual_m.min(), actual_m.max()],
    color="red",
    label="Perfect Prediction"
)

plt.xlabel("Actual Sale Price (Millions $)")
plt.ylabel("Predicted Sale Price (Millions $)")
plt.title("Actual vs Predicted Sale Price")

plt.legend()
plt.savefig("figures/regression_act_vs_pred.png")
plt.show()