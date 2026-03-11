# Data-Science-NYC-Housing-Project
Nico Bonanno, Jake Samela, Emraan Kafihi

This is the final project for our Data Science Fundamentals course at Wentworth Institute of Technology. Our project is analyzing New York City housing data to answer the five research questions below.

## Introduction
- Why was the project undertaken?
We chose to do our project on this topic because we had a shared interest in economics, and housing prices are a significant facet of the economy. We wanted to apply the techniques we have learned in class and through research to a meaningful, real-world project that would allow us to understand the role some of the factors outlined below play in the housing market.

- Research Questions:
1) Do sale prices differ significantly by borough?
2) Does the number of residential units affect sale price per square foot?
3) Is zip code a determinant of sale price?
4) Does building age affect sale price?
5) Can we accurately predict the sale price using building characteristics?

- Purpose:
The purpose of our research was to determine the answers to the above questions and to gather information on the housing situation in New York City, one of the largest cities in the world. We wanted to test whether some of these factors such as the borough and the building age of the house had an impact on sale price. 

## Selection of Data
We selected a datsaet from Kaggle titled NYC Housing Prices. The dataset is included in this repo and can also be found here: https://www.kaggle.com/datasets/ishank2005/nyc-housing-prices-csv.

We chose this dataset because it is large and contains many useful features related to property characteristics and location. The dataset originally contained 34,439 rows, of which 236 were dropped due to missing values (NaN). After cleaning, the dataset contained 34,203 rows.

The dataset contains 19 columns, which are:
borough_x, block, lot, sale_price, zip_code, borough_y, yearbuilt, lotarea, bldgarea, resarea, comarea, unitsres, unitstotal, numfloors, latitude, longitude, landuse, bldgclass, and building_age.

![Data Preview](figures/raw_data_example.png)

### Data Cleaning/Preview
We created two additional features to improve the analysis:
- Price per square foot, calculated as sale price divided by building area.
- Log-transformed sale price, which was created to reduce skew in the sale price distribution.

The sale price variable initially had a skew of about 3.29, indicating a highly right-skewed distribution. After applying the log transformation, the skew was reduced to -0.04, making the distribution more symmetric and better for regression and analysis.

The data was cleaned by removing all rows with missing values. This resulted in the removal of 236 rows, representing only about 0.7% of the dataset, so the impact on the overall dataset was minimal.

Additionally, several columns including zip_code, yearbuilt, and numfloors were converted from float values to integers to better represent categorical identifiers and count-based variables. We also verified that there were no rows containing negative sale prices or building areas less than or equal to zero, ensuring the data was valid for analysis.

## Methods


## Results


## Discussion


## Summary