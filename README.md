# Data-Science-NYC-Housing-Project
Nico Bonanno, Jake Samela, Emraan Kafihi

This is the final project for our Data Science Fundamentals course (COMP 3125) at Wentworth Institute of Technology. Our project is analyzing New York City housing data to determine the impact of various building characteristics on the sale price of a property.

## Introduction
#### Why was the project undertaken?
We chose to do our project on this topic because we had a shared interest in economics, and housing prices are a significant facet of the economy. We wanted to apply the techniques learned in class to a meaningful, real-world dataset in order to understand how various building characteristics influence housing prices in New York City.

#### Research Questions:
1) Do sale prices differ significantly by borough?
2) Does the number of residential units affect sale price per square foot?
3) Is zip code a determinant of sale price?
4) Does building age affect sale price?
5) How are building area and residential units associated with sale price?

#### Purpose:
The purpose of our research was to determine the answers to the above questions and to gather information on the housing situation in New York City, one of the largest cities in the world. We aimed to test whether factors such as location and building characteristics have a measurable impact on sale price. 

## Selection of Data
We selected a dataset from Kaggle titled NYC Housing Prices. The dataset is included in this repo and can also be found here: https://www.kaggle.com/datasets/ishank2005/nyc-housing-prices-csv.

We chose this dataset because it is large and contains many useful features related to property characteristics and location. The dataset originally contained 34,439 rows, of which 236 were dropped due to missing values (NaN). After cleaning, the dataset contained 34,203 rows.

The dataset contains 19 columns, which are:
borough_x, block, lot, sale_price, zip_code, borough_y, yearbuilt, lotarea, bldgarea, resarea, comarea, unitsres, unitstotal, numfloors, latitude, longitude, landuse, bldgclass, and building_age.

#### Data Preview
![Data Preview](figures/raw_data_example.png)

#### Data Cleaning/Feature Engineering
We created one additional feature to improve the analysis:
- Price per square foot, calculated as sale price divided by building area.

The data was cleaned by removing all rows with missing values. This resulted in the removal of 236 rows, representing only about 0.7% of the dataset, so the impact on the overall dataset was minimal.

Additionally, several columns including zip_code, yearbuilt, and numfloors were converted from float values to integers to better represent categorical identifiers and count-based variables. We also verified that there were no rows containing negative sale prices or building areas less than or equal to zero, ensuring the data was valid for analysis.

## Methods
#### Tools:
- Python for writing code
- Pandas and Numpy for data analysis and manipluation
- Matplotlib for creating visuals
- Scikit-learn (sklearn) for implementing the linear regression model
- Github for version control
- VS Code as IDE

#### Analytical Methods
To answer the research questions, we used techniques including grouping, filtering, and aggregation using the Pandas library. Visualizations were created using Matplotlib to compare property characteristics across boroughs and other variables. For the machine learning component of the project, a linear regression model was applied to analyze the relationship between building characteristics and sale price. The model was used to examine how selected variables are associated with housing prices rather than to make predictions.

#### Research Question Distribution
- RQ1: Nico
- RQ2: Jake
- RQ3: Emraan
- RQ4: Emraan
- RQ5: Nico/Jake

## Results
#### Research Question 1: Do sale prices differ significantly by borough?
To answer this question, we calculated both the median and average sale prices for properties in each of the five boroughs.

![Median Sale Price](figures/median_price_by_borough.png)

![Average Sale Price](figures/average_price_by_borough.png)

The results show clear differences in property values across the boroughs. Manhattan and Brooklyn have higher median sale prices compared to the Bronx, Queens, and Staten Island. A similar pattern appears when examining the average sale prices, where Manhattan and Brooklyn also have the highest values among the five boroughs. These results indicate that property sale prices vary across boroughs in New York City.

#### Research Question 2: Does the number of residential units affect sale price per square foot?
To answer this question, the first thing we did was create a sale price per square foot metric. This was done by taking the sale price and dividing it by the building area. There are three plots shown below to analyze this. First is the units vs price per square foot, the second is the same scatterplot, but with the log of the price per square foot to reduce skewness. The last is a bin model of the number of residential units that are binned in 0, 1, 2, 5, 10, 20, 50, 100, and 500, and plotted is the resulting output of price per square foot.

![Units vs Log Price per Square Foot](figures/units_vs_price_per_square_foot_no_log.png)

![Units Vs Price per Square Foot](figures/units_vs_log_price_per_square_foot.png)

![Units vs price per square foot bins](figures/units_vs_price_per_square_foot_bin.png)

As illustrated by these results, as the number of residential units increases the price decreases. This trend is present in all three of the figures. This pattern may be explained by real-world factors. Intuitively, people may prefer to live in less densely populated environments, even within a city, which can lead to higher prices per square foot for properties with fewer residential units.

#### Research Question 3: Is zip code a determinant of sale price?
To answer this question, we calculated both the median and average sale prices for properties within the top 20 ZIP codes by sales volume. The dataset contains 180 unique ZIP codes, and plotting all of them would produce an overly cluttered visualization. Therefore, the analysis focuses on the 20 ZIP codes with the highest sales volume.

![Median Sale Price](figures/median_price_by_zipcode.png)

![Average Sale Price](figures/average_price_by_zipcode.png)

The results show clear variation in property values across ZIP codes. Many of the ZIP codes with the highest median and average sale prices fall within the 100xx and 112xx ranges, which correspond to Manhattan and Brooklyn, while ZIP codes from other boroughs generally show lower property values.

#### Research Question 4: Does building age affect sale price?


#### Research Question 5: How are building area and residential units associated with sale price?
To answer this question, we applied linear regression models to examine how building area and the number of residential units are associated with sale price. Because sale price was highly skewed, the logarithm of sale price was used as the response variable. Building area was also log-transformed to improve the relationship with price. We also removed the top and bottom five percent of the sale price and building area variables to remove outliers. Extreme values in the residential units variable were capped by removing the top one percent.

Three regression models were created for this analysis. The first model used only building area as a predictor, the second used only the number of residential units, and the third used both variables together. This allowed us to compare how well each variable explains variation in sale price both individually and combined.

![Building Area Regression](figures/regression_log_price_vs_area.png)

The figure above shows the regression of log sale price on building area. The relationship appears weak, with a slight overall trend but a large amount of variation in the data.

![Residential Units Regression](figures/regression_log_price_vs_unitsres.png)

The figure above shows the regression of log sale price on the number of residential units. This relationship is clearer, showing a negative trend where sale price tends to decrease as the number of units increases.

The R² values for the three models were relatively low. The model using building area alone had an R² of approximately 0.066, the model using residential units alone had an R² of approximately 0.106, and the model using both variables had an R² of approximately 0.108. These results indicate that neither variable strongly explains variation in sale price, and that combining them only slightly improves the model.

Overall, the regression results suggest that building area and residential units are associated with sale price, but they are not strong predictors on their own. Residential units show a slightly more consistent relationship, while building area has a weaker and more variable effect. These findings suggest that other factors, particularly location-based variables, likely play a more significant role in determining housing prices.

## Discussion
The results of this analysis provide insight into the factors that influence housing prices in New York City and highlight the relative importance of different types of variables. Location-based factors such as borough and ZIP code showed strong and consistent differences in sale price, indicating that geographic location plays a major role in determining property values. This is important because it suggests that where a property is located may have a greater impact on price than certain structural characteristics of the building itself.

The regression analysis examined how building area and the number of residential units are associated with sale price using three separate models. While both variables showed some relationship with sale price, the overall explanatory power of the models was low. The R² values for all three models were relatively small, indicating that these variables alone do not strongly explain variation in housing prices. Although the number of residential units showed a slightly stronger relationship than building area, combining both variables resulted in only a minimal improvement in the model.

These findings are consistent with general expectations in housing market research, where location is often considered one of the most important determinants of property value. While structural characteristics such as size and number of units do influence price, they are typically not sufficient on their own to fully explain differences in property values.

The results also suggest that additional factors not included in this analysis, such as neighborhood quality, proximity to amenities, property condition, and broader economic conditions, likely play a significant role in determining housing prices. This highlights the complexity of the housing market and the limitations of simple regression models using a small number of variables.

For future research, incorporating additional features such as building classification or economic indicators would likely improve the explanatory power of the models. More advanced modeling techniques could also be explored to better capture relationships and interactions between variables.

From a tools and methods perspective, this project demonstrated the effectiveness of Python libraries such as Pandas and NumPy for data cleaning and manipulation, Matplotlib for visualization, and Scikit-learn for implementing regression models. These tools allowed for efficient data analysis and helped translate raw data into meaningful insights about housing prices.

Overall, the results emphasize that housing price determination is influenced by many interconnected factors, and that simple models provide useful but limited insight into these relationships.

## Summary
This project analyzed New York City housing data to understand how location and building characteristics influence sale price. The results showed that property values vary significantly across boroughs and ZIP codes, with Manhattan and Brooklyn generally having higher prices than other areas.

The analysis also found that as the number of residential units increases, the price per square foot tends to decrease. Additionally, regression results indicated that building area and residential units are associated with sale price, but neither variable alone strongly explains price variation.

Overall, the findings suggest that while structural characteristics do have some influence, location plays a much more significant role in determining housing prices. These results highlight the complexity of the housing market and the importance of considering multiple factors when analyzing property values.
