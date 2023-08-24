# Preprocessing World Data for Foreign Aid Analysis

*`(Creation Date: 8 / 21 / 2023)`*

## Introduction

This project aims to identify regions of the world that are most in need of external foreign aid to build infrastructure, health services, and education. The analysis is based on various socio-economic indicators, and the preprocessing module plays a crucial role in preparing the data for modeling.

## Data Preprocessing

### 1. Missing Values Handling

#### a. Imputation Based on Geolocation

* Created a function to impute missing values based on geolocation (latitude and longitude) and reference features from America's data.
* America was chosen as the reference point for imputation, considering its status as the world's hegemonic power.
* Linear regression models were used to predict missing values based on non-missing features.

#### b. Conversion of Currency and Percentage Columns

* Currency columns like 'GDP' and 'Minimum wage' were converted to numerical format by removing currency symbols and commas.
* Percentage columns were identified and converted to numerical format by removing the percentage symbol.

### 2. Scaling Weights Based on Global Wealth Inequality

* Proposed an idea to scale the weight of columns and rows based on global wealth inequality.
* This includes scaling up the weight for zones with lower minimum wages and higher importance for features like infant mortality rate and life expectancy.

### 3. Visualization and Analysis

* Analyzed visualization plots to understand missing values and feature distributions.
* Conducted a thorough review of the preprocessing module and made enhancements in the code organization and configuration.

## Conclusion

The preprocessing stage involved meticulous handling of missing values, data conversions, and the application of domain knowledge to create relevant features. The collaboration focused on step-by-step analysis, programming, and iterative refinements, ensuring that the data is well-prepared for subsequent modeling to achieve the project's goal of identifying regions in need of foreign aid.
