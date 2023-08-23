### Financial Aid Prediction for Climate Disaster Aversion

#### 1. **Problem Definition**

- **Objective** : Identify regions that require financial aid from wealthier nations, with priority given to societal overhaul through the endorsement of materials and labor.
- **Model Choice**: Gaussian Mixture Model (GMM).

#### 2. **Data Exploration & Preprocessing**

- **Source**: `world-data-2023.csv`.
- **Features**: CO2 emissions, agricultural land, population, urban population, tax revenue, unemployment rate, healthcare expenditure, latitude, longitude.
- **Transformations**:
  - **Percentage Conversion**: Transformed percentage values to numerical format.
  - **Missing Values**: Filled with the median.
  - **Normalization**: Scaled features to [0, 100].

#### 3. **Model Selection**

- **K-Means**:
  - Pros: Simple, fast.
  - Cons: Assumes spherical clusters, hard clustering.
- **Gaussian Mixture Model (GMM)**:
  - Pros: Flexibility in cluster shapes, soft clustering.
  - Cons: Computationally intensive.
- **Decision**: GMM, due to its ability to capture complex relationships.

#### 4. **Model Development (FinancialAidGMM)**

- **Training**: 80% of preprocessed data.
- **Components**: 3.
- **Prediction**: Clustered regions.
- **Visualization**: Pairplot of clusters.
- **Saving**: Model as `FinancialAidGMM_model.pkl`, visualization as `clusters_visualization.png`.

---

#### Areas of Highest Concern:

* **Agricultural Land(%):** Countries with a high percentage of agricultural land may indicate a reliance on agriculture and potential need for industrial development.
* **Unemployment Rate:** A high unemployment rate may signal economic distress and a need for job creation through infrastructure projects.
* **Tax Revenue(%):** Low tax revenue may indicate a limited ability to finance public projects, including infrastructure development.
* **Out of Pocket Health Expenditure:** High out-of-pocket health expenditure may suggest a lack of public health infrastructure.
* **Life Expectancy, Infant Mortality, Maternal Mortality Ratio:** These health indicators can provide insights into the general well-being of the population and the need for healthcare infrastructure.

  ---

##### See development documentation in [data](data/).