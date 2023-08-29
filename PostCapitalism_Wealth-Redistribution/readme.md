### Financial Aid Prediction for Climate Disaster Aversion

#### 1. **Introduction**
- **Objective**: Identify regions that require financial aid from wealthier nations, with major support from the USA, Russia, and China, who are the core economic facilitators on Earth today and for the foreseeable future.
- **Approach**: Utilize statistical models to analyze global data, pinpointing regions in need of aid.
- **Model Choice**: Gaussian Mixture Model (GMM) for clustering regions based on economic, health, and social indicators.

#### 2. **Data Exploration & Preprocessing**
- **Source**: Global data including economic and social factors ([world-data-2023.csv](world-data-2023.csv)).
- **Preprocessing**: Cleaning, normalization, and imputation. Refer to [preprocessing.py](preprocessing.py).
- **Outcome**: Prepared data for model training ([preprocessed_world-data-2023_22-08-2023_21-55-43.csv](output/processed-data/preprocessed_world-data-2023_22-08-2023_21-55-43.csv)).

#### 3. **Model Selection & Training**
- **Algorithm**: Gaussian Mixture Model (GMM) for flexible clustering.
- **Training Process**: 80% of preprocessed data. Refer to [finaid_train.py](finaid_train.py).
- **Model Outcome**: Trained model file [FinancialAidGMM_model.pkl](data/output/models/FinancialAidGMM_model.pkl).

#### 4. **Visualization & Analysis**
- **Visualization**: Cluster analysis images in [output/images](output/images).
- **Prediction Script**: For new data prediction, refer to [predict.py](predict.py).

#### 5. **Code Structure**
- **Configuration**: Centralized settings in [config.py](config.py).

#### 6. **Prioritized Features**
- **Agricultural Land(%):** Reliance on agriculture.
- **Unemployment Rate:** Economic distress signal.
- **Tax Revenue(%):** Financing ability.
- **Out of Pocket Health Expenditure:** Public health infrastructure need.
- **Life Expectancy, Infant Mortality, Maternal Mortality Ratio:** Population well-being.

#### 7. **Documentation**
- **Project Goals & Objectives**: [FinAID-goals.md](docs/FinAID-goals.md).
- **Preprocessing Details**: [proc-doc1.md](docs/proc-doc1.md) and [proc-doc2.md](docs/proc-doc2.md).

##### Additional Resources
- Test development documentation, output images, processed data in the [data/output folder](data/output/).
