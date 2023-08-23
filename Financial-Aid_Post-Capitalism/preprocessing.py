from config import ConfigManager
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration setup
config_manager = ConfigManager()
config_manager.setup_configuration()

class DataPreprocessor:
    def __init__(self, feature_range=(0, 1), missing_value_strategy='median'):  # Changed feature range
        self.feature_range = feature_range
        self.missing_value_strategy = missing_value_strategy
        self.datetime_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def visualize_missing_values(self, data: pd.DataFrame, stage: int):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.isnull(), cbar=True, cmap='viridis')
        plt.title("Missing Values Heatmap - Stage " + str(stage))
        config_manager.save_plot('images', f'missing_values_stage_{stage}', self.datetime_str)

    def impute_missing_values_with_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=['number'])
        non_numeric_df = df.select_dtypes(exclude=['number'])
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric_data = imputer.fit_transform(numeric_df)
        imputed_numeric_df = pd.DataFrame(imputed_numeric_data, columns=numeric_df.columns, index=numeric_df.index)
        return pd.concat([imputed_numeric_df, non_numeric_df], axis=1)

    def convert_currency_and_percentage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == 'object':
                first_non_null_value = df[col].dropna().iloc[0]
                if '$' in str(first_non_null_value):
                    df[col] = df[col].replace('[\\\\$,]', '', regex=True).astype(float)
                elif '%' in str(first_non_null_value):
                    df[col] = df[col].replace('%', '', regex=True).astype(float) / 100
        return df

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Handling missing values.")
        if self.missing_value_strategy == 'median':
            return data.fillna(data.median(numeric_only=True))
        elif self.missing_value_strategy == 'mean':
            return data.fillna(data.mean(numeric_only=True))
        else:
            raise ValueError(f"Invalid missing value strategy: {self.missing_value_strategy}")

    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Normalizing the features to the range {self.feature_range}.")
        data = data.apply(lambda x: x.str.replace(',', '').astype(float) if x.dtype == 'object' else x)
        scaler = MinMaxScaler(feature_range=self.feature_range)
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting preprocessing steps.")
        self.visualize_missing_values(data, stage=1)
        data = self.convert_currency_and_percentage_columns(data)
        data = self.handle_missing_values(data)
        data = self.impute_missing_values_with_knn(data)
        data = self.normalize_features(data)
        logging.info("Preprocessing completed successfully.")
        return data

if __name__ == '__main__':
    raw_data_path = 'data/world-data-2023.csv'
    preprocessor = DataPreprocessor()
    raw_data = pd.read_csv(raw_data_path)
    logging.info('Raw data loaded successfully.')
    preprocessed_data = preprocessor.preprocess_data(raw_data)
    preprocessed_data_path = f'data/output/processed-data/preprocessed_world-data-2023_{preprocessor.datetime_str}.csv'
    preprocessed_data.to_csv(preprocessed_data_path, index=False)
    logging.info(f'Preprocessed data saved to {preprocessed_data_path}.')