from config import ConfigManager
config_manager = ConfigManager()
import os
config_manager.check_folder_presence('images')
config_manager.check_folder_presence('logs')
config_manager.check_folder_presence('processed-data')
config_manager.check_folder_presence('models')

"""Module for preprocessing world data, including percentage conversion, missing value handling, and normalization."""

from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, feature_range=(0, 100), missing_value_strategy='median'):
        """Initialize the DataPreprocessor class with customization options."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.feature_range = feature_range
        self.missing_value_strategy = missing_value_strategy
        self.datetime_str = datetime.now().strftime("%d%m%Y_%H%M%S")

    def visualize_missing_values(self, data: pd.DataFrame, stage: int):
        """Visualize missing values using a heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.isnull(), cbar=True, cmap='viridis')
        plt.title("Missing Values Heatmap - Stage " + str(stage))
        config_manager.save_plot('images', f'missing_values_stage_{stage}', self.datetime_str)
        plt.show()

    def convert_percentage_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert percentage values to numerical format."""
        logging.info("Converting percentage values to numerical format.")
        percentage_columns = ['Agricultural Land( %)', 'Tax revenue (%)', 'Unemployment rate']
        for col in percentage_columns:
            data[col] = data[col].str.rstrip('%').astype('float') / 100
        return data

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using the specified strategy."""
        logging.info("Handling missing values.")
        if self.missing_value_strategy == 'median':
            return data.fillna(data.median(numeric_only=True))
        elif self.missing_value_strategy == 'mean':
            return data.fillna(data.mean(numeric_only=True))
        else:
            raise ValueError(f"Invalid missing value strategy: {self.missing_value_strategy}")

    def visualize_feature_distributions(self, data: pd.DataFrame, title: str, stage: int):
        """Visualize the distribution of features."""
        g = sns.PairGrid(data)
        g.map_upper(sns.scatterplot, s=15)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.histplot, kde=True)
        plt.title(title)
        config_manager.save_plot('images', f'feature_distributions_stage_{stage}', self.datetime_str)
        plt.show()

    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the numerical features to the specified range. Excludes non-numeric data."""
        logging.info(f"Normalizing the features to the range {self.feature_range}.")

        # Separate non-numeric columns
        non_numeric_data = data[['Country']]
        numeric_data = data.drop(columns=['Country'])

        # Apply MinMaxScaler to numeric columns only
        scaler = MinMaxScaler(feature_range=self.feature_range)
        data_scaled = scaler.fit_transform(numeric_data)

        # Combine non-numeric and scaled numeric columns
        normalized_data = pd.DataFrame(data_scaled, columns=numeric_data.columns)
        normalized_data['Country'] = non_numeric_data['Country'].values

        return normalized_data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform all preprocessing steps."""
        logging.info("Starting preprocessing steps.")
        self.visualize_missing_values(data, stage=1)
        data = self.convert_percentage_values(data)
        data = self.handle_missing_values(data)
        self.visualize_feature_distributions(data, "Before Normalization", stage=2)
        data = self.normalize_features(data)
        self.visualize_feature_distributions(data, "After Normalization", stage=3)
        logging.info("Preprocessing completed successfully.")
        return data

if __name__ == '__main__':
    raw_data_path = 'data/world-data-2023.csv'
    preprocessor = DataPreprocessor()
    raw_data = pd.read_csv(raw_data_path)
    logging.info('Raw data loaded successfully.')
    preprocessed_data = preprocessor.preprocess_data(raw_data)
    preprocessed_data_path = f'data/output/preprocessed_world-data-2023_{datetime_str}.csv'
    preprocessed_data.to_csv(preprocessed_data_path, index=False)
    logging.info(f'Preprocessed data saved to {preprocessed_data_path}.')