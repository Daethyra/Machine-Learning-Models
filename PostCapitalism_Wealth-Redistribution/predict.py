import pandas as pd
import shap
from joblib import load
from plotly import express as px
from sklearn.tree import DecisionTreeRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

class FinancialAidAnalysis:
    def __init__(self, data_path: str, model_path: str, selected_features: list, report_path: str):
        """
        Initializes the FinancialAidAnalysis class with data, model, selected features, and report path.

        :param data_path: Path to the preprocessed data CSV file.
        :param model_path: Path to the trained model file.
        :param selected_features: List of selected features to use in the analysis.
        :param report_path: Path to save the generated report.
        """
        self.data, self.model = self.load_data_and_model(data_path, model_path)
        self.selected_features = selected_features
        self.report_path = report_path

    def load_data_and_model(self, data_path: str, model_path: str) -> (pd.DataFrame, object):
        """
        Loads data and model from given paths.

        :param data_path: Path to the preprocessed data CSV file.
        :param model_path: Path to the trained model file.
        :return: Loaded data as a DataFrame and trained model object.
        """
        try:
            data = pd.read_csv(data_path)
            # Additional data validation could include checks for missing values, data types, etc.
            model = load(model_path)
        except Exception as e:
            raise Exception(f"Error loading data or model: {str(e)}")
        return data, model

    def customized_ranking(self):
        """
        Ranks the data based on the model's predictions.
        """
        features = self.data[self.selected_features]
        predictions = self.model.predict(features)
        self.data['Financial_Aid_Rank'] = predictions.argsort().argsort()

    def feature_importance_analysis(self):
        """
        Analyzes feature importance using SHAP values.
        """
        tree_model = DecisionTreeRegressor()
        tree_model.fit(self.data[self.selected_features], self.data['Financial_Aid_Rank'])
        shap_values = shap.TreeExplainer(tree_model).shap_values(self.data[self.selected_features])
        self.data['Feature_Importance_Explanation'] = shap_values

    def interactive_visualization(self, plot_type="scatter_matrix"):
        """
        Creates an interactive visualization of the data.

        :param plot_type: Type of plot to create (default is scatter_matrix).
        """
        if plot_type == "scatter_matrix":
            fig = px.scatter_matrix(self.data, dimensions=self.selected_features + ['Financial_Aid_Rank'])
        # Additional plot types can be added here
        fig.show()

    def sensitivity_analysis(self, feature_to_analyze='Unemployment rate') -> list:
        """
        Performs sensitivity analysis on the given feature.

        :param feature_to_analyze: Feature to analyze for sensitivity.
        :return: List of sensitivity results.
        """
        sensitivity_results = []
        for change in range(-10, 11, 2):
            altered_data = self.data[self.selected_features].copy()
            altered_data[feature_to_analyze] += change
            altered_predictions = self.model.predict(altered_data)
            altered_rank = altered_predictions.argsort().argsort()
            sensitivity_results.append((change, altered_rank))
        return sensitivity_results

    def generate_report(self):
        """
        Generates a report in PDF format using the analysis results.
        """
        doc = SimpleDocTemplate(self.report_path)
        styles = getSampleStyleSheet()
        content = []
        content.append(Paragraph("Financial Aid Analysis Report", styles['Heading1']))
        # Additional content can be added, including tables, charts, and text
        doc.build(content)

    def run_analysis(self):
        """
        Runs the entire analysis, including ranking, feature importance analysis, visualization, sensitivity analysis, and report generation.
        """
        self.customized_ranking()
        self.feature_importance_analysis()
        self.interactive_visualization()
        sensitivity_results = self.sensitivity_analysis()
        self.generate_report()

if __name__ == "__main__":
    selected_features = ['Density (P/Km2)', 'Agricultural Land (%)', 'CPI', 'Fertility Rate', 'Unemployment rate', 'Urban_population']
    analysis = FinancialAidAnalysis(
        "path/to/preprocessed_data.csv",
        "data/output/models/FinancialAidGMM_model.pkl",
        selected_features,
        "path/to/report.pdf"
    )
    analysis.run_analysis()
