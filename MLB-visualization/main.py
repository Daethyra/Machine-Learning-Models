from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
    
class DataAnalysis:
    def __init__(self, csv_paths: dict):
        self.csv_paths = csv_paths
        self.data_frames = {name: None for name in csv_paths}

    def load_data(self):
        for name, path in self.csv_paths.items():
            self.data_frames[name] = pd.read_csv(path)

    def plot_trend(self, metric, values, xlabel, ylabel, title):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(metric, values, marker='o', linestyle='-', color='b')
        ax.scatter(metric, values, color='r')
        ax.bar(metric, values, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return fig

    def generate_observations(self, input_text):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", use_auth_token='')
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", use_auth_token='')
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=1024,
            temperature=0.2,
            top_k=30,
            top_p=0.5,
        )
        observations = tokenizer.decode(output[0], skip_special_tokens=True)
        return observations

    def save_to_pdf(self, pdf_path, plots, observations, report_title, author):
        with PdfPages(pdf_path) as pdf:
            for plot in plots:
                pdf.savefig(plot)
                plt.close(plot)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.05, 0.95, "Observations:", fontsize=14, weight="bold")
            obs_text = observations.replace("\n", "\n    ")
            ax.text(0.05, 0.6, obs_text, fontsize=10)
            ax.axis("off")
            pdf.savefig(fig)

            pdf.infodict()['Title'] = report_title
            pdf.infodict()['Author'] = author
            pdf.infodict()['Subject'] = 'Data Analysis Report'
    def analysis_function(self, data_frames):
        # Extracting the required features
        career_stats_df = data_frames['career_stats']

        # Define the features and their weights
        features = ['G', 'IP', 'H', 'R', 'ER', 'HR', 'NP', 'SO', 'AVG']
        weights = [1, 1, 1, 1, 1, 2, 1, -1, 2]  # Higher weights for HR and AVG

        # Normalize and apply weights
        weighted_features = career_stats_df[features].apply(lambda x: x / x.max(), axis=0)
        weighted_features = weighted_features.multiply(weights, axis=1).sum(axis=1)

        # Plotting the trend for weighted features
        metric = career_stats_df['Season']
        plot = plt.figure(figsize=(10, 5))
        plt.plot(metric, weighted_features, marker='o', linestyle='-', color='b')
        plt.title('Weighted Performance Trend')
        plt.xlabel('Season')
        plt.ylabel('Weighted Performance')
        plt.grid(True)

        # Generate observations
        observations = "Shohei Ohtani's performance over seasons, considering the given features and weights."

        return [plot], observations
    
    def analyze(self, pdf_path, report_title, author):
        self.load_data()
        plots, observations = self.analysis_function(self.data_frames)
        observations += "\n" + self.generate_observations("Summary input for LLaMA")
        self.save_to_pdf(pdf_path, plots, observations, report_title, author)

if __name__ == "__main__":
    csv_paths = {
        'career_stats': 'data/group1/basic_career_stats.csv',
        'advanced_stats': 'data/group1/advanced_career_stats.csv',
        'other_stats': 'data/other_stats.csv'
    }
    analysis = DataAnalysis(csv_paths)
    analysis.analyze('analysis_report.pdf', 'Analysis Report', 'Author Name')