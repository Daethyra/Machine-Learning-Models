import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

class ConfigManager:
    def __init__(self):
        pass

    def configure_logger(self):
        # Configuring the logger with a log file path containing the current timestamp
        log_file_path = 'data/output/logs/preprocessing_log_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.log'
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def check_folder_presence(self, folder_name: str):
        # Checking if the folder exists, and if not, creating it
        folder_path = 'data/output/' + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def save_plot(self, folder_name: str, file_name: str, datetime_str: str):
        # Checking and creating the folder if needed
        self.check_folder_presence(folder_name)
        # Saving the plot with the specified folder name, file name, and datetime string
        plt.savefig(f'data/output/{folder_name}/{file_name}_{datetime_str}.png')
