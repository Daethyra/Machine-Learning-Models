import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import inspect

class ConfigManager:
    def __init__(self, module_name: str):
        self.module_name = module_name
    
    def setup_configuration(self):
        self.configure_logger()
        self.check_folder_presence('images')
        self.check_folder_presence('logs')
        self.check_folder_presence('processed-data')
        self.check_folder_presence('models')

    def configure_logger(self):
        # Automatically detects the module name from the caller's frame
        caller_frame = inspect.stack()[1]
        module_name = inspect.getmodule(caller_frame[0]).__name__
        # Configuring the logger with a log file path containing the current timestamp and module name
        log_file_path = 'data/output/logs/' + self.module_name + '_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.log'
        
        # Create a file handler that writes log messages to a file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler that writes log messages to stdout
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Get the root logger and add both handlers
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

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

    def find_latest_preprocessed_file(self):
        "Search for the latest output of preprocessed data"
        folder_path = 'data/output/processed-data'
        files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.csv')]
        
        # Finding the latest file based on modification time
        latest_file = max(files, key=os.path.getmtime) if files else None
        
        return latest_file
    
    def general_exception_handler(func, *args, **kwargs):
        """
        A general exception handler that can be used to wrap any callable.

        :param func: The callable (function or method) to execute.
        :param args: Positional arguments to pass to the callable.
        :param kwargs: Keyword arguments to pass to the callable.
        :return: The result of the callable, or None if an exception occurred.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"An error occurred while executing {func.__name__}: {e}")
            return None