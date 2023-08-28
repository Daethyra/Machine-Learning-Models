"""_summary_:This module provides a set of utility functions for the backend.
"""
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
class Utility:
    def __init__(self):
        self.cache = {}  # Cache storage

    @staticmethod
    def detailed_error_handling(exception: Exception) -> str:
        """
        Takes an exception object and returns a detailed error message.
        """
        return f"An error of type {type(exception).__name__} occurred. Details: {str(exception)}"

    def cache_function_output(self, function, *args, **kwargs):
        """
        Caches the output of a function given its arguments.
        """
        key = f"{function.__name__}{str(args)}{str(kwargs)}"
        if key in self.cache:
            return self.cache[key]
        else:
            result = function(*args, **kwargs)
            self.cache[key] = result
            return result

    @staticmethod
    def validate_string(input_str: str, conditions: list) -> bool:
        """
        Validates a string based on a list of conditions (functions).
        Each condition is a function that takes a string and returns a boolean.
        """
        return all(condition(input_str) for condition in conditions)
    
    @staticmethod
    def load_clip_model():
        clip_model = torch.hub.load('openai/CLIP', 'ViT-B/32')
        clip_transform = Compose([
            Resize((224, 224), interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return clip_model, clip_transform

    @staticmethod
    def load_llama_model():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        return tokenizer, model