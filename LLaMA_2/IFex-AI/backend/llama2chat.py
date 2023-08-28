"This module contains the methods for chaining the conversation w/ LLAMA 2 Chat."
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union, Dict
from utility import Utility

utility = Utility()

class LLaMA2Chat:
    def __init__(self, model_name: str):
        self.tokenizer, self.model = utility.load_llama_model()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.counting_methodology = None  # To store user's counting methodology
        self.cache = {}  # To cache counting methodologies

    def discern_user_needs(self, user_query: str) -> str:
        """
        Use LLaMA 2 to discern user needs from the natural language query.
        """
        if user_query in self.cache:
            return self.cache[user_query]
        
        inputs = self.tokenizer(user_query, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        needs = self.tokenizer.decode(outputs[0])
        
        # Cache and store the entire output as the counting methodology
        self.cache[user_query] = needs
        self.counting_methodology = needs
        
        return needs

    def construct_dialog(self, image_features: List[float]) -> str:
        """
        Construct dialog by using image features, ensuring that image_features is a list of floats.
        """
        if not all(isinstance(feature, float) for feature in image_features):
            return {"error": "image_features must be a list of floats"}
        
        feature_list = ', '.join(map(str, image_features))
        dialog = f"Please focus on {self.counting_methodology}. The image features are as follows: {feature_list}."
        return dialog