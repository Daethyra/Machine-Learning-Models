from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union, Dict

class LLaMA2Chat:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        Construct dialog by using image features.
        """
        feature_list = ', '.join(map(str, image_features))
        dialog = f"Please focus on {self.counting_methodology}. The image features are as follows: {feature_list}."
        return dialog

    def generate_count(self, dialog: str) -> Union[str, Dict]:
        """
        Generate a count based on the user's needs and the image features.
        """
        try:
            final_dialog = f"{self.counting_methodology} {dialog}"
            inputs = self.tokenizer(final_dialog, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            count_result = self.tokenizer.decode(outputs[0])
            return count_result
        except Exception as e:
            return {"error": str(e)}