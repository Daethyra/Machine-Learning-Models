"This module provides a class to describe images using the CLIP model and LLaMA 2 Chat."
from utility import Utility
utility = Utility()
from fastapi import File, UploadFile
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import logging
from typing import Tuple, List

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Create a class to describe images using the CLIP model and LLaMA 2 Chat.
class ImageDescriber:
    """
    A class to describe images using the CLIP model and LLaMA 2 Chat.
    """

    def __init__(self):
        self.clip_model, self.clip_transform = utility.load_clip_model()
        self.llama_tokenizer, self.llama_model = utility.load_llama_model()

    # Create a static method to load the CLIP model.
    @staticmethod
    def load_clip_model() -> Tuple[torch.nn.Module, Compose]:
        # Load the model (pseudo-code)
        return clip_model, clip_transform

    # Create a static method to load the LLaMA 2 Chat model.
    @staticmethod
    def load_llama_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        # Load the model (pseudo-code)
        return tokenizer, model

    # Use a multi-turnkey prompt to call LLAMA 2 Chat and generate a description.
    def describe_image(self, image_file: UploadFile, user_query: str) -> dict:
        try:
            # Validate user_query
            is_valid = utility.validate_string(user_query, [lambda x: len(x) > 0])
            if not is_valid:
                return {"error": "Invalid user_query"}
            
            image = Image.open(image_file.file)
            image_tensor = self.clip_transform(image).unsqueeze(0)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor).tolist()
            
            # Modify prompt to include image_features
            feature_list = ', '.join(map(str, image_features))
            prompt = (
                f"<s>[INST] <<SYS>>"
                "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible."
                "<</SYS>>"
                f"Please take a look at the list of extracted features: {feature_list}."
                "Reply with an accurate count of the items I described earlier."
                "[/INST]</s>"
            )
            
            inputs = self.llama_tokenizer(prompt, return_tensors="pt")
            outputs = self.llama_model.generate(**inputs)
            description = self.llama_tokenizer.decode(outputs[0])
            
            return {"description": description}

        except Exception as e:
            detailed_error = utility.detailed_error_handling(e)
            return {"error": detailed_error}

    def use_llama2chat(self, user_query: str) -> str:
        llama2chat = LLaMA2Chat("meta-llama/Llama-2-7b-chat-hf")
        user_needs = llama2chat.discern_user_needs(user_query)
        return user_needs
