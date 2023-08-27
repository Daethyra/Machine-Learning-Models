from fastapi import File, UploadFile
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import logging
from typing import Tuple, List

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

class ImageDescriber:
    """
    A class to describe images using the CLIP model and LLaMA 2 Chat.
    """

    def __init__(self):
        self.clip_model, self.clip_transform = self.load_clip_model()
        self.llama_tokenizer, self.llama_model = self.load_llama_model()

    @staticmethod
    def load_clip_model() -> Tuple[torch.nn.Module, Compose]:
        clip_model = torch.hub.load('openai/CLIP', 'ViT-B/32')
        clip_transform = Compose([
            Resize((224, 224), interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return clip_model, clip_transform

    @staticmethod
    def load_llama_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        return tokenizer, model

    def describe_image(self, image_file: UploadFile) -> str:
        try:
            image = Image.open(image_file.file)
            image_tensor = self.clip_transform(image).unsqueeze(0)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)

            prompt = (
                "<s>[INST] <<SYS>>"
                "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible."
                "<</SYS>>"
                "What are you counting? I'm counting this. Anything you wanna tell me about those? They have these inside them. Okay great, give me a second to extract the image's features and I'll let you know what I see."
                "[/INST]</s>"
            )

            inputs = self.llama_tokenizer(prompt, return_tensors="pt")
            outputs = self.llama_model.generate(**inputs)
            description = self.llama_tokenizer.decode(outputs[0])

            return description

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return {"error": str(e)}
