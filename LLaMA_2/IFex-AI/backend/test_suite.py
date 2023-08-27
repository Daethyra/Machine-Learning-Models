import pytest
from fastapi.testclient import TestClient
from fastapi import File, UploadFile
from main import app
from image_describer import ImageDescriber
from llama2chat import LLaMA2Chat
from PIL import Image
import io

client = TestClient(app)
image_describer = ImageDescriber()
llama2chat = LLaMA2Chat("meta-llama/Llama-2-7b-chat-hf")

def test_describe_image_endpoint():
    # Create a test image
    img = Image.new('RGB', (60, 30), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Test the FastAPI endpoint
    response = client.post("/describe_image/", files={"image": ("test_image.png", img_byte_arr, "image/png")})
    assert response.status_code == 200
    assert "description" in response.json()

def test_image_feature_extraction():
    # Create a test image
    img = Image.new('RGB', (60, 30), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Test image feature extraction in ImageDescriber
    features = image_describer.extract_features(UploadFile("test_image.png", file=io.BytesIO(img_byte_arr)))
    assert features is not None

def test_llama2chat_methods():
    # Test methods in LLaMA2Chat
    user_query = "Tell me about the red objects in the image."
    user_needs = llama2chat.discern_user_needs(user_query)
    assert user_needs is not None

    dialog = llama2chat.construct_dialog(user_query, [0.1, 0.2, 0.3])
    assert dialog is not None

    description = llama2chat.generate_description(dialog)
    assert description is not None