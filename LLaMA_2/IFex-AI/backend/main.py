from fastapi import FastAPI, File, UploadFile
from image_describer import ImageDescriber
from llama2chat import LLaMA2Chat

app = FastAPI()
image_describer = ImageDescriber()
llama2chat = LLaMA2Chat("meta-llama/Llama-2-7b-chat-hf")

@app.post("/describe_image/")
# Create an asynchronous function to receive an image and a user query, then return a description using CLIP and LLaMA 2 Chat.
async def describe_image(image: UploadFile = File(...), user_query: str = ""): 
    try:
        # Extract image features using ImageDescriber
        image_features = image_describer.describe_image(image)

        # Discern user needs using LLaMA2Chat
        user_needs = llama2chat.discern_user_needs(user_query)

        # Construct dialog
        dialog = llama2chat.construct_dialog(image_features)

        # Generate count based on user's needs and image features
        count_result = llama2chat.generate_count(dialog)

        
        # Use LLaMA2Chat functionality through ImageDescriber
        llama_response = image_describer.use_llama2chat(user_query)
        # Construct the final response
        final_response = {
            "description": image_features,
            "user_needs": user_needs,
            "count_result": count_result
        }

        return final_response  # Return the final response
    except Exception as e:
        return {"error": str(e)}  # Return an error if an exception occurs