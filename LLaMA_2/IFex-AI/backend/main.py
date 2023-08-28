"Execute this file to run the backend server."
from fastapi import FastAPI, File, UploadFile
from image_describer import FinalImageDescriber
from llama2chat import FinalLLaMA2Chat
from utility import Utility

app = FastAPI()
utility = Utility()
image_describer = FinalImageDescriber()
llama2chat = FinalLLaMA2Chat("meta-llama/Llama-2-7b-chat-hf")

@app.post("/describe_image/")
async def describe_image(image: UploadFile = File(...), user_query: str = ""):
    try:
        # Validate user_query
        is_valid = utility.validate_string(user_query, [lambda x: len(x) > 0])
        if not is_valid:
            return {"error": "Invalid user_query"}
        
        # Extract image features using ImageDescriber
        image_features = image_describer.describe_image(image, user_query)

        if "error" in image_features:
            return image_features

        # Discern user needs using LLaMA2Chat
        user_needs = llama2chat.discern_user_needs(user_query)

        # Construct dialog
        dialog = llama2chat.construct_dialog(image_features["description"])

        if "error" in dialog:
            return dialog

        # Generate count based on user's needs and image features
        count_result = llama2chat.generate_count(dialog)

        # Use LLaMA2Chat functionality through ImageDescriber
        llama_response = image_describer.use_llama2chat(user_query)

        # Construct the final response
        final_response = {
            "description": image_features["description"],
            "user_needs": user_needs,
            "count_result": count_result
        }

        return final_response

    except Exception as e:
        detailed_error = utility.detailed_error_handling(e)
        return {"error": detailed_error}