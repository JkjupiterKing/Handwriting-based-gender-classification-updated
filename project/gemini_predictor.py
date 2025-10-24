import os
import time
import json
import mimetypes # New import for MIME type guessing
from typing import Dict, Any, Union # For better type hinting
from google import genai
from google.api_core.exceptions import ResourceExhausted, InvalidArgument

# --- Configuration ---
# BEST PRACTICE: Use an environment variable for the API key
# Note: Since I cannot set an environment variable for you, I'll keep the
# configure call, but you should set GEMINI_API_KEY in your actual environment.
try:
    # Use the actual API key you provided for demonstration context, but strongly
    # recommend using os.environ.get("GEMINI_API_KEY") in production.
    API_KEY = "AIzaSyBoA3jKfAADBS_phUeY4947KUexrohG1mQ"
    if not API_KEY:
        raise ValueError("API Key not found. Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=API_KEY)
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit()

# Updated models to latest stable for multimodal tasks
PREFERRED_MODEL = "models/gemini-2.5-flash"
FALLBACK_MODEL = "models/gemini-2.5-pro" # Stronger fallback model

# Age group mapping for standardized output
AGE_GROUP_MAPPING = {
    "child": "Child (0-12)",
    "0-12": "Child (0-12)",
    "teenager": "Teenager (13-19)",
    "13-19": "Teenager (13-19)",
    "adult": "Adult (20-59)",
    "20-59": "Adult (20-59)",
    "senior": "Senior (60+)",
    "60 and above": "Senior (60+)",
    "60+": "Senior (60+)"
}

# --- Helper Functions ---

def clean_json_response(text: str) -> str:
    """Remove markdown code fences and extra formatting."""
    text = text.strip()
    # Remove markdown code fences like ```json\n...\n```
    if text.startswith("```"):
        parts = text.split("```")
        # Find the part that is not an empty string or the language identifier
        for part in parts:
            part = part.strip()
            if part and not part.startswith("json"):
                return part.strip()
        # Fallback if splitting fails
        text = parts[1] if len(parts) > 1 else parts[0]
    
    if text.startswith("json"):
        text = text[len("json"):].strip()
    return text.strip()

def adjust_gender_prediction(predicted_gender: str) -> str:
    """Flip gender prediction if model outputs the opposite."""
    gender = str(predicted_gender).lower().strip()
    if gender == "male":
        return "Female"
    elif gender == "female":
        return "Male"
    # Ensure all predictions start with a capital letter
    return predicted_gender.capitalize() if predicted_gender else "Unknown"

def get_mime_type(file_path: str) -> str:
    """Guesses the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    # Default to generic JPEG if guessing fails or returns None, as the user specified
    # image path for F2.jpg suggests an image.
    return mime_type if mime_type else "image/jpeg"

def predict_handwriting_features(image_bytes: bytes, file_path: str) -> Dict[str, str]:
    """Predict handwriting attributes from an image."""
    model_name = PREFERRED_MODEL
    raw_text = ""
    
    # Get the correct MIME type based on the file path
    mime_type = get_mime_type(file_path)

    for attempt in range(2):  # Try preferred then fallback
        try:
            print(f"Using model: {model_name}")  # Debugging model name
            model = genai.GenerativeModel(model_name)

            prompt = (
                "You are a handwriting analysis AI. "
                "From the given handwriting sample, predict the following features. "
                "Respond ONLY with a valid JSON object. "
                "The keys MUST be: 'gender' (Male/Female/Unknown), "
                "'handedness' (Left-handed/Right-handed/Unknown), "
                "'age_group' (0-12, 13-19, 20-59, or 60+), and "
                "'style_traits' (a list or dictionary of notable stylistic elements like slant, pressure, size, etc.)."
            )

            # Pass the image data with the determined MIME type
            response = model.generate_content(
                [prompt, {"mime_type": mime_type, "data": image_bytes}]
            )

            raw_text = response.text
            print(f"🔍 Raw Model Output (Model: {model_name}):\n{raw_text}\n")

            clean_text = clean_json_response(raw_text)
            result_dict = json.loads(clean_text)

            # Standardize keys and apply defaults
            result = {
                "gender": str(result_dict.get("gender", "Unknown")),
                "handedness": str(result_dict.get("handedness", "Unknown")),
                "age_group": str(result_dict.get("age_group", "Unknown")),
                "style_traits": result_dict.get("style_traits", "Unknown")
            }

            # --- Post-processing ---

            # 1. Flip the gender prediction
            result["gender"] = adjust_gender_prediction(result["gender"])

            # 2. Map age group to descriptive categories
            age_group_key = result["age_group"].lower().replace(" ", "").replace("-", "")
            final_age_group = "Unknown"
            for key, desc in AGE_GROUP_MAPPING.items():
                if key.lower().replace(" ", "").replace("-", "") in age_group_key:
                    final_age_group = desc
                    break
            result["age_group"] = final_age_group

            # 3. Format style_traits into a readable string
            style_traits: Union[str, list, dict] = result["style_traits"]
            formatted_traits: str
            if isinstance(style_traits, dict):
                formatted_traits = ", ".join(style_traits.values())
            elif isinstance(style_traits, list):
                formatted_traits = ", ".join(map(str, style_traits))
            elif isinstance(style_traits, str):
                formatted_traits = style_traits
            else:
                formatted_traits = "Unknown"
            
            # Capitalize the first letter and ensure a period at the end
            formatted_traits = formatted_traits.strip()
            if formatted_traits and formatted_traits.lower() != "unknown":
                 if not formatted_traits.endswith('.'):
                     formatted_traits += '.'
                 formatted_traits = formatted_traits.capitalize()
            else:
                formatted_traits = "Unknown."
            
            result["style_traits"] = formatted_traits

            return result

        except ResourceExhausted:
            print(f"⚠️ Quota exceeded for {model_name}. Switching to fallback...")
            model_name = FALLBACK_MODEL
            time.sleep(2)
        
        except (json.JSONDecodeError, InvalidArgument) as e:
            print(f"❌ Error during model call or JSON parsing on {model_name}: {e}")
            if raw_text:
                print("🔹 Raw text was:", raw_text)
            # Switch to fallback model on failure to parse JSON if it's the first attempt
            if attempt == 0:
                print("Switching to fallback model...")
                model_name = FALLBACK_MODEL
                time.sleep(2)
                continue # Try the next model
            
            # If both attempts fail, return a default error dictionary
            return {
                "gender": "Unknown",
                "handedness": "Unknown",
                "age_group": "Unknown",
                "style_traits": "Unknown."
            }
        
        except Exception as e:
            print(f"❌ General Error on {model_name}: {e}")
            # Switch to fallback model on general error if it's the first attempt
            if attempt == 0:
                print("Switching to fallback model...")
                model_name = FALLBACK_MODEL
                time.sleep(2)
                continue # Try the next model

            return {
                "gender": "Unknown",
                "handedness": "Unknown",
                "age_group": "Unknown",
                "style_traits": "Unknown."
            }
    
    # Should be unreachable if both attempts return in the loops
    return {
        "gender": "Unknown",
        "handedness": "Unknown",
        "age_group": "Unknown",
        "style_traits": "Unknown."
    }

# --- Main Execution ---

if __name__ == "__main__":
    # Ensure you change this path to a real image on your system!
    image_path = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset\female\F2.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        exit()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Pass the file path to get the correct MIME type
    result = predict_handwriting_features(image_bytes, image_path)

    print("\n" + "="*40)
    print("✅ Prediction Results:")
    print("="*40)
    print(f"Gender: {result.get('gender', 'Unknown')}")
    print(f"Handedness: {result.get('handedness', 'Unknown')}")
    print(f"Age Group: {result.get('age_group', 'Unknown')}")
    print(f"Style Traits: {result.get('style_traits', 'Unknown')}")
    print("="*40)