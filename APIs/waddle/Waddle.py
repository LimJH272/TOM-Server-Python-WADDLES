'''
This module is used to interact with Google Cloud Vision API.
'''
import json
from PIL import Image
from google import generativeai as genai
from gtts import gTTS
from Utilities import logging_utility

# Import the model id
from . import MODEL_ID, GENAI_API_KEY

_logger = logging_utility.setup_logger(__name__)

class VisionClient:
    '''
    This class is responsible for the Google Cloud Vision API.
    '''

    def __init__(self):
      """
      This method initializes the Google Cloud Vision API.
      """
      self.model = genai.GenerativeModel(MODEL_ID)
      genai.configure(api_key=GENAI_API_KEY)

    def analyze_location_and_image(self, location_text: str, image_path: str) -> tuple[list, str]:
        """
        Uses the LLM to analyze both the location info AND the image in a single prompt.
        Returns a list of keywords (5) and a short summary. Expects the LLM to return JSON.
        """

        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except FileNotFoundError:
            # If the image isn't found, just pass a short placeholder message
            image_comment = f"Image not found at {image_path}"
            image_data = None
        except Exception as e:
            image_comment = f"Error opening image: {e}"
            image_data = None
        else:
            image_comment = "Analyze the image to identify potential safety concerns or suspicious activity."
            image_data = img

        # Build a single prompt that includes both the location news + the image context.
        # We’ll ask the model to return valid JSON with "words" & "summary".
        combined_prompt = (
            f"You  help travelers in a forign area descreetly avoid suspicious and unknown situations to them. You combine location-based news (the area the traveler is in right now) and an image (what the traveler is seeing right now) to identify potential risks and avoid dangerous situations by giving them context of the situation and advice how to discreetly proceed.\n\n"
            f"LOCATION DATA:\n{location_text}\n\n"
            f"IMAGE DATA: {image_comment}\n"
            "Please provide short 3 sentence summary of the relevent information,the 1st sentance should be what you see, the second sentance should be the context behind it, the 3rd sentance should be advice on how to discreetly avoid the situation. "
            "that incorporates both the location info and what's seen in the image. then condence the summary to produce 5 key words to be displayed for the traveler to quicky view to understand\n\n"
            "Return your answer in valid JSON with the keys: 'words' (array of strings) "
            "and 'summary' (string). Example:\n"
            "{\n"
            "  \"words\": [\"keyword1\", \"keyword2\", ...],\n"
            "  \"summary\": \"one or two sentences here\"\n"
            "}\n"
        )
        print(location_text)
        # Make the LLM call (Call #2)
        if image_data:
            # We can pass [prompt, image] if the model supports image inputs
            response = self.model.generate_content([combined_prompt, image_data])
        else:
            # If the image is not available, just pass the text
            response = self.model.generate_content(combined_prompt)
        print(response)
        # Extract text
        raw_output = "".join(part.text for part in response.candidates[0].content.parts)

        # --- FIX: Extract JSON from within the response ---
        try:
            # Look for JSON within markdown code blocks:
            if "```json" in raw_output:
                json_str = raw_output.split("```json")[1].split("```")[0].strip()
            else:
                json_str = raw_output  # Assume direct JSON if no code block
            parsed = json.loads(json_str)
            words = parsed.get("words", [])
            summary = parsed.get("summary", "")
        except (json.JSONDecodeError, AttributeError, IndexError):
            words, summary = [], "No valid JSON returned or parsing error."
        # --- END FIX ---

        return words, summary

    def generate_tts_summary(self, summary_text: str) -> str:
        """
        Converts the summary text to speech and saves as mp3.
        Returns the audio file path or None if fails.
        """
        if summary_text.strip():
            try:
                tts = gTTS(text=summary_text, lang='en')
                audio_file = "summary_audio.mp3"
                tts.save(audio_file)
                return audio_file
            except Exception as e:
                print("Error generating TTS audio:", e)
                return None
        else:
            print("No summary text to convert to speech.")
            return None