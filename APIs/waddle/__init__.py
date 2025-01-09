import base_keys
from Utilities.environment_utility import set_env_variable
from Utilities.file_utility import get_credentials_file_path, read_json_file
from Utilities import logging_utility
import os
import googlemaps
from google import genai as google_genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv

_logger = logging_utility.setup_logger(__name__)

# Model name—adjust to one you have access to (e.g., "gemini-1.5-pro-latest")
MODEL_ID = "gemini-2.0-flash-exp"

def _get_credential(filepath: str, key: str):
    _logger.info(f'Getting {key} from {filepath}')
    return read_json_file(filepath)[key]

GMAPS_API_KEY = _get_credential('./credential/google_maps_credential.json', 'map_api_key')
GENAI_API_KEY = _get_credential('./credential/gemini_credential.json', 'gemini_api_key')
client = google_genai.Client(api_key=GENAI_API_KEY)
gmaps = googlemaps.Client(key=GMAPS_API_KEY)

def get_news_nearby(latitude: float, longitude: float) -> str:
    """
    Reverse-geocode the given latitude/longitude to find a textual area name.
    Then use a Google Search tool to find suspicious activity or relevant news
    near that area. Returns a textual summary from the LLM.
    """
    try:
        results = gmaps.reverse_geocode((latitude, longitude))
        area = results[0]['formatted_address'] if results else "the specified location"
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        area = "the specified location"

    search_prompt = (
        f"Search for any recent suspicious activity, police reports, or relevant news stories/information "
        f"near {area} and its surrounding neighborhood. Provide the most credible information available prioritizing proximity."
    )

    # Create and configure the Google Search tool
    google_search_tool = Tool(google_search=GoogleSearch())

    # Make the LLM call (Call #1)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=search_prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )

    if response.candidates and response.candidates[0].content.parts:
        text_parts = [part.text for part in response.candidates[0].content.parts]
        return "\n".join(text_parts)
    else:
        return "No relevant news or activity found."