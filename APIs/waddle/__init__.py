from Utilities.file_utility import get_credentials_file_path, read_json_file, get_project_root
from Utilities import logging_utility
import os
import googlemaps
from google import genai as google_genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from bs4 import BeautifulSoup
import base_keys

_logger = logging_utility.setup_logger(__name__)

# Model name—adjust to one you have access to (e.g., "gemini-1.5-pro-latest")
MODEL_ID = "gemini-2.0-flash-exp"


def _get_credential(filepath: str, key: str):
    _logger.info(f'Getting {key} from {filepath}')
    return read_json_file(filepath)[key]

try:
    GMAPS_KEY_LOC = get_credentials_file_path(base_keys.GOOGLE_MAPS_CREDENTIAL_FILE_KEY_NAME)
except:
    _logger.info('get_credentials_file_path failed for GMAPS_API_KEY. Using fallback')
    GMAPS_KEY_LOC = os.path.join(get_project_root(), "credential", 'google_maps_credential.json')

try:
    GENAI_KEY_LOC = get_credentials_file_path(base_keys.GEMINI_CREDENTIAL_FILE_KEY_NAME)
except:
    _logger.info('get_credentials_file_path failed for GENAI_API_KEY. Using fallback')
    GENAI_KEY_LOC = os.path.join(get_project_root(), "credential", 'gemini_credential.json')

try:
    EMAIL_KEY_LOC = get_credentials_file_path(base_keys.EMAIL_CREDENTIAL_FILE_KEY_NAME)
except:
    _logger.info('get_credentials_file_path failed for EMAIL_CREDENTIAL_FILE_KEY_NAME. Using fallback')
    EMAIL_KEY_LOC = os.path.join(get_project_root(), "credential", 'email_credential.json')

GMAPS_API_KEY = _get_credential(GMAPS_KEY_LOC, 'map_api_key')
GENAI_API_KEY = _get_credential(GENAI_KEY_LOC, 'gemini_api_key')
EMAIL_SENDER = _get_credential(EMAIL_KEY_LOC, 'email_sender')
EMAIL_PASSWORD = _get_credential(EMAIL_KEY_LOC, 'email_password')
EMAIL_RECEIVER = EMAIL_SENDER
client = google_genai.Client(api_key=GENAI_API_KEY)
gmaps = googlemaps.Client(key=GMAPS_API_KEY)

def get_news_nearby(latitude: float, longitude: float) -> tuple[str, list]:
    """
    Reverse-geocode the given latitude/longitude to find a textual area name.
    Then use a Google Search tool to find suspicious activity or relevant news
    near that area. Returns a textual summary from the LLM and a list of sources.
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
        location_text = "\n".join(text_parts)

        # Extract links and names using BeautifulSoup
        rendered_content = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
        soup = BeautifulSoup(rendered_content, 'html.parser')
        sources = []
        for a_tag in soup.find_all('a', class_='chip'):
            link = a_tag['href']
            name = a_tag.get_text(strip=True)
            sources.append({"link": link, "name": name})

        return location_text, sources
    else:
        return "No relevant news or activity found.", []
