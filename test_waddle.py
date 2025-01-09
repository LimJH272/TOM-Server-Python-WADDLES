# Example usage in another file (e.g., test_api.py)
from api import Waddle
from api import get_news_nearby

# Hardcoded lat/long for example
latitude, longitude = 1.3132469, 103.8834834

# A local image path for analysis
IMAGE_PATH = "geylang.jpg"


def main():
     # Initialize the Waddle API client
    vision_client = Waddle.VisionClient()

    # ---------------------------------------------------------------------
    # CALL #1: LOCATION ANALYSIS
    # ---------------------------------------------------------------------
    location_info = get_news_nearby(latitude, longitude)

    # ---------------------------------------------------------------------
    # CALL #2: LOCATION + IMAGE -> Single Combined Analysis
    # ---------------------------------------------------------------------
    words_array, summary_text = vision_client.analyze_location_and_image(location_info, IMAGE_PATH)

    # ---------------------------------------------------------------------
    # OUTPUT #1: JSON of Keywords
    # ---------------------------------------------------------------------
    import json
    print(json.dumps({"words": words_array}, indent=2))

    # ---------------------------------------------------------------------
    # OUTPUT #2: TTS of the Summary
    # ---------------------------------------------------------------------
    audio_file = vision_client.generate_tts_summary(summary_text)
    if audio_file:
        print(f"\nTTS audio saved as {audio_file}.")
    else:
        print("No audio file generated")


if __name__ == "__main__":
    main()