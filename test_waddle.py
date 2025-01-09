# Example usage in another file (e.g., test_api.py)
from APIs.waddle import Waddle, get_news_nearby, EMAIL_RECEIVER

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
    location_info, sources = get_news_nearby(latitude, longitude)

    # ---------------------------------------------------------------------
    # CALL #2: LOCATION + IMAGE -> Single Combined Analysis
    # ---------------------------------------------------------------------
    safe_or_danger_string, summary_text = vision_client.analyze_location_and_image(location_info, IMAGE_PATH)

    # ---------------------------------------------------------------------
    # CREATE REPORT
    # ---------------------------------------------------------------------
    report_text = vision_client.create_report(location_info, summary_text, sources)

    # ---------------------------------------------------------------------
    # TTS (Text-to-Speech)
    # ---------------------------------------------------------------------
    audio_file = vision_client.generate_tts_summary(summary_text)

    # ---------------------------------------------------------------------
    # SEND EMAIL
    # ---------------------------------------------------------------------
    vision_client.send_email(report_text, safe_or_danger_string, EMAIL_RECEIVER, audio_file, IMAGE_PATH)

    # ---------------------------------------------------------------------
    # CREATE JSON OUTPUT
    # ---------------------------------------------------------------------
    vision_client.create_json_output(safe_or_danger_string, audio_file)


if __name__ == "__main__":
    main()
