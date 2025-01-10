import os
import json
from bs4 import BeautifulSoup
import googlemaps
import google.generativeai as genai
from google import genai as google_genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv
from PIL import Image
from gtts import gTTS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import markdown
import base64

# -------------------------------------------------------------------------
# 1) CONFIG
# -------------------------------------------------------------------------
load_dotenv()
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
EMAIL_SENDER = os.environ.get('EMAIL_SENDER')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
EMAIL_RECEIVER = EMAIL_SENDER  # Send to yourself

# Model name—adjust to one you have access to (e.g., "gemini-1.5-pro-latest")
MODEL_ID = "gemini-2.0-flash-exp"

# Hardcoded lat/long for example
latitude, longitude = 37.788132839912215, -122.40753565325528

# A local image path for analysis
IMAGE_PATH = "union_sq.jpg"

# SMTP server details (for Gmail)
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# -------------------------------------------------------------------------
# 2) SETUP CLIENTS
# -------------------------------------------------------------------------
genai.configure(api_key=GENAI_API_KEY)
client = google_genai.Client(api_key=GENAI_API_KEY)
gmaps = googlemaps.Client(key=GMAPS_API_KEY)

# -------------------------------------------------------------------------
# 3) GET LOCATION INFO (LLM + Google Search Tool) -- CALL #1
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 4) COMBINE IMAGE + LOCATION IN A SINGLE PROMPT -- CALL #2
# -------------------------------------------------------------------------
def analyze_location_and_image(location_text: str, image_path: str) -> list[str, str]:
    """
    Uses the LLM to analyze both the location info AND the image in a single prompt.
    Returns a string that says "Safe" or "Danger", and a short summary.
    """

    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except FileNotFoundError:
        image_comment = f"Image not found at {image_path}"
        image_data = None
    except Exception as e:
        image_comment = f"Error opening image: {e}"
        image_data = None
    else:
        image_comment = "Analyze the image to identify potential safety concerns or suspicious activity."
        image_data = img

    combined_prompt = (
        f"You  help travelers in a forign area descreetly avoid suspicious and unknown situations to them. You combine location-based news (the area the traveler is in right now) and an image (what the traveler is seeing right now) to identify potential risks and avoid dangerous situations by giving them context of the situation and advice how to discreetly proceed.\n\n"
        f"LOCATION DATA:\n{location_text}\n\n"
        f"IMAGE DATA: {image_comment}\n"
        "Please provide short 3 sentence summary of the relevent information,the 1st sentance should be what you see, the second sentance should be the context behind it, the 3rd sentance should be advice on how to discreetly avoid the situation. "
        "then condence the summary and output either 'Safe' or 'Danger' \n\n"
        "Return your answer in valid JSON with the keys: 'safe_or_danger' (string) "
        "and 'summary' (string). Example:\n"
        "{\n"
        "  \"safe_or_danger\": \"Safe or Danger\",\n"
        "  \"summary\": \"three sentances sentences here\"\n"
        "}\n"
    )

    # Make the LLM call (Call #2)
    model = genai.GenerativeModel(MODEL_ID)
    if image_data:
        response = model.generate_content([combined_prompt, image_data])
    else:
        response = model.generate_content(combined_prompt)

    # Extract text
    raw_output = "".join(part.text for part in response.candidates[0].content.parts)

    try:
        if "```json" in raw_output:
            json_str = raw_output.split("```json")[1].split("```")[0].strip()
        else:
            json_str = raw_output
        parsed = json.loads(json_str)
        safe_or_danger = parsed.get("safe_or_danger", "Error")
        summary = parsed.get("summary", "")
    except (json.JSONDecodeError, AttributeError, IndexError):
        safe_or_danger, summary = "Error", "No valid JSON returned or parsing error."

    return safe_or_danger, summary

# -------------------------------------------------------------------------
# 5) CREATE REPORT
# -------------------------------------------------------------------------
def create_report(location_text: str, summary: str, sources: list) -> str:
    """
    Combines the location text, summary, and sources into a single report string.
    """
    report = "**Location-Based Safety Report**\n\n"
    report += "**Location Information:**\n" + location_text + "\n\n"
    report += "**Analysis Summary:**\n" + summary + "\n\n"

    if sources:
        report += "**Sources:**\n"
        for source in sources:
            report += f"- {source['name']}: {source['link']}\n"
    else:
        report += "**Sources:**\nNo sources found.\n"

    return report

# -------------------------------------------------------------------------
# 6) SEND EMAIL FUNCTIONALITY
# -------------------------------------------------------------------------
def send_email(report_text, safe_or_danger, audio_file=None, image_path=None):
    """Sends the report, safe/danger status, image, and audio via email."""

    # Create the email message
    msg = MIMEMultipart()
    msg['Subject'] = "Location Safety Report"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    # Convert Markdown to HTML
    html = markdown.markdown(report_text)

    # Email body (HTML part)
    body = f"""
    <html>
      <head></head>
      <body>
        <p><b>Report:</b></p>
        {html}
        <p><b>Safe or Danger:</b> {safe_or_danger}</p>
      </body>
    </html>
    """
    msg.attach(MIMEText(body, 'html'))  # Attach as HTML

    # Attach image (if available)
    if image_path:
        try:
            with open(image_path, "rb") as img_file:
                img = MIMEImage(img_file.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                msg.attach(img)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
        except Exception as e:
            print(f"Error attaching image: {e}")

    # Attach audio file (if available)
    if audio_file:
        with open(audio_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(audio_file)}",
        )
        msg.attach(part)

    try:
        # Connect to the SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Start TLS encryption
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print(f"An error occurred while sending email: {e}")

# -------------------------------------------------------------------------
# 7) CREATE JSON FILE
# -------------------------------------------------------------------------
def create_json_output(safe_or_danger, audio_file=None):
    """Creates a JSON file with safe/danger status and base64 audio."""

    json_data = {
        "safe_or_danger": safe_or_danger
    }

    if audio_file:
        try:
            with open(audio_file, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
            json_data["audio_base64"] = encoded_audio
        except FileNotFoundError:
            print(f"Error: Audio file not found at {audio_file}")
        except Exception as e:
            print(f"Error encoding audio to base64: {e}")

    with open("output.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    print("JSON output saved to output.json")

def main():
    # ---------------------------------------------------------------------
    # CALL #1: LOCATION ANALYSIS
    # ---------------------------------------------------------------------
    location_info, sources = get_news_nearby(latitude, longitude)

    # ---------------------------------------------------------------------
    # CALL #2: LOCATION + IMAGE -> Single Combined Analysis
    # ---------------------------------------------------------------------
    safe_or_danger_string, summary_text = analyze_location_and_image(location_info, IMAGE_PATH)

    # ---------------------------------------------------------------------
    # CREATE REPORT
    # ---------------------------------------------------------------------
    report_text = create_report(location_info, summary_text, sources)

    # ---------------------------------------------------------------------
    # TTS (Text-to-Speech)
    # ---------------------------------------------------------------------
    audio_file = None  # Initialize to None
    if summary_text.strip():
        try:
            tts = gTTS(text=summary_text, lang='en')
            audio_file = "summary_audio.mp3"
            tts.save(audio_file)
            #print(f"\nTTS audio saved as {audio_file}.") #Commented out to not print anything
        except Exception as e:
            print("Error generating TTS audio:", e)

    # ---------------------------------------------------------------------
    # SEND EMAIL
    # ---------------------------------------------------------------------
    send_email(report_text, safe_or_danger_string, audio_file, IMAGE_PATH)

    # ---------------------------------------------------------------------
    # CREATE JSON OUTPUT
    # ---------------------------------------------------------------------
    create_json_output(safe_or_danger_string, audio_file)

    # ---------------------------------------------------------------------
    # NO CONSOLE OUTPUT (Except for error messages and email/JSON creation success)
    # ---------------------------------------------------------------------

if __name__ == "__main__":
    main()