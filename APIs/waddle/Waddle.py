import json
import os
from PIL import Image
from google import generativeai as genai
from gtts import gTTS
from Utilities import logging_utility
import markdown
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import base64

# Import the model id
from . import MODEL_ID, GENAI_API_KEY, EMAIL_SENDER, EMAIL_PASSWORD

_logger = logging_utility.setup_logger(__name__)

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

class VisionClient:

    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_ID)
        genai.configure(api_key=GENAI_API_KEY)

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
                _logger.error(f"Error generating TTS audio: {e}")
                return None
        else:
            _logger.warn("No summary text to convert to speech.")
            return None


    def analyze_location_and_image(self, location_text: str, image_path: str) -> tuple[str, str]:
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
        if image_data:
            response = self.model.generate_content([combined_prompt, image_data])
        else:
            response = self.model.generate_content(combined_prompt)

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

    def create_report(self, location_text: str, summary: str, sources: list) -> str:
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

    def send_email(self, report_text, safe_or_danger, email_receiver, audio_file=None, image_path=None, video_path=None):
        """Sends the report, safe/danger status, image, video, and audio via email."""

        # Create the email message
        msg = MIMEMultipart()
        msg['Subject'] = "Location Safety Report"
        msg['From'] = EMAIL_SENDER
        msg['To'] = email_receiver

        # Convert Markdown to HTML
        html = markdown.markdown(report_text)

        # Email body (HTML part)
        body = f"""
        <html>
          <head></head>
          <body>
            <h2>You are {'in' if safe_or_danger.lower() == 'danger' else ''} {safe_or_danger}</h2>
            <p><b>Report:</b></p>
            {html}
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
                _logger.error(f"Error: Image file not found at {image_path}")
            except Exception as e:
                _logger.error(f"Error attaching image: {e}")

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

        if video_path:
            with open(video_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {os.path.basename(video_path)}",
            )
            msg.attach(part)

        try:
            # Connect to the SMTP server
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()  # Start TLS encryption
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, email_receiver, msg.as_string())
            _logger.info(f"Email sent successfully to {email_receiver}!")

        except Exception as e:
            _logger.error(f"An error occurred while sending email: {e}")

    def create_json_output(self, safe_or_danger, audio_file=None):
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
                _logger.error(f"Error: Audio file not found at {audio_file}")
            except Exception as e:
                _logger.error(f"Error encoding audio to base64: {e}")

        with open("output.json", "w") as json_file:
            json.dump(json_data, json_file, indent=2)
        _logger.info("JSON output saved to output.json")
