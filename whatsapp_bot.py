from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv
import os
from LLMsearch import query

load_dotenv()

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", None)
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", None)
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", None)

app = Flask(__name__)


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Handle incoming WhatsApp messages."""
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "")

    # Get response from LLMsearch
    response_text = query(incoming_msg)

    # Create Twilio response
    resp = MessagingResponse()
    resp.message(response_text)

    return str(resp)


def main() -> None:
    """Start the WhatsApp bot server."""
    port = int(os.getenv("WHATSAPP_BOT_PORT", 5000))

    print(f"WhatsApp bot webhook running on port {port}")
    print(f"Webhook URL: http://your-domain.com/whatsapp")
    print("Press Ctrl+C to stop.")

    # Run Flask app
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
