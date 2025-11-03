# Obsidivec Bot Guide

This guide covers setting up and using bot interfaces for querying your Obsidian vault via Telegram and WhatsApp.

## Table of Contents

- [Overview](#overview)
- [Telegram Bot Setup](#telegram-bot-setup)
- [WhatsApp Bot Setup](#whatsapp-bot-setup)
- [Bot Architecture](#bot-architecture)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

Obsidivec supports two messaging bot interfaces:

- **Telegram Bot**: Async polling-based bot for direct Telegram integration
- **WhatsApp Bot**: Webhook-based bot using Twilio's WhatsApp API

Both bots provide the same functionality:
1. Receive user messages
2. Query the vector search API
3. Generate AI-powered responses from your vault
4. Return intelligent summaries to the user

---

## Telegram Bot Setup

### Prerequisites

- A Telegram account
- Telegram Bot Token from [@BotFather](https://t.me/botfather)
- OpenRouter API key for AI responses

### Step 1: Create Your Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Follow the prompts to:
   - Choose a display name (e.g., "My Vault Assistant")
   - Choose a username (must end in `bot`, e.g., `my_vault_bot`)
4. Copy the bot token (format: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Configure Environment Variables

Edit your `.env` file:

```env
# Enable Telegram bot
ENABLE_TELEGRAM_BOT=True

# Your bot token from BotFather
TELEGRAM_API_KEY=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Required for AI responses
OPENROUTER_KEY=your_openrouter_key_here
```

### Step 3: Start the Bot

Using Docker Compose (recommended):
```bash
docker-compose up -d
```

Or run standalone:
```bash
python telegram_bot.py
```

You should see:
```
Bot is running. Press Ctrl+C to stop.
```

### Step 4: Test Your Bot

1. Open Telegram and search for your bot username
2. Click **Start** or send `/start`
3. Send any message, e.g., "What are my notes about Python?"
4. The bot will respond with AI-generated summaries from your vault

---

## WhatsApp Bot Setup

### Prerequisites

- A Twilio account with WhatsApp enabled
- Twilio Account SID and Auth Token
- WhatsApp-enabled phone number (Twilio sandbox or approved number)
- OpenRouter API key for AI responses
- Public webhook URL (use ngrok for local testing)

### Step 1: Set Up Twilio WhatsApp

1. **Create Twilio Account**:
   - Sign up at [https://www.twilio.com](https://www.twilio.com)
   - Complete phone verification

2. **Enable WhatsApp Sandbox** (for testing):
   - Go to [Twilio Console → Messaging → Try it out → Send a WhatsApp message](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn)
   - Follow instructions to connect your WhatsApp to the sandbox
   - Note the sandbox WhatsApp number (format: `whatsapp:+14155238886`)

3. **Get Production Access** (optional):
   - For production use, request WhatsApp Business API access
   - Complete Twilio's approval process
   - Get a dedicated WhatsApp Business number

### Step 2: Get Twilio Credentials

1. Go to [Twilio Console Dashboard](https://console.twilio.com)
2. Find **Account Info** section
3. Copy:
   - **Account SID** (starts with `AC...`)
   - **Auth Token** (click to reveal)

### Step 3: Configure Environment Variables

Edit your `.env` file:

```env
# Enable WhatsApp bot
ENABLE_WHATSAPP_BOT=True

# Twilio credentials
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Server port for webhook
WHATSAPP_BOT_PORT=5000

# Required for AI responses
OPENROUTER_KEY=your_openrouter_key_here
```

### Step 4: Expose Webhook URL

The WhatsApp bot requires a publicly accessible webhook. For local testing, use ngrok:

```bash
# Install ngrok (if not already installed)
# Download from https://ngrok.com

# Start ngrok tunnel
ngrok http 5000
```

Copy the HTTPS forwarding URL (e.g., `https://abc123.ngrok.io`)

### Step 5: Configure Twilio Webhook

1. Go to [Twilio Console → Messaging → Settings → WhatsApp Sandbox Settings](https://console.twilio.com/us1/develop/sms/settings/whatsapp-sandbox)
2. Under **"When a message comes in"**:
   - Paste your webhook URL: `https://abc123.ngrok.io/whatsapp`
   - Method: **POST**
3. Click **Save**

### Step 6: Start the WhatsApp Bot

```bash
# Install required dependencies
pip install flask twilio

# Run the bot
python whatsapp_bot.py
```

You should see:
```
WhatsApp bot webhook running on port 5000
Webhook URL: http://your-domain.com/whatsapp
Press Ctrl+C to stop.
```

### Step 7: Test Your WhatsApp Bot

1. Open WhatsApp on your phone
2. Send the sandbox join message (shown in Twilio Console)
   - Example: "join [sandbox-keyword]" to the Twilio sandbox number
3. Once connected, send any message:
   - "What are my notes about machine learning?"
4. The bot will respond with AI-generated summaries from your vault

---

## Bot Architecture

### Telegram Bot Architecture

```
┌─────────────────────────────────────────────┐
│         Telegram Servers                    │
└──────────────┬──────────────────────────────┘
               │ Polling
               ▼
┌─────────────────────────────────────────────┐
│  telegram_bot.py (Separate Process)         │
│  ├── Application.run_polling()              │
│  ├── MessageHandler for text messages       │
│  └── Async event loop                       │
└──────────────┬──────────────────────────────┘
               │ Calls
               ▼
┌─────────────────────────────────────────────┐
│  LLMsearch.query()                          │
│  ├── POST /search → Vector Search API       │
│  ├── OpenRouter LLM for summarization       │
│  └── Returns AI response                    │
└─────────────────────────────────────────────┘
```

**Process Details**:
- Runs in **separate process** to avoid asyncio conflicts
- Uses `multiprocessing.Process` for isolation
- Polls Telegram servers for new messages
- Handles errors independently with logging

### WhatsApp Bot Architecture

```
┌─────────────────────────────────────────────┐
│         WhatsApp User                       │
└──────────────┬──────────────────────────────┘
               │ Message
               ▼
┌─────────────────────────────────────────────┐
│         Twilio WhatsApp API                 │
└──────────────┬──────────────────────────────┘
               │ Webhook POST
               ▼
┌─────────────────────────────────────────────┐
│  whatsapp_bot.py (Flask Server)             │
│  ├── POST /whatsapp endpoint                │
│  ├── Extract message from request           │
│  └── Return TwiML response                  │
└──────────────┬──────────────────────────────┘
               │ Calls
               ▼
┌─────────────────────────────────────────────┐
│  LLMsearch.query()                          │
│  ├── POST /search → Vector Search API       │
│  ├── OpenRouter LLM for summarization       │
│  └── Returns AI response                    │
└─────────────────────────────────────────────┘
```

**Process Details**:
- Runs Flask web server listening for webhooks
- Synchronous request/response model
- Twilio handles message delivery
- Stateless webhook processing

---

## Configuration

### Environment Variables Reference

| Variable | Description | Required | Bot Type |
|----------|-------------|----------|----------|
| `ENABLE_TELEGRAM_BOT` | Enable/disable Telegram bot | No (default: `False`) | Telegram |
| `TELEGRAM_API_KEY` | Bot token from BotFather | Yes (if Telegram enabled) | Telegram |
| `ENABLE_WHATSAPP_BOT` | Enable/disable WhatsApp bot | No (default: `False`) | WhatsApp |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID | Yes (if WhatsApp enabled) | WhatsApp |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token | Yes (if WhatsApp enabled) | WhatsApp |
| `TWILIO_WHATSAPP_NUMBER` | Twilio WhatsApp number (with `whatsapp:` prefix) | Yes (if WhatsApp enabled) | WhatsApp |
| `WHATSAPP_BOT_PORT` | Flask server port for webhook | No (default: `5000`) | WhatsApp |
| `OPENROUTER_KEY` | OpenRouter API key for LLM | Yes (for AI responses) | Both |
| `API_KEY` | FastAPI authentication key | No | Both |

### Complete .env Example

```env
## OPENROUTER
OPENROUTER_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxx

## INTERNAL SETTINGS
API_KEY=your_fastapi_secret_key

## TELEGRAM BOT SETTINGS
ENABLE_TELEGRAM_BOT=True
TELEGRAM_API_KEY=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

## WHATSAPP BOT SETTINGS
ENABLE_WHATSAPP_BOT=True
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
WHATSAPP_BOT_PORT=5000
```

---

## Usage Examples

### Telegram Bot

Once your bot is running:

**User:** "What are my notes about Python?"

**Bot Response:**
```
Based on your vault, here's what I found about Python:

You have several notes covering Python fundamentals including:
- Data structures (lists, dictionaries, sets)
- Object-oriented programming concepts
- Common libraries like requests and pandas

Your most recent notes focus on async/await patterns
and using FastAPI for web development.
```

### WhatsApp Bot

Same functionality via WhatsApp:

**User:** (via WhatsApp) "Summarize my machine learning notes"

**Bot Response:** (via WhatsApp)
```
Here's a summary of your machine learning notes:

Key Topics:
• Supervised learning algorithms (linear regression,
  decision trees, neural networks)
• Model evaluation metrics (accuracy, precision, recall)
• Libraries: scikit-learn, TensorFlow, PyTorch

Recent Focus:
You've been exploring transformer architectures and
attention mechanisms for NLP tasks.
```

### Query Tips

**Good Queries**:
- "What did I write about project management?"
- "Summarize my notes on React hooks"
- "Find information about database optimization"
- "What are my thoughts on productivity?"

**Less Effective**:
- "Hello" (no search context)
- Very vague queries without specific topics
- Questions about information not in your vault

---

## Troubleshooting

### Telegram Bot Issues

#### Bot Not Starting

**Error:** `set_wakeup_fd only works in main thread`

**Solution:** Ensure you're using the latest version which runs the bot in a separate process via `multiprocessing.Process`.

**Check:**
```python
# In main.py or your startup script
from multiprocessing import Process
bot_process = Process(target=telegram_bot.main)
bot_process.start()
```

#### Bot Not Responding

**Possible Causes:**
1. **Invalid Token**: Verify `TELEGRAM_API_KEY` is correct
2. **Bot Not Started**: Check logs for startup confirmation
3. **Network Issues**: Ensure container/server can reach Telegram API
4. **API Issues**: Verify OpenRouter key and vector search API are working

**Debug Steps:**
```bash
# Check bot logs
docker-compose logs obsidian-search

# Test API directly
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "n_results": 3}'
```

#### Rate Limiting

**Error:** Telegram API rate limit exceeded

**Solution:**
- Implement message queuing
- Add delays between responses
- Use webhook mode instead of polling for high-volume bots

---

### WhatsApp Bot Issues

#### Webhook Not Receiving Messages

**Possible Causes:**
1. **ngrok tunnel expired**: Restart ngrok and update Twilio webhook URL
2. **Incorrect webhook URL**: Verify URL in Twilio console matches your server
3. **Firewall blocking**: Ensure port 5000 (or configured port) is accessible
4. **Flask not running**: Check if `whatsapp_bot.py` is running

**Debug Steps:**
```bash
# Check if Flask is running
curl http://localhost:5000/whatsapp

# Check ngrok status
curl http://127.0.0.1:4040/api/tunnels

# Test Twilio webhook manually
curl -X POST https://your-ngrok-url.ngrok.io/whatsapp \
  -d "Body=test message" \
  -d "From=whatsapp:+1234567890"
```

#### Authentication Errors

**Error:** Twilio authentication failed

**Solution:**
1. Verify `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are correct
2. Check credentials haven't expired
3. Ensure no extra spaces in `.env` file

#### Sandbox Connection Issues

**Error:** "You are not currently connected to this sandbox"

**Solution:**
1. Send the join code to the Twilio sandbox number from WhatsApp
2. Wait for confirmation message
3. Check sandbox participants in Twilio Console

#### Response Timeout

**Error:** WhatsApp messages timing out

**Solution:**
- Twilio webhooks have a 15-second timeout
- Optimize LLM response time
- Implement async processing for long queries:
  ```python
  # Send immediate acknowledgment
  resp.message("Processing your query...")

  # Process in background and send follow-up
  # (requires Twilio client for outbound messages)
  ```

---

### General Bot Issues

#### No Search Results

**Symptoms:** Bot responds but says no relevant information found

**Solutions:**
1. **Check vault indexing**: `curl http://localhost:8000/health`
   - Verify `collection_count > 0`
2. **Trigger reindex**: `curl -X POST http://localhost:8000/reindex`
3. **Verify file watcher**: Check logs for file processing
4. **Test search directly**:
   ```bash
   python LLMsearch.py "test query"
   ```

#### Poor Response Quality

**Issue:** Bot responses are not helpful or inaccurate

**Improvements:**
1. **Adjust chunk size**: Modify `split_token` in frontmatter
2. **Increase result count**: Edit `n_results` in `LLMsearch.py`
3. **Better embedding model**: Use `BAAI/bge-base-en-v1.5` instead of default
4. **Query refinement**: Use more specific queries
5. **LLM model**: Try different OpenRouter models

#### High API Costs

**Issue:** OpenRouter costs growing too high

**Solutions:**
1. **Cache responses**: Implement response caching for common queries
2. **Cheaper models**: Use faster, cheaper models for simple queries
3. **Limit result count**: Reduce `n_results` to send less context
4. **Rate limiting**: Implement per-user rate limits

---

## Production Deployment

### Telegram Bot Production

For production deployment:

```yaml
# docker-compose.yml
services:
  obsidian-search:
    environment:
      - ENABLE_TELEGRAM_BOT=True
      - TELEGRAM_API_KEY=${TELEGRAM_API_KEY}
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Best Practices:**
- Use environment variables, never hardcode tokens
- Enable logging for debugging
- Set up monitoring and alerts
- Implement rate limiting
- Use webhook mode for high-volume bots

### WhatsApp Bot Production

For production with a real domain:

```bash
# Use a reverse proxy (nginx) for SSL
server {
    listen 443 ssl;
    server_name bot.yourdomain.com;

    location /whatsapp {
        proxy_pass http://localhost:5000/whatsapp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Best Practices:**
- Use HTTPS (required by Twilio)
- Implement request validation (verify Twilio signatures)
- Set up monitoring and health checks
- Use production-grade WSGI server (gunicorn):
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 whatsapp_bot:app
  ```
- Implement message queuing for high volume
- Get WhatsApp Business API approval (not sandbox)

---

## Advanced Features

### Multi-Bot Support

You can run both bots simultaneously:

```env
ENABLE_TELEGRAM_BOT=True
ENABLE_WHATSAPP_BOT=True
```

Both bots will access the same vault and provide consistent responses.

### Custom Response Templates

Modify `LLMsearch.py` to customize bot responses:

```python
def query(user_query: str) -> str:
    results = search_vault(user_query)

    # Custom prompt for bot responses
    prompt = f"""You are a helpful assistant with access to my Obsidian vault.

    User Question: {user_query}

    Relevant Notes:
    {results}

    Provide a concise, helpful response based on the notes."""

    return llm_generate(prompt)
```

### Analytics and Logging

Track bot usage:

```python
# Add to bot files
import logging
from datetime import datetime

logging.basicConfig(
    filename='bot_analytics.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Log each query
logging.info(f"User: {user_id} | Query: {query} | Results: {result_count}")
```

---

## Support

For issues and questions:
- **GitHub Issues**: [Your repo URL]
- **Documentation**: See main [README.md](README.md)
- **Telegram**: [@BotFather](https://t.me/botfather) for bot creation
- **Twilio Support**: [Twilio Help Center](https://support.twilio.com)

---

**Built with ❤️ for the Obsidian community**