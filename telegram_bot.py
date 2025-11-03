from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import os
from LLMsearch import query

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_API_KEY", None)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(query(update.message.text))


def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # Handle all text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print("Bot is running. Press Ctrl+C to stop.")
    # This will block and keep the bot running
    application.run_polling(allowed_updates=None)


if __name__ == "__main__":
    main()
