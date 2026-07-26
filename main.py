import logging
import os

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from bot.handlers import start, help_command, handle_photo, handle_document

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_application():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "BOT_TOKEN environment variable is not set. "
            "Set it in your .env file locally, or in Railway's Variables tab."
        )

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))

    return application


def main():
    application = build_application()
    logger.info("Starting Watermark1EraseBot (polling mode)...")
    application.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
