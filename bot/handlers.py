import io
import logging

import cv2
import numpy as np
from telegram import Update, InputFile
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from bot.watermark_remover import remove_watermark

logger = logging.getLogger(__name__)


WELCOME_TEXT = (
    "👋 Hi! I'm *Watermark1EraseBot*.\n\n"
    "Send me a photo and I'll try to automatically detect and remove "
    "watermarks from it using image inpainting.\n\n"
    "Commands:\n"
    "/start – show this message\n"
    "/help – how it works and tips\n\n"
    "⚠️ Auto-detection is heuristic-based — it works best on "
    "semi-transparent text/logo watermarks. Results can vary depending "
    "on the image."
)

HELP_TEXT = (
    "*How it works*\n"
    "1. You send a photo.\n"
    "2. I look for small, locally-contrasted overlay patterns typical "
    "of text or logo watermarks.\n"
    "3. I remove the detected regions and reconstruct the pixels "
    "underneath using inpainting.\n\n"
    "*Tips*\n"
    "- Works best on clear, semi-transparent watermarks (text or logos).\n"
    "- Very complex or fully opaque watermarks may not be fully removed.\n"
    "- Send the image as a *file/document* (uncompressed) instead of a "
    "normal photo if you want to avoid Telegram's compression reducing "
    "quality."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_TEXT, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def _process_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, file_bytes: bytes) -> None:
    chat = update.effective_chat
    await chat.send_action(ChatAction.UPLOAD_PHOTO)

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        await update.message.reply_text(
            "Sorry, I couldn't read that image. Please try a different file."
        )
        return

    try:
        result_bgr, mask = remove_watermark(img_bgr)
    except Exception:
        logger.exception("Watermark removal failed")
        await update.message.reply_text(
            "Something went wrong while processing that image. Please try again."
        )
        return

    success, buffer = cv2.imencode(".png", result_bgr)
    if not success:
        await update.message.reply_text("Failed to encode the result image.")
        return

    output = io.BytesIO(buffer.tobytes())
    output.name = "watermark_removed.png"

    if np.any(mask):
        caption = "✅ Done! Here's your image with detected watermark regions removed."
    else:
        caption = "ℹ️ I didn't detect a clear watermark pattern, so this is your original image."

    await update.message.reply_document(
        document=InputFile(output, filename="watermark_removed.png"),
        caption=caption,
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo = update.message.photo[-1]  # highest resolution
    tg_file = await photo.get_file()
    file_bytes = await tg_file.download_as_bytearray()
    await _process_and_reply(update, context, bytes(file_bytes))


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("Please send an image file.")
        return
    tg_file = await doc.get_file()
    file_bytes = await tg_file.download_as_bytearray()
    await _process_and_reply(update, context, bytes(file_bytes))
