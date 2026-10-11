from __future__ import annotations

import asyncio
import html
import logging
import re

import httpx
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from .config import Settings, load_settings

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LENGTH = 4096
BACKEND_TIMEOUT_SECONDS = 180.0

WELCOME_MESSAGE = """Assalamu'alaikum! Saya Islamic Daris Bot 🤲

Saya dapat membantu Anda:
📖 Mencari ayat Al-Quran berdasarkan topik — lengkap dengan teks Arab dan terjemahan
📜 Mencari hadith dari 6 kitab induk (Bukhari, Muslim, Abu Dawud, Tirmidhi, Nasai, Ibnu Majah)
🔍 Menampilkan ayat/hadith spesifik — misalnya "ayat kursi" atau "Sahih al-Bukhari #1"
🎲 Ayat & hadith acak untuk tadabbur harian
💡 Menjelaskan konsep dasar Islam dalam Bahasa Indonesia dan Inggris

Setiap rujukan saya tandai terverifikasi dari database. Saya bersifat edukatif — bukan mufti atau ustaz — dan untuk persoalan hukum agama yang spesifik, saya akan menyarankan Anda bertanya langsung ke ulama yang kompeten.

Silakan mulai bertanya! Contoh: hadis tentang niat atau ayat tentang sabar."""

NON_TEXT_REPLY = (
    "Saat ini saya hanya dapat memproses pesan teks. "
    "Silakan ketik pertanyaan Anda, misalnya: hadis tentang niat."
)

ERROR_REPLY = (
    "Maaf, terjadi gangguan. Silakan coba lagi sebentar. "
    "Jika berlanjut, coba mulai chat baru dengan /start."
)

PROGRESS_REPLY = "🔍 Sebentar ya, saya carikan dalilnya…"

LOOP_REPLY = (
    "Pertanyaan ini membutuhkan banyak langkah dan saya kehabisan batas iterasi. "
    "Coba sederhanakan pertanyaan atau pecah menjadi beberapa pertanyaan kecil."
)

RATE_LIMITED_REPLY = (
    "Terlalu banyak permintaan dalam waktu singkat. "
    "Tunggu sebentar lalu coba lagi."
)

SERVICE_ERROR_REPLY = (
    "Layanan sedang bermasalah. Silakan coba lagi sebentar."
)


def error_reply_for_status(status_code: int | None) -> str:
    if status_code == 508:
        return LOOP_REPLY
    if status_code == 429:
        return RATE_LIMITED_REPLY
    if status_code in (502, 503, 504):
        return SERVICE_ERROR_REPLY
    return ERROR_REPLY


def telegram_session_id(chat_id: int) -> str:
    return f"tg:{chat_id}"


def forward_header_for(user_id: int) -> dict[str, str]:
    return {"x-forwarded-for": f"tg:{user_id}"}


def markdown_to_html(text: str) -> str:
    """Convert the LLM's light Markdown into Telegram HTML.

    Conservative on purpose: headers and bold become <b>, quotes become <i>.
    Lists are kept as plain lines with a bullet. Anything else passes through.
    """
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("### "):
            lines.append(f"<b>{html.escape(line[4:].strip())}</b>")
        elif line.startswith("## "):
            lines.append(f"<b>{html.escape(line[3:].strip())}</b>")
        elif line.startswith("# "):
            lines.append(f"<b>{html.escape(line[2:].strip())}</b>")
        elif line.startswith("> "):
            lines.append(f"<i>{html.escape(line[2:].strip())}</i>")
        elif re.match(r"^[-*]\s+", line):
            lines.append(f"• {inline_to_html(line[2:].strip())}")
        elif re.match(r"^\d+[.)]\s+", line):
            content = re.sub(r"^\d+[.)]\s+", "", line)
            lines.append(f"{line.split()[0]} {inline_to_html(content)}")
        else:
            lines.append(inline_to_html(raw_line))
    return "\n".join(lines).strip()


def inline_to_html(text: str) -> str:
    parts = re.split(r"(\*\*.+?\*\*|`[^`]+?`)", text)
    out: list[str] = []
    for part in parts:
        if len(part) >= 4 and part.startswith("**") and part.endswith("**"):
            out.append(f"<b>{html.escape(part[2:-2])}</b>")
        elif len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
        else:
            out.append(re.sub(r"\*([^*\n]+?)\*", r"<i>\1</i>", html.escape(part)))
    return "".join(out)


def format_references_html(references: list[dict]) -> str:
    lines: list[str] = []
    for ref in references:
        label = ref.get("label", "?")
        verified = ref.get("verified")
        mark = "✓" if verified is True else ("?" if verified is None else "⚠")
        lines.append(f"{mark} <i>{html.escape(str(label))}</i>")
    return "\n".join(lines)


def format_warnings_text(warnings: list[str]) -> str:
    return "\n".join(f"⚠ {warning}" for warning in warnings)


def build_reply(answer: str, references: list[dict], warnings: list[str]) -> str:
    chunks = [markdown_to_html(answer)]
    if references:
        chunks.append("<b>Rujukan:</b>\n" + format_references_html(references))
    if warnings:
        chunks.append(format_warnings_text(warnings))
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def split_message(text: str, limit: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        extra = len(paragraph) + (2 if current else 0)
        if current and current_len + extra > limit:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if len(paragraph) > limit:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for index in range(0, len(paragraph), limit):
                chunks.append(paragraph[index : index + limit])
        else:
            current.append(paragraph)
            current_len += extra
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


async def post_chat(
    client: httpx.AsyncClient,
    api_base: str,
    message: str,
    chat_id: int,
    user_id: int,
) -> dict:
    response = await client.post(
        f"{api_base.rstrip('/')}/chat",
        json={"message": message, "session_id": telegram_session_id(chat_id)},
        headers=forward_header_for(user_id),
        timeout=BACKEND_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


async def send_reply(message, text: str) -> None:
    chunks = split_message(text)
    for chunk in chunks:
        try:
            await message.reply_text(chunk, parse_mode=ParseMode.HTML)
        except BadRequest:
            logger.warning("HTML parse failed, falling back to plain text")
            await message.reply_text(re.sub(r"<[^>]+>", "", chunk))


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await send_reply(update.message, html.escape(WELCOME_MESSAGE))


async def typing_keepalive(chat, stop: asyncio.Event, interval: float = 4.0) -> None:
    """Resend the typing indicator until `stop` is set.

    Telegram shows each indicator for ~5 seconds, so long backend runs would
    otherwise leave the user staring at silence.
    """
    while not stop.is_set():
        try:
            await chat.send_action(ChatAction.TYPING)
        except TelegramError:
            logger.warning("Typing indicator failed, continuing anyway")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    settings: Settings = context.bot_data["settings"]
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(typing_keepalive(message.chat, stop_typing))
    try:
        try:
            progress = await message.reply_text(PROGRESS_REPLY)
        except TelegramError:
            logger.warning("Progress message failed, continuing anyway")
            progress = None
        try:
            async with httpx.AsyncClient() as client:
                body = await post_chat(
                    client,
                    settings.telegram_api_base,
                    message.text,
                    message.chat_id,
                    message.from_user.id if message.from_user else message.chat_id,
                )
            reply = build_reply(
                body.get("answer", ""),
                body.get("references", []),
                body.get("warnings", []),
            )
            await send_reply(message, reply or ERROR_REPLY)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.warning("Backend returned HTTP %s for Telegram message", status_code)
            try:
                await message.reply_text(error_reply_for_status(status_code))
            except TelegramError:
                logger.exception("Failed to send error reply")
        except Exception as exc:
            logger.exception("Failed to answer Telegram message: %s", exc)
            try:
                await message.reply_text(ERROR_REPLY)
            except TelegramError:
                logger.exception("Failed to send error reply")
        finally:
            if progress is not None:
                try:
                    await progress.delete()
                except TelegramError:
                    pass
    finally:
        stop_typing.set()
        await typing_task


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(NON_TEXT_REPLY)


def build_application(settings: Settings) -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
    request = HTTPXRequest(
        connect_timeout=settings.telegram_connect_timeout,
        read_timeout=settings.telegram_read_timeout,
        proxy=settings.telegram_proxy,
    )
    builder = Application.builder().token(settings.telegram_bot_token).request(request)
    if settings.telegram_proxy:
        builder = builder.proxy(settings.telegram_proxy).get_updates_proxy(settings.telegram_proxy)
    application = builder.build()
    application.bot_data["settings"] = settings
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    application.add_handler(
        MessageHandler(~filters.TEXT & ~filters.COMMAND, non_text_handler)
    )
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    build_application(settings).run_polling()


if __name__ == "__main__":
    main()
