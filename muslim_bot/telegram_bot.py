from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
from telegram import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from .config import Settings, load_settings
from .hijri import UNAVAILABLE_MESSAGE, HijriCalendar, calendar_from_settings, format_today_message, today_in
from .locations import find_city, nearest_city
from .prayer_times import (
    PRAYER_KEYS,
    PRAYER_NAMES,
    SavedLocation,
    client_from_settings,
    location_from_prefs,
    schedule_reply,
    tz_abbreviation,
)
from .progress_messages import classify_progress, pick_progress
from .reminders import OFF_WORDS, ON_WORDS, due_reminders, parse_prayer_name, prayer_list, reminder_text

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
📅 Tanggal Hijriyah hari ini — ketik /hijriyah
🕌 Jadwal sholat sesuai lokasi — atur dengan /lokasi, lihat dengan /jadwal
⏰ Pengingat sholat tepat waktu, termasuk Imsak dan Terbit — nyalakan dengan /pengingat on

Setiap rujukan saya tandai terverifikasi dari database. Saya bersifat edukatif — bukan mufti atau ustaz — dan untuk persoalan hukum agama yang spesifik, saya akan menyarankan Anda bertanya langsung ke ulama yang kompeten.

Silakan mulai bertanya! Contoh: hadis tentang niat atau ayat tentang sabar.

Atur sesuai keinginan Anda dengan /settings (bahasa, gaya jawaban, catatan, panjang riwayat). Hapus riwayat percakapan dengan /reset."""

SETTINGS_HELP = """Pengaturan Anda:
{current}

Ubah dengan:
/bahasa id|en — bahasa jawaban
/gaya ringkas|lengkap — gaya jawaban
/catatan <teks> — catatan gaya untuk bot (maks 200 karakter), /catatan hapus untuk menghapus
/riwayat <0-{max_turns}> — jumlah percakapan terakhir yang diingat (0 = tidak mengingat)
/lokasi — atur lokasi untuk jadwal sholat (/lokasi hapus untuk menghapus)
/pengingat — pengingat sholat tepat waktu (on/off)
/reset — hapus riwayat percakapan
/lupa — hapus riwayat dan semua pengaturan

Riwayat dan pengaturan dihapus otomatis jika tidak aktif beberapa hari."""

DEFAULT_LABEL = "bawaan"
CLEAR_WORDS = {"hapus", "reset", "default", "clear"}

NON_TEXT_REPLY = (
    "Saat ini saya hanya dapat memproses pesan teks. "
    "Silakan ketik pertanyaan Anda, misalnya: hadis tentang niat."
)

ERROR_REPLY = (
    "Maaf, terjadi gangguan. Silakan coba lagi sebentar. "
    "Jika berlanjut, coba mulai chat baru dengan /start atau /reset."
)

# Show the waiting message only if the answer takes longer than this, so instant
# replies (fast path, cache) do not flash a message that is deleted right away.
PROGRESS_DELAY_SECONDS = 1.5
PROGRESS_STATE_KEY = "progress_last"

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


async def backend_call(settings: Settings, method: str, path: str, chat_id: int, json: dict | None = None) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method,
            f"{settings.telegram_api_base.rstrip('/')}/sessions/{telegram_session_id(chat_id)}/{path}",
            json=json,
            timeout=BACKEND_TIMEOUT_SECONDS,
        )
    if response.status_code == 422:
        raise ValueError(response.json().get("detail", "Nilai tidak valid"))
    response.raise_for_status()
    return response.json()


def format_prefs(prefs: dict) -> str:
    def show(value) -> str:
        return DEFAULT_LABEL if value is None else str(value)

    return "\n".join(
        [
            f"• Bahasa: {show(prefs.get('language'))}",
            f"• Gaya: {show(prefs.get('style'))}",
            f"• Catatan: {show(prefs.get('note'))}",
            f"• Riwayat: {show(prefs.get('history_turns'))}",
            f"• Lokasi: {prefs.get('location_label') or 'belum diatur'}",
            f"• Pengingat: {format_reminder_summary(prefs)}",
        ]
    )


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
    settings: Settings = context.bot_data["settings"]
    try:
        await backend_call(settings, "DELETE", "history", update.message.chat_id)
    except Exception:  # noqa: BLE001 - the welcome message matters more than the reset.
        logger.warning("Could not reset history on /start")
    await send_reply(update.message, html.escape(WELCOME_MESSAGE))


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    settings: Settings = context.bot_data["settings"]
    try:
        body = await backend_call(settings, "DELETE", "history", message.chat_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to reset history")
        await message.reply_text(ERROR_REPLY)
        return
    count = body.get("deleted_turns", 0)
    await message.reply_text(f"Riwayat percakapan dihapus ({count} percakapan). Pengaturan Anda tetap tersimpan.")


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    settings: Settings = context.bot_data["settings"]
    try:
        await backend_call(settings, "DELETE", "history", message.chat_id)
        await backend_call(settings, "DELETE", "prefs", message.chat_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to forget session")
        await message.reply_text(ERROR_REPLY)
        return
    await message.reply_text("Riwayat dan semua pengaturan Anda sudah dihapus.")


async def hijri_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    calendar: HijriCalendar = context.bot_data["hijri"]
    today = today_in()
    hijri = await calendar.convert(today)
    if hijri is None:
        await message.reply_text(UNAVAILABLE_MESSAGE)
        return
    await send_reply(message, markdown_to_html(format_today_message(hijri, today)))


LOCATION_BUTTON = "📍 Kirim lokasi saya"
# A pin is labelled with a listed city only when it is clearly in that city.
PIN_LABEL_MAX_KM = 50.0

LOCATION_HELP = """Untuk jadwal sholat, saya perlu lokasi Anda.

📍 Ketuk tombol di bawah untuk mengirim lokasi Anda, atau ketik nama kota, misalnya:
/lokasi Bandung
/lokasi Depok, Jawa Barat

Saya hanya menyimpan koordinat perkiraan (sekitar 1 km), bukan alamat. Hapus kapan saja dengan /lokasi hapus atau /lupa."""

NO_LOCATION_REPLY = (
    "Lokasi Anda belum diatur. Ketuk tombol di bawah untuk mengirim lokasi, "
    "atau ketik /lokasi <nama kota>."
)


def location_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[KeyboardButton(LOCATION_BUTTON, request_location=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def saved_location(prefs: dict) -> SavedLocation | None:
    return location_from_prefs(
        SimpleNamespace(
            latitude=prefs.get("latitude"),
            longitude=prefs.get("longitude"),
            timezone=prefs.get("timezone"),
            location_label=prefs.get("location_label"),
        )
    )


async def save_location_and_show(
    message, context, *, latitude: float, longitude: float, timezone: str, label: str
) -> None:
    """Store the location for this chat, confirm it, and show today's schedule."""
    settings: Settings = context.bot_data["settings"]
    try:
        prefs = await backend_call(
            settings,
            "PUT",
            "prefs",
            message.chat_id,
            json={"latitude": latitude, "longitude": longitude, "timezone": timezone, "location_label": label},
        )
    except ValueError as exc:
        await message.reply_text(str(exc), reply_markup=ReplyKeyboardRemove())
        return
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save location")
        await message.reply_text(ERROR_REPLY, reply_markup=ReplyKeyboardRemove())
        return
    await message.reply_text(
        f"Lokasi diatur ke: {label} ({tz_abbreviation(timezone)}).\n"
        "Saya hanya menyimpan koordinat perkiraan, bukan alamat. Hapus kapan saja dengan /lokasi hapus.",
        reply_markup=ReplyKeyboardRemove(),
    )
    location = saved_location(prefs)
    if location is not None:
        text = await schedule_reply(context.bot_data["prayer"], location)
        await send_reply(message, markdown_to_html(text))


async def location_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    settings: Settings = context.bot_data["settings"]
    query = " ".join(context.args or []).strip()
    if not query:
        await message.reply_text(LOCATION_HELP, reply_markup=location_keyboard())
        return
    if query.lower() in CLEAR_WORDS:
        try:
            await backend_call(settings, "PUT", "prefs", message.chat_id, json={"latitude": None, "longitude": None})
        except Exception:  # noqa: BLE001
            logger.exception("Failed to clear location")
            await message.reply_text(ERROR_REPLY)
            return
        await message.reply_text(
            "Lokasi Anda sudah dihapus. Pengingat sholat, bila aktif, ikut dimatikan.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    match = find_city(query)
    city = match.found
    if city is None:
        if match.ambiguous:
            first = match.cities[0]
            options = "\n".join(f"• {candidate.label}" for candidate in match.cities)
            await message.reply_text(
                f"Ada beberapa tempat yang cocok:\n{options}\n\n"
                f"Ulangi dengan provinsinya, misalnya: /lokasi {first.name}, {first.province}"
            )
        else:
            await message.reply_text(
                "Saya tidak menemukan tempat itu. Coba nama kota atau kabupaten lain, "
                "atau ketuk tombol di bawah untuk mengirim lokasi Anda.",
                reply_markup=location_keyboard(),
            )
        return
    await save_location_and_show(
        message,
        context,
        latitude=city.latitude,
        longitude=city.longitude,
        timezone=city.timezone,
        label=city.label,
    )


async def location_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A location pin sent with the button (or any location the user shares)."""
    message = update.message
    if message is None or message.location is None:
        return
    latitude, longitude = message.location.latitude, message.location.longitude
    timezone = await context.bot_data["prayer"].resolve_timezone(latitude, longitude)
    if timezone is None:
        await message.reply_text(
            "Saya belum bisa menentukan zona waktu lokasi itu. Coba lagi sebentar, "
            "atau ketik nama kota dengan /lokasi.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    city = nearest_city(latitude, longitude, max_km=PIN_LABEL_MAX_KM)
    label = f"sekitar {city.label}" if city is not None else f"{latitude:.2f}, {longitude:.2f}"
    await save_location_and_show(
        message, context, latitude=latitude, longitude=longitude, timezone=timezone, label=label
    )


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    settings: Settings = context.bot_data["settings"]
    tomorrow = any(arg.lower() in {"besok", "tomorrow"} for arg in (context.args or []))
    try:
        prefs = await backend_call(settings, "GET", "prefs", message.chat_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load prefs for /jadwal")
        await message.reply_text(ERROR_REPLY)
        return
    location = saved_location(prefs)
    if location is None:
        await message.reply_text(NO_LOCATION_REPLY, reply_markup=location_keyboard())
        return
    text = await schedule_reply(context.bot_data["prayer"], location, tomorrow=tomorrow)
    await send_reply(message, markdown_to_html(text))


REMINDER_HELP = """Pengingat sholat dikirim tepat waktu (tanpa selisih menit) sesuai /jadwal.

/pengingat on — nyalakan untuk Imsak, Subuh, Terbit, Dzuhur, Ashar, Maghrib, dan Isya
/pengingat off — matikan semuanya
/pengingat imsak off — matikan satu waktu saja (juga: subuh, terbit, dzuhur, ashar, maghrib, isya); ganti off dengan on untuk menyalakannya lagi

Perlu lokasi: atur dengan /lokasi."""

REMINDERS_UNAVAILABLE_REPLY = "Pengingat belum tersedia di server ini. Silakan hubungi pengelola bot."
REMINDER_NEEDS_LOCATION_REPLY = (
    "Pengingat butuh lokasi untuk menghitung waktu sholat. Ketuk tombol di bawah atau ketik "
    "/lokasi <nama kota>, lalu ketik /pengingat on lagi."
)

REMINDER_INTERVAL_SECONDS = 60
# Telegram allows about 30 messages a second to different chats; stay well under it.
REMINDER_SEND_DELAY_SECONDS = 0.04
REMINDER_TASK_KEY = "reminder_task"


def format_reminder_summary(prefs: dict) -> str:
    if not prefs.get("reminders_enabled"):
        return "mati"
    prayers = prefs.get("reminder_prayers") or list(PRAYER_KEYS)
    return f"aktif ({len(prayers)} waktu)"


def format_reminder_status(prefs: dict) -> str:
    prayers = prefs.get("reminder_prayers") or list(PRAYER_KEYS)
    lines = [
        f"Pengingat sholat: {'aktif ✅' if prefs.get('reminders_enabled') else 'mati'}",
        f"Waktu: {prayer_list(prayers)}",
    ]
    if prefs.get("location_label"):
        lines.append(f"Lokasi: {prefs['location_label']}")
    return "\n".join(lines)


async def reminder_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    settings: Settings = context.bot_data["settings"]
    args = [arg.lower() for arg in (context.args or [])]
    try:
        prefs = await backend_call(settings, "GET", "prefs", message.chat_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load prefs for /pengingat")
        await message.reply_text(ERROR_REPLY)
        return
    if not args:
        await message.reply_text(f"{format_reminder_status(prefs)}\n\n{REMINDER_HELP}")
        return

    prayer: str | None = None
    switch_on = False
    chosen = prefs.get("reminder_prayers") or list(PRAYER_KEYS)
    if args[0] in ON_WORDS or args[0] in OFF_WORDS:
        switch_on = args[0] in ON_WORDS
        change: dict = {"enabled": switch_on}
    else:
        prayer = parse_prayer_name(args[0])
        word = args[1] if len(args) > 1 else ""
        if prayer is None or (word not in ON_WORDS and word not in OFF_WORDS):
            await message.reply_text(f"Cara pakai:\n\n{REMINDER_HELP}")
            return
        switch_on = word in ON_WORDS
        remaining = {key for key in chosen if key != prayer} | ({prayer} if switch_on else set())
        if not remaining:
            await message.reply_text(
                "Minimal satu waktu harus aktif. Pakai /pengingat off untuk mematikan semua pengingat."
            )
            return
        change = {"prayers": [key for key in PRAYER_KEYS if key in remaining]}

    if change.get("enabled"):
        if not settings.admin_token:
            logger.warning("/pengingat on refused: ADMIN_TOKEN is not set for the bot, so reminders cannot be delivered")
            await message.reply_text(REMINDERS_UNAVAILABLE_REPLY)
            return
        if saved_location(prefs) is None:
            await message.reply_text(REMINDER_NEEDS_LOCATION_REPLY, reply_markup=location_keyboard())
            return
    try:
        result = await backend_call(settings, "PUT", "reminders", message.chat_id, json=change)
    except ValueError as exc:
        await message.reply_text(str(exc))
        return
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save reminders")
        await message.reply_text(ERROR_REPLY)
        return

    active = prayer_list(result.get("prayers") or [])
    if prayer is None and switch_on:
        reply = f"Pengingat sholat dinyalakan ✅\nDikirim tepat waktu untuk: {active}.\nMatikan kapan saja dengan /pengingat off."
    elif prayer is None:
        reply = "Pengingat sholat dimatikan."
    else:
        reply = f"Pengingat {PRAYER_NAMES[prayer]} {'dinyalakan' if switch_on else 'dimatikan'}. Waktu aktif: {active}."
        if not result.get("enabled"):
            reply += "\nPengingat masih mati; nyalakan dengan /pengingat on."
    await message.reply_text(reply)


# --- delivering the reminders --------------------------------------------------------------------


def chat_id_of(session_id: str) -> int | None:
    prefix, _, value = session_id.partition(":")
    if prefix != "tg":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _admin_headers(settings: Settings) -> dict[str, str]:
    return {"x-admin-token": settings.admin_token or ""}


async def fetch_subscriptions(settings: Settings) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.telegram_api_base.rstrip('/')}/api/reminders",
            headers=_admin_headers(settings),
            timeout=20.0,
        )
    response.raise_for_status()
    return response.json()


async def admin_post(settings: Settings, path: str, json: dict | None = None) -> None:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.telegram_api_base.rstrip('/')}{path}",
            headers=_admin_headers(settings),
            json=json,
            timeout=20.0,
        )
    response.raise_for_status()


def _parse_moment(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _chat_is_gone(exc: TelegramError) -> bool:
    return isinstance(exc, Forbidden) or (isinstance(exc, BadRequest) and "chat not found" in str(exc).lower())


async def send_due_reminders(
    bot,
    settings: Settings,
    prayer,
    now: datetime,
    sent_cache: dict[str, datetime],
    *,
    delay: float = REMINDER_SEND_DELAY_SECONDS,
) -> int:
    """Send every reminder that is due at `now` (timezone-aware). Returns how many were sent.

    A reminder is recorded as sent right after Telegram accepts it: in `sent_cache` at once (so a failed
    bookkeeping call cannot cause a repeat) and on the server so a restart does not repeat it either.
    """
    subscriptions = await fetch_subscriptions(settings)
    for session_id in [sid for sid, moment in sent_cache.items() if now - moment > timedelta(days=1)]:
        del sent_cache[session_id]
    sent = 0
    for sub in subscriptions:
        session_id = sub["session_id"]
        chat_id = chat_id_of(session_id)
        if chat_id is None:
            continue
        local_now = now.astimezone(ZoneInfo(sub["timezone"]))
        timings = await prayer.timings(sub["latitude"], sub["longitude"], sub["timezone"], local_now.date())
        if timings is None:
            continue
        last_sent = max(
            (moment for moment in (_parse_moment(sub.get("last_sent")), sent_cache.get(session_id)) if moment),
            default=None,
        )
        for key, moment in due_reminders(timings, sub["prayers"], last_sent, local_now):
            try:
                await bot.send_message(chat_id=chat_id, text=reminder_text(key, timings, sub["location_label"]))
            except TelegramError as exc:
                if _chat_is_gone(exc):
                    logger.info("Chat %s is gone (%s); turning its reminders off", chat_id, exc)
                    await _quietly(admin_post(settings, f"/api/reminders/{session_id}/disable"))
                else:
                    logger.warning("Could not send a reminder to %s: %s", chat_id, exc)  # retried next minute
                break
            sent_cache[session_id] = moment
            await _quietly(
                admin_post(settings, f"/api/reminders/{session_id}/sent", {"sent_at": moment.astimezone(timezone.utc).isoformat()})
            )
            sent += 1
            await asyncio.sleep(delay)
    return sent


async def _quietly(awaitable) -> None:
    try:
        await awaitable
    except Exception:  # noqa: BLE001 - bookkeeping must never stop the reminders.
        logger.warning("Reminder bookkeeping call failed", exc_info=True)


async def reminder_loop(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    prayer = application.bot_data["prayer"]
    sent_cache: dict[str, datetime] = {}
    while True:
        try:
            await send_due_reminders(application.bot, settings, prayer, datetime.now(timezone.utc), sent_cache)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one bad minute must not end the loop.
            logger.exception("Reminder check failed; trying again next minute")
        # Wake just after the next minute starts, so "Subuh 04:12" goes out at 04:12.
        await asyncio.sleep(REMINDER_INTERVAL_SECONDS - time.time() % REMINDER_INTERVAL_SECONDS + 1)


async def start_reminders(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    if not settings.admin_token:
        logger.warning("ADMIN_TOKEN is not set for the bot: prayer reminders are disabled")
        return
    application.bot_data[REMINDER_TASK_KEY] = asyncio.create_task(reminder_loop(application))


async def stop_reminders(application: Application) -> None:
    task = application.bot_data.pop(REMINDER_TASK_KEY, None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    settings: Settings = context.bot_data["settings"]
    try:
        prefs = await backend_call(settings, "GET", "prefs", message.chat_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load prefs")
        await message.reply_text(ERROR_REPLY)
        return
    await send_reply(
        message,
        html.escape(SETTINGS_HELP.format(current=format_prefs(prefs), max_turns=prefs.get("max_history_turns", 12))),
    )


def parse_pref_args(field: str, args: list[str]) -> tuple[bool, object]:
    """Turn command arguments into a prefs value. Returns (ok, value); value None resets the field."""
    text = " ".join(args).strip()
    if not text:
        return False, None
    if text.lower() in CLEAR_WORDS:
        return True, None
    if field == "history_turns":
        if not text.isdigit():
            return False, None
        return True, int(text)
    if field == "note":
        return True, text
    return True, text.lower()


def make_pref_command(field: str, usage: str, label: str):
    async def command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None:
            return
        ok, value = parse_pref_args(field, list(context.args or []))
        if not ok:
            await message.reply_text(f"Cara pakai: {usage}")
            return
        settings: Settings = context.bot_data["settings"]
        try:
            prefs = await backend_call(settings, "PUT", "prefs", message.chat_id, json={field: value})
        except ValueError as exc:
            await message.reply_text(f"{exc}\nCara pakai: {usage}")
            return
        except Exception:  # noqa: BLE001
            logger.exception("Failed to save pref %s", field)
            await message.reply_text(ERROR_REPLY)
            return
        shown = prefs.get(field)
        await message.reply_text(f"{label} diatur ke: {DEFAULT_LABEL if shown is None else shown}")

    return command


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


_FILLER = r"(?:(?:tolong|mohon|please|bisa|ya|oke|ok)\s+)*"
_SUFFIX = r"(?:\s+(?:saja|aja|ya|dong|deh|terus|seterusnya|please))*\s*[.!]*$"
_STYLE_RE = re.compile(
    rf"^{_FILLER}(?:(?:jawab|balas)(?:an)?(?:nya)?\s+)?(?:(?:yang|dengan|secara)\s+)?"
    rf"(singkat|ringkas|pendek|lengkap|detail|terperinci|panjang){_SUFFIX}",
    re.IGNORECASE,
)
_LANGUAGE_RE = re.compile(
    rf"^{_FILLER}(?:(?:jawab|balas|bicara|berbicara|gunakan|pakai|menggunakan)\s+(?:(?:dalam|pakai|dengan|menggunakan)\s+)?"
    rf"|(?:reply|answer|respond|speak|use)\s+(?:in\s+)?)(?:bahasa\s+)?"
    rf"(inggris|english|indonesia|indonesian){_SUFFIX}",
    re.IGNORECASE,
)
_NAME_RE = re.compile(
    rf"^{_FILLER}panggil\s+(?:saya|aku|gue)\s+(.{{1,60}}?)(?:\s+(?:ya|saja|aja|dong))?\s*[.!]*$",
    re.IGNORECASE,
)
_REMEMBER_RE = re.compile(r"^ingat(?:lah)?(?:\s+bahwa\b|:)?\s+(.{3,200}?)\s*$", re.IGNORECASE | re.DOTALL)
_STYLE_WORDS = {
    "singkat": "ringkas",
    "ringkas": "ringkas",
    "pendek": "ringkas",
    "lengkap": "lengkap",
    "detail": "lengkap",
    "terperinci": "lengkap",
    "panjang": "lengkap",
}
_LANGUAGE_WORDS = {"inggris": "en", "english": "en", "indonesia": "id", "indonesian": "id"}


@dataclass(frozen=True)
class PrefIntent:
    field: str
    value: str
    label: str


def detect_pref_intent(text: str) -> PrefIntent | None:
    """Recognise a message that is *only* a preference instruction ("jawab singkat saja").

    Patterns are anchored to the whole message, so a real question that merely
    contains such words is never mistaken for a setting. The caller must still
    ask the user for confirmation before saving.
    """
    text = text.strip()
    if not text or text.startswith("/"):
        return None
    if match := _STYLE_RE.match(text):
        style = _STYLE_WORDS[match.group(1).lower()]
        return PrefIntent("style", style, f"Gaya jawaban: {style}")
    if match := _LANGUAGE_RE.match(text):
        language = _LANGUAGE_WORDS[match.group(1).lower()]
        return PrefIntent("language", language, f"Bahasa jawaban: {language}")
    if match := _NAME_RE.match(text):
        name = match.group(1).strip()
        return PrefIntent("note", f"Panggil pengguna dengan nama {name}", f"Catatan: panggil Anda {name}")
    if match := _REMEMBER_RE.match(text):
        note = " ".join(match.group(1).split())
        if len(note) <= 200:
            return PrefIntent("note", note, f"Catatan: {note}")
    return None


PENDING_KEY = "pending_pref"


def pending_store(context) -> dict | None:
    return getattr(context, "chat_data", None)


async def confirm_pref_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    store = pending_store(context)
    intent: PrefIntent | None = store.pop(PENDING_KEY, None) if store is not None else None
    if intent is None:
        await message.reply_text("Tidak ada pengaturan yang menunggu konfirmasi.")
        return
    settings: Settings = context.bot_data["settings"]
    try:
        await backend_call(settings, "PUT", "prefs", message.chat_id, json={intent.field: intent.value})
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save confirmed pref")
        await message.reply_text(ERROR_REPLY)
        return
    await message.reply_text(f"Tersimpan — {intent.label}. Lihat semua pengaturan dengan /settings.")


async def cancel_pref_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None:
        return
    store = pending_store(context)
    had_pending = store is not None and store.pop(PENDING_KEY, None) is not None
    await message.reply_text("Dibatalkan." if had_pending else "Tidak ada pengaturan yang menunggu konfirmasi.")


async def progress_after_delay(message, text: str, done: asyncio.Event, delay: float):
    """Send the waiting message unless `done` is set within `delay` seconds.

    Returns the sent message (to delete later) or None. Never cancelled: the
    caller sets `done` and awaits, so a send in flight cannot be orphaned.
    """
    try:
        await asyncio.wait_for(done.wait(), timeout=delay)
        return None
    except asyncio.TimeoutError:
        pass
    try:
        return await message.reply_text(text)
    except TelegramError:
        logger.warning("Progress message failed, continuing anyway")
        return None


def choose_progress_text(text: str, store: dict | None) -> str | None:
    """Pick a waiting message for `text`, avoiding a repeat of the last one in this chat."""
    category = classify_progress(text)
    last = store.get(PROGRESS_STATE_KEY, {}) if store is not None else {}
    chosen, index = pick_progress(category, last.get(category))
    if chosen is not None and store is not None:
        store[PROGRESS_STATE_KEY] = {**last, category: index}
    return chosen


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return
    store = pending_store(context)
    intent = detect_pref_intent(message.text)
    if intent is not None and store is not None:
        store[PENDING_KEY] = intent
        await message.reply_text(
            f"Simpan sebagai pengaturan tetap?\n{intent.label}\n\n"
            "Balas /ya untuk menyimpan atau /tidak untuk membatalkan."
        )
        return
    if store is not None:
        store.pop(PENDING_KEY, None)
    settings: Settings = context.bot_data["settings"]
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(typing_keepalive(message.chat, stop_typing))
    answer_ready = asyncio.Event()
    progress_text = choose_progress_text(message.text, store)
    progress_task = (
        asyncio.create_task(progress_after_delay(message, progress_text, answer_ready, PROGRESS_DELAY_SECONDS))
        if progress_text is not None
        else None
    )
    try:
        try:
            async with httpx.AsyncClient() as client:
                body = await post_chat(
                    client,
                    settings.telegram_api_base,
                    message.text,
                    message.chat_id,
                    message.from_user.id if message.from_user else message.chat_id,
                )
            answer_ready.set()
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
            answer_ready.set()
            progress = await progress_task if progress_task is not None else None
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
    application = builder.post_init(start_reminders).post_shutdown(stop_reminders).build()
    application.bot_data["settings"] = settings
    application.bot_data["hijri"] = calendar_from_settings(settings)
    application.bot_data["prayer"] = client_from_settings(settings)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("hijriyah", hijri_command))
    application.add_handler(CommandHandler("lokasi", location_command))
    application.add_handler(CommandHandler("jadwal", schedule_command))
    application.add_handler(CommandHandler("pengingat", reminder_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("lupa", forget_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("bahasa", make_pref_command("language", "/bahasa id|en", "Bahasa")))
    application.add_handler(CommandHandler("gaya", make_pref_command("style", "/gaya ringkas|lengkap", "Gaya jawaban")))
    application.add_handler(CommandHandler("catatan", make_pref_command("note", "/catatan <teks> atau /catatan hapus", "Catatan")))
    application.add_handler(CommandHandler("riwayat", make_pref_command("history_turns", "/riwayat <jumlah>", "Riwayat")))
    application.add_handler(CommandHandler("ya", confirm_pref_command))
    application.add_handler(CommandHandler("tidak", cancel_pref_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    # Before the catch-all below, which would answer a location pin with "text only".
    application.add_handler(MessageHandler(filters.LOCATION, location_message_handler))
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
