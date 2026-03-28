"""
Telegram 換裝/換臉 Bot — V2

指令：
  /go_c  — 換裝（Qwen Image Edit 兩階段）
  /go_f  — 換臉（ReActor 直接換臉）
  /go_k  — Klein VTON + 換臉（Klein 4B + Florence2 + ReActor，3 張圖）
  /cancel — 取消
  /status — 檢查 ComfyUI
"""
import os
import json
import time
import shutil
import asyncio
import logging
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters,
)

load_dotenv(Path(__file__).parent / ".env")

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
COMFYUI_API = os.environ.get("COMFYUI_API", "http://127.0.0.1:8000")
COMFYUI_INPUT_DIR = Path(os.environ.get(
    "COMFYUI_INPUT_DIR", r"K:\ComfyUI_610\input"
))
COMFYUI_OUTPUT_DIR = Path(os.environ.get(
    "COMFYUI_OUTPUT_DIR", r"K:\ComfyUI_610\output"
))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Per-user session state ──────────────────────────────────────────
sessions: dict[int, dict] = {}

def get_session(user_id: int) -> dict:
    if user_id not in sessions:
        sessions[user_id] = {
            "mode": None,      # "clothing" | "face" | "klein"
            "person": None,
            "clothing": None,
            "face": None,
            "step": "idle",    # idle/person/clothing/face
        }
    return sessions[user_id]

def reset_session(user_id: int):
    sessions.pop(user_id, None)


# ── ComfyUI API helpers ────────────────────────────────────────────
def comfy_post(endpoint: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{COMFYUI_API}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def comfy_get_history(prompt_id: str) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"{COMFYUI_API}/history/{prompt_id}", timeout=10
        ) as resp:
            data = json.loads(resp.read())
            return data.get(prompt_id)
    except Exception:
        return None


def poll_result(prompt_id: str, timeout: int = 300) -> dict | None:
    """Poll ComfyUI until job is complete. Returns the history entry."""
    start = time.time()
    while time.time() - start < timeout:
        result = comfy_get_history(prompt_id)
        if result:
            status = result.get("status", {})
            msgs = status.get("messages", [])
            for m in msgs:
                if m[0] in ("execution_success", "execution_error"):
                    return result
        time.sleep(3)
    return None


def invalidate_lora_cache():
    """Touch loras folder to invalidate ComfyUI's model list cache."""
    try:
        lora_dir = Path(r"K:\ComfyUI_610\models\loras")
        lora_dir.touch()
    except Exception:
        pass


def get_output_images(result: dict) -> list[dict]:
    """Extract output images from any SaveImage node in the result."""
    outputs = result.get("outputs", {})
    for nid, out in outputs.items():
        imgs = out.get("images", [])
        if imgs:
            return imgs
    return []


def get_error_message(result: dict) -> str:
    """Extract error message from a failed result."""
    msgs = result.get("status", {}).get("messages", [])
    for m in msgs:
        if m[0] == "execution_error" and len(m) > 1:
            return m[1].get("exception_message", "")[:300]
    return "未知錯誤"


# ── Workflow imports ──────────────────────────────────────────────
from workflows.ClothingChange_V1 import build_stage1_extract, build_stage2_transfer
from workflows.FaceOff_V1 import build_faceoff_workflow
from workflows.KleinVTON_V1 import build_klein_vton_workflow


# ── Telegram handlers ──────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    reset_session(update.effective_user.id)
    await update.message.reply_text(
        "👋 *換裝換臉 Bot*\n\n"
        "指令：\n"
        "👗 /go\\_c — 換裝（Qwen，人物＋服裝）\n"
        "😊 /go\\_f — 換臉（ReActor，人物＋臉孔）\n"
        "✨ /go\\_k — Klein VTON（換裝＋換臉，3 張圖）\n"
        "❌ /cancel — 取消\n"
        "📊 /status — 檢查 ComfyUI",
        parse_mode="Markdown",
    )


async def cmd_go_c(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """開始換裝流程"""
    sess = get_session(update.effective_user.id)
    sess["mode"] = "clothing"
    sess["person"] = None
    sess["clothing"] = None
    sess["face"] = None
    sess["step"] = "person"
    await update.message.reply_text(
        "👗 *換裝模式*\n\n"
        "📸 *步驟 1/2*\n請傳送【人物圖】（要換裝的那個人）",
        parse_mode="Markdown",
    )


async def cmd_go_f(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """開始換臉流程"""
    sess = get_session(update.effective_user.id)
    sess["mode"] = "face"
    sess["person"] = None
    sess["clothing"] = None
    sess["face"] = None
    sess["step"] = "person"
    await update.message.reply_text(
        "😊 *換臉模式*\n\n"
        "📸 *步驟 1/2*\n請傳送【人物圖】（要被換臉的那個人）",
        parse_mode="Markdown",
    )


async def cmd_go_k(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """開始 Klein VTON 換裝+換臉流程"""
    sess = get_session(update.effective_user.id)
    sess["mode"] = "klein"
    sess["person"] = None
    sess["clothing"] = None
    sess["face"] = None
    sess["step"] = "person"
    await update.message.reply_text(
        "✨ *Klein VTON 模式*（換裝 + 換臉）\n\n"
        "📸 *步驟 1/3*\n請傳送【人物圖】（要換裝的那個人）",
        parse_mode="Markdown",
    )


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    reset_session(update.effective_user.id)
    await update.message.reply_text("❌ 已取消")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        with urllib.request.urlopen(f"{COMFYUI_API}/system_stats", timeout=5) as resp:
            stats = json.loads(resp.read())
        gpu = stats.get("devices", [{}])[0]
        gpu_name = gpu.get("name", "Unknown")
        vram_total = gpu.get("vram_total", 0) / 1024**3
        vram_free = gpu.get("vram_free", 0) / 1024**3
        await update.message.reply_text(
            f"✅ ComfyUI 連線正常\n"
            f"GPU: {gpu_name}\n"
            f"VRAM: {vram_free:.1f} / {vram_total:.1f} GB 可用"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ ComfyUI 無法連線\n{e}")


# ── Image receiving ────────────────────────────────────────────────

async def save_image(file, user_id, step, ext=".jpg"):
    """Download and save image to ComfyUI input dir."""
    filename = f"tg_{user_id}_{step}_{int(time.time())}{ext}"
    local_path = COMFYUI_INPUT_DIR / filename
    await file.download_to_drive(str(local_path))
    log.info(f"Downloaded {step} image: {local_path}")
    return filename


async def route_after_image(message, sess, user_id, ctx):
    """Route to next step based on mode and current step."""
    mode = sess["mode"]
    step = sess["step"]

    if mode == "clothing":
        if step == "person":
            sess["step"] = "clothing"
            await message.reply_text(
                "✅ 人物圖已收到\n\n"
                "📸 *步驟 2/2*\n請傳送【服裝圖】（目標衣服）",
                parse_mode="Markdown",
            )
        elif step == "clothing":
            await start_processing(message, ctx, sess, user_id)

    elif mode == "face":
        if step == "person":
            sess["step"] = "face"
            await message.reply_text(
                "✅ 人物圖已收到\n\n"
                "📸 *步驟 2/2*\n請傳送【臉孔圖】（要換上去的臉）",
                parse_mode="Markdown",
            )
        elif step == "face":
            await start_processing(message, ctx, sess, user_id)

    elif mode == "klein":
        if step == "person":
            sess["step"] = "clothing"
            await message.reply_text(
                "✅ 人物圖已收到\n\n"
                "📸 *步驟 2/3*\n請傳送【服裝圖】（要穿的衣服）",
                parse_mode="Markdown",
            )
        elif step == "clothing":
            sess["step"] = "face"
            await message.reply_text(
                "✅ 服裝圖已收到\n\n"
                "📸 *步驟 3/3*\n請傳送【臉孔圖】（要換上去的臉）",
                parse_mode="Markdown",
            )
        elif step == "face":
            await start_processing(message, ctx, sess, user_id)


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    sess = get_session(user_id)

    if sess["step"] == "idle":
        await update.message.reply_text("請先輸入 /go_c 或 /go_f 開始")
        return

    photo = update.message.photo[-1]
    file = await ctx.bot.get_file(photo.file_id)
    step = sess["step"]
    filename = await save_image(file, user_id, step)
    sess[step] = filename

    await route_after_image(update.message, sess, user_id, ctx)


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    sess = get_session(user_id)

    if sess["step"] == "idle":
        await update.message.reply_text("請先輸入 /go_c 或 /go_f 開始")
        return

    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("請傳送圖片檔案")
        return

    file = await ctx.bot.get_file(doc.file_id)
    step = sess["step"]
    ext = Path(doc.file_name).suffix if doc.file_name else ".jpg"
    filename = await save_image(file, user_id, step, ext)
    sess[step] = filename

    await route_after_image(update.message, sess, user_id, ctx)


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    sess = get_session(user_id)
    if sess["step"] == "idle":
        await update.message.reply_text("輸入 /go_c 換裝 或 /go_f 換臉")
    else:
        await update.message.reply_text("目前在等圖片，請傳送圖片")


# ── Processing ─────────────────────────────────────────────────────

async def start_processing(source, ctx: ContextTypes.DEFAULT_TYPE,
                           sess: dict, user_id: int):
    # Determine chat_id
    if hasattr(source, "chat_id"):
        chat_id = source.chat_id
    elif hasattr(source, "message") and source.message:
        chat_id = source.message.chat_id
    elif hasattr(source, "effective_chat"):
        chat_id = source.effective_chat.id
    else:
        log.error(f"Cannot determine chat_id from source: {type(source)}")
        return

    mode = sess["mode"]
    sess["step"] = "idle"

    # Invalidate LoRA cache to ensure new models are visible
    invalidate_lora_cache()

    if mode == "clothing":
        await process_clothing(chat_id, ctx, sess, user_id)
    elif mode == "klein":
        await process_klein(chat_id, ctx, sess, user_id)
    else:
        await process_face(chat_id, ctx, sess, user_id)


async def process_clothing(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE,
                           sess: dict, user_id: int):
    """Two-stage clothing change: extract → transfer"""

    msg = await ctx.bot.send_message(
        chat_id=chat_id,
        text=(
            "⏳ 👗 換裝處理中...\n\n"
            f"👤 人物: {sess['person']}\n"
            f"👗 服裝: {sess['clothing']}\n\n"
            "📌 階段 1/2：提取服裝..."
        ),
    )

    # ── Stage 1: Extract outfit ──
    try:
        wf1 = build_stage1_extract(sess["clothing"])
        resp1 = comfy_post("/prompt", {"prompt": wf1})
    except Exception as e:
        err = e.read().decode()[:300] if hasattr(e, 'read') else str(e)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ Stage 1 提交失敗\n{err}"
        )
        reset_session(user_id)
        return

    pid1 = resp1.get("prompt_id")
    log.info(f"[clothing] Stage 1 submitted: {pid1}")

    result1 = await asyncio.to_thread(poll_result, pid1, timeout=180)
    if result1 is None:
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text="❌ Stage 1 逾時"
        )
        reset_session(user_id)
        return

    imgs1 = get_output_images(result1)
    if not imgs1:
        err = get_error_message(result1)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ Stage 1 失敗\n{err}"
        )
        reset_session(user_id)
        return

    # Copy extracted outfit from output to input
    extracted_filename = imgs1[0]["filename"]
    src_path = COMFYUI_OUTPUT_DIR / extracted_filename
    dst_path = COMFYUI_INPUT_DIR / extracted_filename
    try:
        shutil.copy2(str(src_path), str(dst_path))
    except Exception as e:
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ 無法複製提取結果\n{e}"
        )
        reset_session(user_id)
        return

    log.info(f"[clothing] Stage 1 done: {extracted_filename}")

    # ── Stage 2: Transfer outfit ──
    await ctx.bot.edit_message_text(
        chat_id=chat_id, message_id=msg.message_id,
        text=(
            "⏳ 👗 換裝處理中...\n\n"
            f"👤 人物: {sess['person']}\n"
            f"👗 服裝: {sess['clothing']}\n\n"
            "✅ 階段 1 完成：服裝已提取\n"
            "📌 階段 2/2：轉移服裝到人物..."
        ),
    )

    try:
        wf2 = build_stage2_transfer(sess["person"], extracted_filename)
        resp2 = comfy_post("/prompt", {"prompt": wf2})
    except Exception as e:
        err = e.read().decode()[:300] if hasattr(e, 'read') else str(e)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ Stage 2 提交失敗\n{err}"
        )
        reset_session(user_id)
        return

    pid2 = resp2.get("prompt_id")
    log.info(f"[clothing] Stage 2 submitted: {pid2}")

    result2 = await asyncio.to_thread(poll_result, pid2, timeout=180)
    if result2 is None:
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text="❌ Stage 2 逾時"
        )
        reset_session(user_id)
        return

    imgs2 = get_output_images(result2)
    if not imgs2:
        err = get_error_message(result2)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ Stage 2 失敗\n{err}"
        )
        reset_session(user_id)
        return

    # Send result
    await send_result_image(chat_id, ctx, msg, imgs2[0], "換裝", user_id)


async def process_face(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE,
                       sess: dict, user_id: int):
    """FaceOff: Flux Klein 4B + ReferenceLatent + ReActor face swap"""

    msg = await ctx.bot.send_message(
        chat_id=chat_id,
        text=(
            "⏳ 😊 換臉處理中...\n\n"
            f"👤 人物: {sess['person']}\n"
            f"😊 臉孔: {sess['face']}\n\n"
            "請稍候約 30~60 秒..."
        ),
    )

    try:
        wf = build_faceoff_workflow(sess["person"], sess["face"])
        resp = comfy_post("/prompt", {"prompt": wf})
    except Exception as e:
        err = e.read().decode()[:300] if hasattr(e, 'read') else str(e)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ 提交失敗\n{err}"
        )
        reset_session(user_id)
        return

    pid = resp.get("prompt_id")
    log.info(f"[face] Submitted: {pid}")

    result = await asyncio.to_thread(poll_result, pid, timeout=300)
    if result is None:
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text="❌ 處理逾時（超過 5 分鐘）"
        )
        reset_session(user_id)
        return

    imgs = get_output_images(result)
    if not imgs:
        err = get_error_message(result)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ 處理失敗\n{err}"
        )
        reset_session(user_id)
        return

    await send_result_image(chat_id, ctx, msg, imgs[0], "換臉", user_id)


async def process_klein(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE,
                        sess: dict, user_id: int):
    """Klein VTON + ReActor: 換裝（Florence2 自動描述）+ 換臉"""

    msg = await ctx.bot.send_message(
        chat_id=chat_id,
        text=(
            "⏳ ✨ Klein VTON 處理中...\n\n"
            f"👤 人物: {sess['person']}\n"
            f"👗 服裝: {sess['clothing']}\n"
            f"😊 臉孔: {sess['face']}\n\n"
            "Florence2 描述服裝中...請稍候約 60~120 秒"
        ),
    )

    try:
        wf = build_klein_vton_workflow(sess["person"], sess["clothing"], sess["face"])
        resp = comfy_post("/prompt", {"prompt": wf})
    except Exception as e:
        err = e.read().decode()[:300] if hasattr(e, 'read') else str(e)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ 提交失敗\n{err}"
        )
        reset_session(user_id)
        return

    pid = resp.get("prompt_id")
    log.info(f"[klein] Submitted: {pid}")

    result = await asyncio.to_thread(poll_result, pid, timeout=360)
    if result is None:
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text="❌ 處理逾時（超過 6 分鐘）"
        )
        reset_session(user_id)
        return

    imgs = get_output_images(result)
    if not imgs:
        err = get_error_message(result)
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"❌ 處理失敗\n{err}"
        )
        reset_session(user_id)
        return

    await send_result_image(chat_id, ctx, msg, imgs[0], "Klein VTON", user_id)


async def send_result_image(chat_id, ctx, status_msg, img_info, mode_label, user_id):
    """Download result image from ComfyUI and send to user."""
    img_filename = img_info["filename"]
    subfolder = img_info.get("subfolder", "")
    img_url = (
        f"{COMFYUI_API}/view?"
        f"filename={img_filename}&subfolder={subfolder}&type=output"
    )

    try:
        with urllib.request.urlopen(img_url, timeout=30) as resp:
            img_data = resp.read()
    except Exception as e:
        await ctx.bot.edit_message_text(
            chat_id=chat_id, message_id=status_msg.message_id,
            text=f"❌ 無法取得結果圖片\n{e}"
        )
        reset_session(user_id)
        return

    await ctx.bot.edit_message_text(
        chat_id=chat_id, message_id=status_msg.message_id,
        text="✅ 處理完成！正在傳送結果..."
    )
    await ctx.bot.send_photo(
        chat_id=chat_id,
        photo=img_data,
        caption=f"✅ {mode_label}完成！\n/go_c 換裝 | /go_f 換臉",
    )

    reset_session(user_id)
    log.info(f"Done [{mode_label}]! Sent {img_filename} to user {user_id}")


# ── Main ───────────────────────────────────────────────────────────
def reset_telegram_polling():
    """Force-reset Telegram polling before starting to prevent 409 Conflict."""
    import http.client
    host = "api.telegram.org"
    for endpoint in [
        f"/bot{TOKEN}/deleteWebhook?drop_pending_updates=true",
        f"/bot{TOKEN}/getUpdates?offset=-1&timeout=0",
    ]:
        try:
            conn = http.client.HTTPSConnection(host, timeout=10)
            conn.request("GET", endpoint)
            resp = conn.getresponse()
            resp.read()
            conn.close()
        except Exception as e:
            log.warning(f"Reset polling warning: {e}")
    log.info("Telegram polling reset done")


def main():
    # Force-reset polling to prevent 409 from zombie sessions
    reset_telegram_polling()
    time.sleep(2)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("go_c", cmd_go_c))
    app.add_handler(CommandHandler("go_f", cmd_go_f))
    app.add_handler(CommandHandler("go_k", cmd_go_k))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Bot started! Waiting for messages...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        poll_interval=1.0,
        timeout=10,
    )


if __name__ == "__main__":
    main()
