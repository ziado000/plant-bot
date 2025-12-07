import os
import requests
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from collections import OrderedDict
import gc
import zipfile
import threading

# Telegram imports
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Flask for health check
from flask import Flask

# Download models from Dropbox
def download_models_from_dropbox():
    models_dir = Path('models')
    if models_dir.exists() and len(list(models_dir.glob('*.keras'))) >= 18:
        print("✅ Models exist")
        return
    
    print("📥 Downloading models from Dropbox...")
    
    # Your Dropbox direct download link (change dl=0 to dl=1)
    DROPBOX_URL = "https://www.dropbox.com/scl/fi/1qhklwrp1qxe8cvsa0zf9/models.zip?rlkey=69s5wrz9kjg9dkb7yhkjz45xa&st=djrc963c&dl=1"
    
    try:
        print("Downloading... (this may take 5-10 minutes)")
        
        response = requests.get(DROPBOX_URL, stream=True)
        
        with open("models.zip", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ Download complete!")
        
        # Extract
        with zipfile.ZipFile("models.zip", 'r') as z:
            z.extractall('.')
        os.remove("models.zip")
        print("✅ Models ready!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Paths
BASE_DIR = Path('.')
MODELS_DIR = BASE_DIR / 'models'

# Download models on startup
download_models_from_dropbox()

# Check for local files
if Path('disease_translations.json').exists():
    TRANSLATIONS_FILE = Path('disease_translations.json')
else:
    TRANSLATIONS_FILE = BASE_DIR / 'disease_translations.json'

if Path('class_indices.json').exists():
    INDICES_FILE = Path('class_indices.json')
else:
    INDICES_FILE = BASE_DIR / 'class_indices.json'

# Global Cache with LRU
MAX_MODELS = 10
loaded_models = OrderedDict()
class_indices = {}
translations = {}

# User sessions: {user_id: selected_crop}
user_sessions = {}

# Load resources
def load_resources():
    global class_indices, translations
    
    if TRANSLATIONS_FILE.exists():
        with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        print("✅ Translations Loaded")
    
    if INDICES_FILE.exists():
        with open(INDICES_FILE, 'r') as f:
            class_indices = json.load(f)
        print(f"✅ Class Indices Loaded: {len(class_indices)} models")

# Model loading with LRU cache
def get_model(model_name):
    if model_name in loaded_models:
        loaded_models.move_to_end(model_name)
        print(f"♻️ Using cached model: {model_name}")
        return loaded_models[model_name]
    
    model_path = MODELS_DIR / f"{model_name}_final.keras"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    if len(loaded_models) >= MAX_MODELS:
        oldest = next(iter(loaded_models))
        print(f"🗑️ Removing old model: {oldest}")
        del loaded_models[oldest]
        gc.collect()
    
    print(f"📥 Loading model: {model_name}... ({len(loaded_models)+1}/{MAX_MODELS})")
    model = tf.keras.models.load_model(model_path)
    loaded_models[model_name] = model
    print(f"✅ Model loaded: {model_name}")
    
    return model

# Prediction
def predict_image(model, img_data, model_name):
    img = Image.open(BytesIO(img_data)).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx] * 100)
    
    indices = class_indices.get(model_name, {})
    english_label = indices.get(str(predicted_class_idx), f"Class_{predicted_class_idx}")
    
    crop_name = model_name.replace('_', ' ').title()
    arabic_label = english_label
    
    if crop_name in translations and english_label in translations[crop_name]:
        arabic_label = translations[crop_name][english_label]
        
    return arabic_label, confidence

# Telegram Bot Handlers

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user_id = update.effective_user.id
    user_sessions.pop(user_id, None)
    
    keyboard = [
        ['🌴 نخيل - أوراق', '🍊 حمضيات - ثمار'],
        ['🍃 حمضيات - أوراق', '🍅 طماطم - أوراق'],
        ['🥔 بطاطس - أوراق', '🥒 خيار - أوراق'],
        ['🌽 ذرة - أوراق', '🍇 عنب - أوراق'],
        ['🍎 رمان - ثمار', '🥬 خس - أوراق'],
        ['🌾 قمح - أوراق', '🥭 مانجو - أوراق'],
        ['🍌 موز - ثمار', '🍃 موز - أوراق'],
        ['🫘 فاصوليا - أوراق', '🍆 باذنجان - أوراق'],
        ['🥗 ملفوف - أوراق', '🌶️ فلفل - أوراق']
    ]
    
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    
    await update.message.reply_text(
        "🌿 *أهلاً بك في طبيب النباتات السعودي!* 🇸🇦\n\n"
        "اختر المحصول من القائمة:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle crop selection"""
    user_id = update.effective_user.id
    text = update.message.text
    
    # Crop mapping
    crop_map = {
        '🌴 نخيل - أوراق': 'date_palm_leaves',
        '🍊 حمضيات - ثمار': 'citrus_fruits',
        '🍃 حمضيات - أوراق': 'citrus_leaves',
        '🍅 طماطم - أوراق': 'tomato_leaves',
        '🥔 بطاطس - أوراق': 'potato_leaves',
        '🥒 خيار - أوراق': 'cucumber_leaves',
        '🌽 ذرة - أوراق': 'corn_leaves',
        '🍇 عنب - أوراق': 'grape_leaves',
        '🍎 رمان - ثمار': 'pomegranate_fruits',
        '🥬 خس - أوراق': 'lettuce_leaves',
        '🌾 قمح - أوراق': 'wheat_leaves',
        '🥭 مانجو - أوراق': 'mango_leaves',
        '🍌 موز - ثمار': 'banana_fruits',
        '🍃 موز - أوراق': 'banana_leaves',
        '🫘 فاصوليا - أوراق': 'bean_leaves',
        '🍆 باذنجان - أوراق': 'eggplant_leaves',
        '🥗 ملفوف - أوراق': 'cabbage_leaves',
        '🌶️ فلفل - أوراق': 'pepper_leaves'
    }
    
    if text in crop_map:
        selected_crop = crop_map[text]
        user_sessions[user_id] = selected_crop
        
        sample_type = 'الثمرة' if 'fruits' in selected_crop else 'الورقة'
        
        await update.message.reply_text(
            f"✅ تم اختيار: *{text}*\n\n"
            f"📸 *الآن، أرسل صورة {sample_type} المصابة.*",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            "⚠️ الرجاء اختيار محصول من القائمة أولاً، أو اضغط /start"
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads"""
    user_id = update.effective_user.id
    
    if user_id not in user_sessions:
        await update.message.reply_text(
            "⚠️ *الرجاء اختيار المحصول أولاً!*\nاضغط /start",
            parse_mode='Markdown'
        )
        return
    
    current_crop = user_sessions[user_id]
    
    try:
        # Get the photo
        photo = update.message.photo[-1]  # Highest resolution
        file = await context.bot.get_file(photo.file_id)
        
        # Download image data
        img_bytes = await file.download_as_bytearray()
        img_data = bytes(img_bytes)
        
        print(f"📥 Image received from user {user_id}: {len(img_data)} bytes")
        
        # Load model
        model = get_model(current_crop)
        if not model:
            await update.message.reply_text("❌ عذراً، هذا الموديل غير متوفر حالياً.")
            return
        
        # Predict
        print(f"🔬 Analyzing image...")
        diagnosis, conf = predict_image(model, img_data, current_crop)
        print(f"   ✅ Result: {diagnosis} ({conf:.1f}%)")
        
        # Escape HTML special characters in diagnosis
        diagnosis_escaped = diagnosis.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Build result with HTML formatting
        result_text = f"🔍 <b>التشخيص:</b> {diagnosis_escaped}\n🎯 <b>الدقة:</b> {conf:.1f}%\n\n"
        
        if conf < 60:
            result_text += "⚠️ <b>ملاحظة:</b> لست متأكداً تماماً. يرجى استشارة مهندس زراعي."
        else:
            result_text += "✅ <b>التشخيص موثوق.</b>"
        
        await update.message.reply_text(result_text, parse_mode='HTML')
        print(f"✅ Sent reply to user {user_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text("❌ حدث خطأ أثناء تحليل الصورة. حاول مرة أخرى.")

def main():
    """Start the bot"""
    # Load resources
    load_resources()
    
    # Get bot token from environment variable ONLY
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not TOKEN:
        raise ValueError("❌ TELEGRAM_BOT_TOKEN must be set in environment variables!")
    
    # Get port and webhook URL
    PORT = int(os.environ.get('PORT', 10000))
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL', 'https://plant-bot-yqxl.onrender.com')
    
    # Create Telegram application
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Flask app for webhook
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return 'Telegram Bot is running!', 200
    
    @app.route('/health')
    def health():
        return 'OK', 200
    
    @app.route(f'/{TOKEN}', methods=['POST'])
    def telegram_webhook():
        """Handle incoming updates via webhook"""
        try:
            update = Update.de_json(request.get_json(force=True), application.bot)
            # Run async code in sync context
            import asyncio
            asyncio.run(application.process_update(update))
            return 'OK'
        except Exception as e:
            print(f"❌ Webhook error: {e}")
            import traceback
            traceback.print_exc()
            return 'Error', 500
    
    # Set webhook
    async def set_webhook():
        webhook_url = f"{WEBHOOK_URL}/{TOKEN}"
        await application.bot.set_webhook(url=webhook_url)
        print(f"✅ Webhook set to: {webhook_url}")
    
    # Initialize bot
    import asyncio
    asyncio.run(set_webhook())
    
    print(f"✅ Telegram bot started with webhook!")
    print(f"✅ Server running on port {PORT}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=PORT, debug=False)

if __name__ == '__main__':
    main()
