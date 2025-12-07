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

# Telegram imports
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Statistics logging
from bot_statistics import log_prediction, initialize_stats_file

# Download models from Dropbox
def download_models_from_dropbox():
    models_dir = Path('models')
    
    print(f"🔍 Checking models directory...")
    print(f"   Directory exists: {models_dir.exists()}")
    
    if models_dir.exists():
        model_files = list(models_dir.glob('*.keras'))
        print(f"   Found {len(model_files)} .keras files")
        if len(model_files) >= 18:
            print("✅ All 18 models exist, skipping download")
            return
        else:
            print(f"⚠️ Only {len(model_files)}/18 models found, downloading...")
    else:
        print("   Models directory doesn't exist, creating and downloading...")
    
    print("📥 Downloading models from Hugging Face...")
    
    HUGGINGFACE_URL = "https://huggingface.co/ziadabdullah/saudi-plant-disease-models/resolve/main/models.zip"
    
    try:
        print("⏬ Starting download... (this may take 5-10 minutes)")
        
        response = requests.get(HUGGINGFACE_URL, stream=True)
        response.raise_for_status()  # Raise error for bad status codes
        
        print(f"   Download response status: {response.status_code}")
        
        with open("models.zip", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ Download complete!")
        print("📦 Extracting models.zip...")
        
        # Extract
        with zipfile.ZipFile("models.zip", 'r') as z:
            file_list = z.namelist()
            print(f"   Extracting {len(file_list)} files...")
            z.extractall('.')
        
        os.remove("models.zip")
        
        # Find models - check both root and nested directories
        model_files = list(Path('.').rglob('*.keras'))
        print(f"   Found {len(model_files)} .keras files")
        
        # Move models to correct location if they're nested
        if model_files and not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
        
        for model_file in model_files:
            if model_file.parent != models_dir:
                target = models_dir / model_file.name
                print(f"   Moving {model_file.name} to models/")
                model_file.rename(target)
        
        # Verify final count
        final_count = len(list(models_dir.glob('*.keras')))
        print(f"✅ Models ready! ({final_count} files in models/ directory)")
        
        if final_count < 18:
            print(f"⚠️ Warning: Expected 18 models, only found {final_count}")
        
    except Exception as e:
        print(f"❌ Download Error: {e}")
        import traceback
        traceback.print_exc()

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

# User sessions
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
    
    print(f"   🔍 Translation lookup: model={model_name}, label={english_label}")
    
    # Get bilingual label (Arabic + English)
    bilingual_label = english_label  # Default to English only
    
    # Try multiple matching strategies
    found = False
    
    # Strategy 1: Exact match
    if model_name in translations and english_label in translations[model_name]:
        bilingual_label = translations[model_name][english_label]
        found = True
        print(f"   ✅ Found exact match: {bilingual_label}")
    
    # Strategy 2: Lowercase with underscores
    if not found:
        english_lower = english_label.lower().replace(' ', '_')
        if model_name in translations and english_lower in translations[model_name]:
            bilingual_label = translations[model_name][english_lower]
            found = True
            print(f"   ✅ Found lowercase match: {bilingual_label}")
    
    # Strategy 3: Try just lowercase
    if not found:
        english_simple_lower = english_label.lower()
        if model_name in translations and english_simple_lower in translations[model_name]:
            bilingual_label = translations[model_name][english_simple_lower]
            found = True
            print(f"   ✅ Found simple lowercase: {bilingual_label}")
    
    if not found:
        print(f"   ⚠️ No translation found, using English only")
        # Show what keys are available for debugging
        if model_name in translations:
            available_keys = list(translations[model_name].keys())[:3]
            print(f"   Available keys: {available_keys}...")
    
    # If translation doesn't include English in parentheses, add it
    if english_label not in bilingual_label and '(' not in bilingual_label:
        bilingual_label = f"{bilingual_label} ({english_label})"
        
    return bilingual_label, confidence

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
        "🌿 أهلاً بك في طبيب النباتات السعودي! 🇸🇦\n\n"
        "اختر المحصول من القائمة:",
        reply_markup=reply_markup
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
            f"✅ تم اختيار: {text}\n\n"
            f"📸 الآن، أرسل صورة {sample_type} المصابة."
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
            "⚠️ الرجاء اختيار المحصول أولاً!\nاضغط /start"
        )
        return
    
    current_crop = user_sessions[user_id]
    
    try:
        # Get the photo
        photo = update.message.photo[-1]
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
        
        # Log statistics
        log_prediction(
            user_id=user_id,
            crop_type=current_crop,
            disease=diagnosis,
            confidence=conf,
            platform="Telegram"
        )
        
        # Escape HTML special characters
        diagnosis_escaped = diagnosis.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Build result with universal disclaimer
        result_text = (
            f"🔍 <b>التشخيص:</b> {diagnosis_escaped}\n"
            f"🎯 <b>الدقة:</b> {conf:.1f}%\n\n"
            f"⚠️ <b>ملاحظة هامة:</b> هذا تشخيص أولي لمساعدتك. "
            f"يُنصح باستشارة مهندس زراعي للحصول على تشخيص دقيق وخطة علاج مناسبة.\n\n"
            f"<i>This is a preliminary diagnosis. Please consult an agricultural engineer for accurate diagnosis and treatment plan.</i>"
        )
        
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
    
    # Initialize statistics logging
    initialize_stats_file()
    
    # Get bot token from environment
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not TOKEN:
        raise ValueError("❌ TELEGRAM_BOT_TOKEN environment variable is not set!")
    
    # Validate token length (Telegram tokens are typically 45-46 characters)
    if len(TOKEN) < 45:
        raise ValueError(
            f"❌ TELEGRAM_BOT_TOKEN appears to be truncated!\n"
            f"   Current length: {len(TOKEN)} characters\n"
            f"   Expected: 45+ characters\n"
            f"   Token starts with: {TOKEN[:20]}...\n"
            f"   Please check your environment variable on Render!"
        )
    
    print(f"✅ Token validated (length: {len(TOKEN)} characters)")
    
    # Create application with increased timeouts for slower networks
    from telegram.request import HTTPXRequest
    
    # Configure request with longer timeouts (Render can be slow)
    request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=30.0,    # 30 seconds to connect
        read_timeout=60.0,       # 60 seconds to read data
        write_timeout=60.0,      # 60 seconds to write data
        pool_timeout=30.0        # 30 seconds to get connection from pool
    )
    
    application = Application.builder().token(TOKEN).request(request).build()

    
    # Set up bot commands menu (appears when user types "/")
    async def post_init(app: Application):
        """Set bot commands after initialization"""
        from telegram import BotCommand
        await app.bot.set_my_commands([
            BotCommand("start", "🌿 ابدأ من جديد - Start over"),
            BotCommand("help", "❓ المساعدة - Get help"),
        ])
        print("✅ Bot commands menu configured")
    
    application.post_init = post_init
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))  # Help = Start
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Start bot with polling
    print("✅ Telegram bot started with polling!")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
