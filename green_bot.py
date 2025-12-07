import os
import requests
import numpy as np
import tensorflow as tf
import json
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from pathlib import Path
from PIL import Image
from io import BytesIO
import gdown
import zipfile
from collections import OrderedDict
import gc

# --- CONFIGURATION ---
app = Flask(__name__)

# Download models from Dropbox
def download_models_from_dropbox():
    models_dir = Path('models')
    if models_dir.exists() and len(list(models_dir.glob('*.keras'))) >= 18:
        print("✅ Models exist")
        return
    
    print("📥 Downloading models from Dropbox...")
    
    # Your Dropbox direct download link (change dl=0 to dl=1)
    DROPBOX_URL = "https://www.dropbox.com/scl/fi/1qhklwrp1qxe8cvsa0zf9/models.zip?rlkey=69s5wrz9kjg9dkb7yhkjz45xa&st=djrc963c&dl=1"  # Replace with your Dropbox link
    
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
MAX_MODELS = 10  # Keep max 10 models in memory
loaded_models = OrderedDict()
class_indices = {}
translations = {}

# --- LOAD RESOURCES ---
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

# --- MODEL LOADING WITH LRU CACHE ---
def get_model(model_name):
    """Load model with LRU cache - keeps max 10 models in memory"""
    
    # If already loaded, move to end (most recent)
    if model_name in loaded_models:
        loaded_models.move_to_end(model_name)
        print(f"♻️ Using cached model: {model_name}")
        return loaded_models[model_name]
    
    model_path = MODELS_DIR / f"{model_name}_final.keras"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Remove oldest model if at limit
    if len(loaded_models) >= MAX_MODELS:
        oldest = next(iter(loaded_models))
        print(f"🗑️ Removing old model: {oldest} (freeing memory)")
        del loaded_models[oldest]
        gc.collect()  # Force garbage collection
    
    # Load new model
    print(f"📥 Loading model: {model_name}... ({len(loaded_models)+1}/{MAX_MODELS})")
    model = tf.keras.models.load_model(model_path)
    loaded_models[model_name] = model
    print(f"✅ Model loaded: {model_name}")
    
    return model

# --- PREDICTION ---
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

# --- WHATSAPP BOT ---
user_sessions = {}

@app.route('/whatsapp', methods=['POST'])
def whatsapp_reply():
    try:
        print("🔵 Request received!")
        incoming_msg = request.values.get('Body', '').strip().lower()
        sender = request.values.get('From', '')
        num_media = int(request.values.get('NumMedia', 0))
        
        print(f"📱 From: {sender}")
        print(f"📝 Message: {incoming_msg}")
        print(f"📸 Media count: {num_media}")
    
    resp = MessagingResponse()
    msg = resp.message()
    
    # GREETING
    if incoming_msg in ['hi', 'hello', 'هلا', 'سلام', 'بداية', 'start', 'menu']:
        user_sessions.pop(sender, None)
        msg.body("🌿 *أهلاً بك في طبيب النباتات السعودي!* 🇸🇦\n\nاختر المحصول (أرسل الرقم): 👇\n\n" +
                 "1. 🌴 نخيل - أوراق\n" +
                 "2. 🍊 حمضيات - ثمار\n" +
                 "3. 🍃 حمضيات - أوراق\n" +
                 "4. 🍅 طماطم - أوراق\n" +
                 "5. 🥔 بطاطس - أوراق\n" +
                 "6. 🥒 خيار - أوراق\n" +
                 "7. 🌽 ذرة - أوراق\n" +
                 "8. 🍇 عنب - أوراق\n" +
                 "9. 🍎 رمان - ثمار\n" +
                 "10. 🥬 خس - أوراق\n" +
                 "11. 🌾 قمح - أوراق\n" +
                 "12. 🥭 مانجو - أوراق\n" +
                 "13. 🍌 موز - ثمار\n" +
                 "14. 🍃 موز - أوراق\n" +
                 "15. 🫘 فاصوليا - أوراق\n" +
                 "16. 🍆 باذنجان - أوراق\n" +
                 "17. 🥗 ملفوف - أوراق\n" +
                 "18. 🌶️ فلفل - أوراق")
        return str(resp)

    # CROP SELECTION
    crop_map = {
        '1': 'date_palm_leaves', '2': 'citrus_fruits', '3': 'citrus_leaves',
        '4': 'tomato_leaves', '5': 'potato_leaves', '6': 'cucumber_leaves',
        '7': 'corn_leaves', '8': 'grape_leaves', '9': 'pomegranate_fruits',
        '10': 'lettuce_leaves', '11': 'wheat_leaves', '12': 'mango_leaves',
        '13': 'banana_fruits', '14': 'banana_leaves', '15': 'bean_leaves',
        '16': 'eggplant_leaves', '17': 'cabbage_leaves', '18': 'pepper_leaves'
    }
    
    if incoming_msg in crop_map:
        selected_crop = crop_map[incoming_msg]
        user_sessions[sender] = selected_crop
        
        crop_display = {
            'date_palm_leaves': 'نخيل - أوراق', 'citrus_fruits': 'حمضيات - ثمار',
            'citrus_leaves': 'حمضيات - أوراق', 'tomato_leaves': 'طماطم - أوراق',
            'potato_leaves': 'بطاطس - أوراق', 'cucumber_leaves': 'خيار - أوراق',
            'corn_leaves': 'ذرة - أوراق', 'grape_leaves': 'عنب - أوراق',
            'pomegranate_fruits': 'رمان - ثمار', 'lettuce_leaves': 'خس - أوراق',
            'wheat_leaves': 'قمح - أوراق', 'mango_leaves': 'مانجو - أوراق',
            'banana_fruits': 'موز - ثمار', 'banana_leaves': 'موز - أوراق',
            'bean_leaves': 'فاصوليا - أوراق', 'eggplant_leaves': 'باذنجان - أوراق',
            'cabbage_leaves': 'ملفوف - أوراق', 'pepper_leaves': 'فلفل - أوراق'
        }
        
        crop_name = crop_display.get(selected_crop, selected_crop)
        sample_type = 'الثمرة' if 'fruits' in selected_crop else 'الورقة'
        msg.body(f"✅ تم اختيار: *{crop_name}*.\n\n📸 *الآن، أرسل صورة {sample_type} المصابة.*")
        return str(resp)
    
    # IMAGE HANDLING
    if num_media > 0:
        current_crop = user_sessions.get(sender)
        
        if not current_crop:
            msg.body("⚠️ *الرجاء اختيار المحصول أولاً!* \nأرسل كلمة 'هلا' للبدء.")
            return str(resp)
            
        image_url = request.values.get('MediaUrl0')
        
        try:
            print(f"📥 Downloading: {image_url}")
            TWILIO_SID = os.getenv('TWILIO_SID')
            TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
            
            response = requests.get(image_url, auth=(TWILIO_SID, TWILIO_TOKEN), allow_redirects=True)
            
            if response.status_code != 200:
                msg.body("❌ فشل تحميل الصورة.")
                return str(resp)
            
            img_data = response.content
            print(f"   ✅ Downloaded {len(img_data)} bytes")
            
            model = get_model(current_crop)
            if not model:
                msg.body("❌ عذراً، هذا الموديل غير متوفر حالياً.")
                return str(resp)
                
            print(f"🔬 Analyzing image...")
            diagnosis, conf = predict_image(model, img_data, current_crop)
            print(f"   ✅ Result: {diagnosis} ({conf:.1f}%)")
            
            # Build result text
            result_text = f"🔍 *التشخيص:* {diagnosis}\n🎯 *الدقة:* {conf:.1f}%\n\n"
            
            if conf < 60:
                result_text += "⚠️ *ملاحظة:* لست متأكداً تماماً. يرجى استشارة مهندس زراعي."
            else:
                result_text += "✅ *التشخيص موثوق.*"
            
            # Send reply with error handling
            try:
                msg.body(result_text)
                print(f"✅ Sent reply successfully")
            except Exception as reply_error:
                print(f"❌ Failed to send reply: {reply_error}")
                # Try simpler message
                msg.body(f"التشخيص: {diagnosis}\nالدقة: {conf:.1f}%")
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            msg.body("❌ حدث خطأ أثناء تحليل الصورة. حاول مرة أخرى.")
            
        return str(resp)

    # FALLBACK
    msg.body("🤖 لم أفهم رسالتك. أرسل 'هلا' للبدء.")
    return str(resp)
    
    except Exception as e:
        print(f"🔴 CRITICAL ERROR in whatsapp_reply: {e}")
        import traceback
        traceback.print_exc()
        resp = MessagingResponse()
        resp.message("❌ حدث خطأ. حاول مرة أخرى.")
        return str(resp)

@app.route('/health')
def health():
    return 'OK', 200

if __name__ == '__main__':
    load_resources()
    app.run(port=5000)
