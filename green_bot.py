import os
import requests
import numpy as np
import tensorflow as tf
import json
from flask import Flask, request, render_template_string, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from pathlib import Path
from PIL import Image
from io import BytesIO
from analytics import get_analytics
import gdown
import zipfile

def download_models_from_drive():
    models_dir = Path('models')
    if models_dir.exists() and len(list(models_dir.glob('*.keras'))) >= 18:
        print("✅ Models exist")
        return
    
    print("📥 Downloading models...")
    FILE_ID = "1-roxByuAoh1uQuVRz9c2t36L2L18EOVu"
    
    try:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, "models.zip", quiet=False)
        
        with zipfile.ZipFile("models.zip", 'r') as z:
            z.extractall('.')
        os.remove("models.zip")
        print("✅ Done!")
    except Exception as e:
        print(f"❌ Error: {e}")

# --- CONFIGURATION ---
app = Flask(__name__)

# Paths (Adjusted for Colab Local Uploads)
BASE_DIR = Path('.')
MODELS_DIR = BASE_DIR / 'models'
download_models_from_drive()

# Check Local Dir first, then Drive
if Path('disease_translations.json').exists():
    TRANSLATIONS_FILE = Path('disease_translations.json')
else:
    TRANSLATIONS_FILE = BASE_DIR / 'disease_translations.json'

if Path('class_indices.json').exists():
    INDICES_FILE = Path('class_indices.json')
else:
    INDICES_FILE = BASE_DIR / 'class_indices.json'

# Global Cache for Models (Lazy Loading)
loaded_models = {}
class_indices = {}
translations = {}

# Analytics Instance
analytics = None

# --- LOAD RESOURCES ---
def load_resources():
    global class_indices, translations, analytics
    
    # Load Translations
    if TRANSLATIONS_FILE.exists():
        with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        print("✅ Translations Loaded.")
    else:
        print("⚠️ Warning: Translations file not found.")

    # Load Class Indices
    if INDICES_FILE.exists():
        with open(INDICES_FILE, 'r') as f:
            class_indices = json.load(f)
        print("✅ Class Indices Loaded.")
    else:
        print("❌ CRITICAL: Class indices not found.")

# --- HELPER FUNCTIONS ---
def get_model(crop_name):
    """Loads model into RAM only when requested."""
    if crop_name in loaded_models:
        return loaded_models[crop_name]
    
    model_path = MODELS_DIR / f"{crop_name}_final.keras"
    if not model_path.exists():
        model_path = MODELS_DIR / f"{crop_name}_best.keras"
    
    if not model_path.exists():
        return None
        
    print(f"⏳ Loading Model: {crop_name}...")
    model = tf.keras.models.load_model(model_path)
    loaded_models[crop_name] = model # Cache it
    return model

def predict_image(model, img_bytes, crop_name):
    # Preprocess
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # CRITICAL: Apply same preprocessing as training (EfficientNet expects this!)
    # This converts from [0, 255] to [-1, 1] range
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    img_array = np.expand_dims(img_array, axis=0) # Batch of 1
    
    # Predict
    preds = model.predict(img_array)
    score = preds[0] # Model already has Softmax output
    
    # Decode
    idx = np.argmax(score)
    confidence = 100 * np.max(score)
    
    # Get Label (English)
    labels_map = class_indices.get(crop_name)
    if not labels_map: return "Error: Label Map Missing", 0
    english_label = labels_map.get(str(idx), "Unknown")
    
    # Translate to Arabic
    arabic_label = english_label
    if crop_name in translations and english_label in translations[crop_name]:
        arabic_label = translations[crop_name][english_label]
        
    return arabic_label, confidence

# --- WHATSAPP BOT LOGIC ---
# User State: { 'phone_number': 'current_crop' }
user_sessions = {}

@app.route('/whatsapp', methods=['POST'])
def whatsapp_reply():
    incoming_msg = request.values.get('Body', '').strip().lower()
    sender = request.values.get('From', '')
    num_media = int(request.values.get('NumMedia', 0))
    
    resp = MessagingResponse()
    msg = resp.message()
    
    # 1. GREETING / RESET
    if incoming_msg in ['hi', 'hello', 'هلا', 'سلام', 'بداية', 'start', 'menu']:
        user_sessions.pop(sender, None) # Reset session
        msg.body("🌿 *أهلاً بك في طبيب النباتات السعودي!* 🇸🇦\n\nاختر المحصول (أرسل الرقم): 👇\n\n" + \
                 "1. 🌴 نخيل - أوراق (Date Palm Leaves)\n" + \
                 "2. 🍊 حمضيات - ثمار (Citrus Fruits)\n" + \
                 "3. 🍃 حمضيات - أوراق (Citrus Leaves)\n" + \
                 "4. 🍅 طماطم - أوراق (Tomato Leaves)\n" + \
                 "5. 🥔 بطاطس - أوراق (Potato Leaves)\n" + \
                 "6. 🥒 خيار - أوراق (Cucumber Leaves)\n" + \
                 "7. 🌽 ذرة - أوراق (Corn Leaves)\n" + \
                 "8. 🍇 عنب - أوراق (Grape Leaves)\n" + \
                 "9. 🍎 رمان - ثمار (Pomegranate Fruits)\n" + \
                 "10. 🥬 خس - أوراق (Lettuce Leaves)\n" + \
                 "11. 🌾 قمح - أوراق (Wheat Leaves)\n" + \
                 "12. 🥭 مانجو - أوراق (Mango Leaves)\n" + \
                 "13. 🍌 موز - ثمار (Banana Fruits)\n" + \
                 "14. 🍃 موز - أوراق (Banana Leaves)\n" + \
                 "15. 🫘 فاصوليا - أوراق (Bean Leaves)\n" + \
                 "16. 🍆 باذنجان - أوراق (Eggplant Leaves)\n" + \
                 "17. 🥗 ملفوف - أوراق (Cabbage Leaves)\n" + \
                 "18. 🌶️ فلفل - أوراق (Pepper Leaves)")
        return str(resp)

    # 2. CROP SELECTION
    crop_map = {
        '1': 'date_palm_leaves',
        '2': 'citrus_fruits',
        '3': 'citrus_leaves',
        '4': 'tomato_leaves',
        '5': 'potato_leaves',
        '6': 'cucumber_leaves',
        '7': 'corn_leaves',
        '8': 'grape_leaves',
        '9': 'pomegranate_fruits',
        '10': 'lettuce_leaves',
        '11': 'wheat_leaves',
        '12': 'mango_leaves',
        '13': 'banana_fruits',
        '14': 'banana_leaves',
        '15': 'bean_leaves',
        '16': 'eggplant_leaves',
        '17': 'cabbage_leaves',
        '18': 'pepper_leaves'
    }
    
    if incoming_msg in crop_map:
        selected_crop = crop_map[incoming_msg]
        user_sessions[sender] = selected_crop
        
        # Create friendly confirmation message
        crop_display = {
            'date_palm_leaves': 'نخيل - أوراق',
            'citrus_fruits': 'حمضيات - ثمار',
            'citrus_leaves': 'حمضيات - أوراق',
            'tomato_leaves': 'طماطم - أوراق',
            'potato_leaves': 'بطاطس - أوراق',
            'cucumber_leaves': 'خيار - أوراق',
            'corn_leaves': 'ذرة - أوراق',
            'grape_leaves': 'عنب - أوراق',
            'pomegranate_fruits': 'رمان - ثمار',
            'lettuce_leaves': 'خس - أوراق',
            'wheat_leaves': 'قمح - أوراق',
            'mango_leaves': 'مانجو - أوراق',
            'banana_fruits': 'موز - ثمار',
            'banana_leaves': 'موز - أوراق',
            'bean_leaves': 'فاصوليا - أوراق',
            'eggplant_leaves': 'باذنجان - أوراق',
            'cabbage_leaves': 'ملفوف - أوراق',
            'pepper_leaves': 'فلفل - أوراق'
        }
        
        crop_name = crop_display.get(selected_crop, selected_crop)
        msg.body(f"✅ تم اختيار: *{crop_name}*.\n\n📸 *الآن، أرسل صورة {'الثمرة' if 'fruits' in selected_crop else 'الورقة'} المصابة.*")
        return str(resp)
    
    # 3. IMAGE HANDLING
    if num_media > 0:
        current_crop = user_sessions.get(sender)
        
        if not current_crop:
            msg.body("⚠️ *الرجاء اختيار المحصول أولاً!* \nأرسل كلمة 'هلا' للبدء.")
            return str(resp)
            
        # Get Image URL
        image_url = request.values.get('MediaUrl0')
        
        # Download Image
        try:
            print(f"📥 Downloading: {image_url}")
            # Use Basic Auth with User Credentials (The Nuclear Option)
            TWILIO_SID = os.getenv('TWILIO_SID')
            TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
            
            response = requests.get(image_url, auth=(TWILIO_SID, TWILIO_TOKEN), allow_redirects=True)
            print(f"   🔹 Status Code: {response.status_code}")
            print(f"   🔹 Content Type: {response.headers.get('Content-Type')}")
            
            if response.status_code != 200:
                print(f"   ❌ Failed to download. Body: {response.text[:100]}")
                msg.body("❌ فشل تحميل الصورة من المصدر.")
                return str(resp)
            
            img_data = response.content
            print(f"   🔹 Downloaded Bytes: {len(img_data)}")
            
            # Load Model
            model = get_model(current_crop)
            if not model:
                msg.body("❌ عذراً، هذا الموديل غير متوفر حالياً.")
                return str(resp)
                
            # Predict
            diagnosis, conf = predict_image(model, img_data, current_crop)
            
            # Log to Analytics
            if analytics:
                try:
                    analytics.log_detection(
                        user_id=sender,
                        crop=current_crop,
                        diagnosis=diagnosis,
                        confidence=conf
                    )
                except Exception as e:
                    print(f"⚠️ Analytics logging failed: {e}")
            
            # Reply
            result_text = f"🔍 *التشخيص:* {diagnosis}\n🎯 *الدقة:* {conf:.1f}%\n\n"
            
            if conf < 60:
                result_text += "⚠️ *ملاحظة:* لست متأكداً تماماً. يرجى استشارة مهندس زراعي أو إرسال صورة أوضح."
            else:
                result_text += "✅ *التشخيص موثوق.*"
                
            msg.body(result_text)
            
        except Exception as e:
            print(f"Error: {e}")
            msg.body("❌ حدث خطأ أثناء تحليل الصورة. حاول مرة أخرى.")
            
        return str(resp)

    # 4. FALLBACK
    msg.body("🤖 لم أفهم رسالتك. أرسل 'هلا' للبدء.")
    return str(resp)

# --- ANALYTICS DASHBOARD ---
@app.route('/analytics')
def analytics_dashboard():
    """Simple analytics dashboard"""
    if not analytics:
        return "Analytics not initialized", 500
    
    stats = analytics.get_statistics()
    daily_report = analytics.get_daily_report()
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Bot Analytics</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            h1 {{ color: #2e7d32; }}
            .stat-card {{
                background: white;
                padding: 20px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-number {{
                font-size: 32px;
                font-weight: bold;
                color: #1976d2;
            }}
            .stat-label {{
                color: #666;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin-top: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background: #2e7d32;
                color: white;
            }}
        </style>
    </head>
    <body>
        <h1>🌿 Plant Disease Bot Analytics Dashboard</h1>
        
        <h2>Overall Statistics</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div class="stat-card">
                <div class="stat-number">{stats['total_detections']}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['unique_users']}</div>
                <div class="stat-label">Unique Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['health_ratio']:.1f}%</div>
                <div class="stat-label">Healthy Plants</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['avg_confidence']:.1f}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>
        
        {'<h2>Today\'s Report</h2>' + generate_daily_html(daily_report) if daily_report else '<p>No data for today yet.</p>'}
        
        <p style="margin-top: 40px; color: #888; text-align: center;">
            Generated: {from datetime import datetime; datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </body>
    </html>
    '''
    
    return render_template_string(html)

def generate_daily_html(report):
    """Generate HTML for daily report"""
    crops_html = '<table><tr><th>Crop</th><th>Count</th></tr>'
    for crop in report['top_crops']:
        crops_html += f'<tr><td>{crop["crop"]}</td><td>{crop["count"]}</td></tr>'
    crops_html += '</table>'
    
    diseases_html = '<table><tr><th>Crop</th><th>Disease</th><th>Count</th></tr>'
    for disease in report['top_diseases']:
        diseases_html += f'<tr><td>{disease["crop"]}</td><td>{disease["disease"]}</td><td>{disease["count"]}</td></tr>'
    diseases_html += '</table>'
    
    return f'''
        <div class="stat-card">
            <p><strong>Total Queries:</strong> {report['total_queries']}</p>
            <p><strong>Healthy:</strong> {report['healthy_count']} | <strong>Diseased:</strong> {report['diseased_count']}</p>
            <p><strong>Avg Confidence:</strong> {report['avg_confidence']:.1f}%</p>
        </div>
        
        <h3>Top Crops Today</h3>
        {crops_html}
        
        <h3>Top Diseases Detected</h3>
        {diseases_html}
    '''

@app.route('/health')
def health():
    """Health check endpoint for keep-alive"""
    return 'OK', 200

if __name__ == '__main__':
    load_resources()
    
    # Initialize Analytics
    analytics = get_analytics('analytics.db')
    print("✅ Analytics initialized.")
    
    app.run(port=5000)
