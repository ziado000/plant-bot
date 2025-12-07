"""
Statistics Logger for Plant Disease Bot
Logs all predictions to CSV file for analysis
"""

import csv
import os
from datetime import datetime
from pathlib import Path

STATS_FILE = "bot_statistics.csv"

def initialize_stats_file():
    """Create CSV file with headers if it doesn't exist"""
    if not Path(STATS_FILE).exists():
        with open(STATS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Date',
                'Time',
                'User_ID',
                'Crop_Type',
                'Disease_Detected',
                'Confidence_%',
                'Platform'
            ])
        print(f"📊 Created statistics file: {STATS_FILE}")

def log_prediction(user_id, crop_type, disease, confidence, platform="Telegram"):
    """
    Log a prediction to the CSV file
    
    Args:
        user_id: Telegram user ID
        crop_type: e.g. "tomato_leaves", "banana_fruits"
        disease: Diagnosed disease (bilingual format)
        confidence: Confidence percentage (0-100)
        platform: "Telegram", "WhatsApp", etc.
    """
    try:
        # Ensure file exists
        initialize_stats_file()
        
        now = datetime.now()
        timestamp = now.isoformat()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        # Append to CSV
        with open(STATS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                date,
                time,
                user_id,
                crop_type,
                disease,
                f"{confidence:.2f}",
                platform
            ])
        
        print(f"📊 Logged: {crop_type} → {disease} ({confidence:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ Failed to log statistics: {e}")
        # Don't crash the bot if logging fails

def get_statistics_summary():
    """Get summary statistics from the CSV file"""
    if not Path(STATS_FILE).exists():
        return "No statistics available yet"
    
    with open(STATS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return "No predictions logged yet"
    
    total = len(rows)
    
    # Count by crop
    crops = {}
    diseases = {}
    
    for row in rows:
        crop = row['Crop_Type']
        disease = row['Disease_Detected']
        
        crops[crop] = crops.get(crop, 0) + 1
        diseases[disease] = diseases.get(disease, 0) + 1
    
    summary = f"""
📊 Statistics Summary
━━━━━━━━━━━━━━━━━━
Total Predictions: {total}

Top Crops:
{chr(10).join(f"  • {crop}: {count}" for crop, count in sorted(crops.items(), key=lambda x: x[1], reverse=True)[:5])}

Top Diseases:
{chr(10).join(f"  • {disease}: {count}" for disease, count in sorted(diseases.items(), key=lambda x: x[1], reverse=True)[:5])}
"""
    
    return summary
