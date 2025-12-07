#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytics Database Module
Handles all data logging and analytics for the plant disease detection bot
"""

import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import threading

class BotAnalytics:
    def __init__(self, db_path='analytics.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_hash TEXT NOT NULL,
                    crop TEXT NOT NULL,
                    diagnosis TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_healthy BOOLEAN NOT NULL,
                    session_id TEXT
                )
            ''')
            
            # Daily summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date DATE PRIMARY KEY,
                    total_queries INTEGER DEFAULT 0,
                    unique_users INTEGER DEFAULT 0,
                    healthy_count INTEGER DEFAULT 0,
                    diseased_count INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0
                )
            ''')
            
            # Crop statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crop_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    crop TEXT NOT NULL,
                    count INTEGER DEFAULT 0,
                    UNIQUE(date, crop)
                )
            ''')
            
            # Disease statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS disease_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    crop TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    count INTEGER DEFAULT 0,
                    UNIQUE(date, crop, disease)
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def hash_user_id(self, user_id):
        """Hash user ID for privacy"""
        return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]
    
    def log_detection(self, user_id, crop, diagnosis, confidence, session_id=None):
        """Log a detection event"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            user_hash = self.hash_user_id(user_id)
            is_healthy = 'healthy' in diagnosis.lower()
            
            cursor.execute('''
                INSERT INTO detections 
                (user_hash, crop, diagnosis, confidence, is_healthy, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_hash, crop, diagnosis, confidence, is_healthy, session_id))
            
            conn.commit()
            conn.close()
            
            # Update daily stats
            self._update_daily_stats()
    
    def _update_daily_stats(self):
        """Update daily statistics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            # Get today's stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT user_hash) as unique_users,
                    SUM(CASE WHEN is_healthy = 1 THEN 1 ELSE 0 END) as healthy,
                    SUM(CASE WHEN is_healthy = 0 THEN 1 ELSE 0 END) as diseased,
                    AVG(confidence) as avg_conf
                FROM detections
                WHERE DATE(timestamp) = ?
            ''', (today,))
            
            stats = cursor.fetchone()
            
            # Insert or update
            cursor.execute('''
                INSERT OR REPLACE INTO daily_summary 
                (date, total_queries, unique_users, healthy_count, diseased_count, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (today, stats[0], stats[1], stats[2], stats[3], stats[4]))
            
            # Update crop stats
            cursor.execute('''
                INSERT OR REPLACE INTO crop_stats (date, crop, count)
                SELECT DATE(timestamp), crop, COUNT(*)
                FROM detections
                WHERE DATE(timestamp) = ?
                GROUP BY crop
            ''', (today,))
            
            # Update disease stats
            cursor.execute('''
                INSERT OR REPLACE INTO disease_stats (date, crop, disease, count)
                SELECT DATE(timestamp), crop, diagnosis, COUNT(*)
                FROM detections
                WHERE DATE(timestamp) = ? AND is_healthy = 0
                GROUP BY crop, diagnosis
            ''', (today,))
            
            conn.commit()
            conn.close()
    
    def get_daily_report(self, date=None):
        """Get daily report"""
        if date is None:
            date = datetime.now().date()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('''
            SELECT total_queries, unique_users, healthy_count, diseased_count, avg_confidence
            FROM daily_summary
            WHERE date = ?
        ''', (date,))
        
        summary = cursor.fetchone()
        
        if not summary:
            conn.close()
            return None
        
        # Top crops
        cursor.execute('''
            SELECT crop, count
            FROM crop_stats
            WHERE date = ?
            ORDER BY count DESC
            LIMIT 5
        ''', (date,))
        
        top_crops = cursor.fetchall()
        
        # Top diseases
        cursor.execute('''
            SELECT crop, disease, count
            FROM disease_stats
            WHERE date = ?
            ORDER BY count DESC
            LIMIT 5
        ''', (date,))
        
        top_diseases = cursor.fetchall()
        
        conn.close()
        
        return {
            'date': str(date),
            'total_queries': summary[0],
            'unique_users': summary[1],
            'healthy_count': summary[2],
            'diseased_count': summary[3],
            'avg_confidence': round(summary[4], 2) if summary[4] else 0,
            'top_crops': [{'crop': c[0], 'count': c[1]} for c in top_crops],
            'top_diseases': [{'crop': d[0], 'disease': d[1], 'count': d[2]} for d in top_diseases]
        }
    
    def get_period_report(self, start_date, end_date):
        """Get report for a period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('''
            SELECT 
                SUM(total_queries) as total,
                AVG(unique_users) as avg_users,
                SUM(healthy_count) as healthy,
                SUM(diseased_count) as diseased,
                AVG(avg_confidence) as avg_conf
            FROM daily_summary
            WHERE date BETWEEN ? AND ?
        ''', (start_date, end_date))
        
        summary = cursor.fetchone()
        
        # Top crops
        cursor.execute('''
            SELECT crop, SUM(count) as total
            FROM crop_stats
            WHERE date BETWEEN ? AND ?
            GROUP BY crop
            ORDER BY total DESC
            LIMIT 10
        ''', (start_date, end_date))
        
        top_crops = cursor.fetchall()
        
        # Top diseases
        cursor.execute('''
            SELECT crop, disease, SUM(count) as total
            FROM disease_stats
            WHERE date BETWEEN ? AND ?
            GROUP BY crop, disease
            ORDER BY total DESC
            LIMIT 10
        ''', (start_date, end_date))
        
        top_diseases = cursor.fetchall()
        
        # Daily trend
        cursor.execute('''
            SELECT date, total_queries, diseased_count
            FROM daily_summary
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', (start_date, end_date))
        
        daily_trend = cursor.fetchall()
        
        conn.close()
        
        return {
            'period': f"{start_date} to {end_date}",
            'total_queries': summary[0] or 0,
            'avg_daily_users': round(summary[1], 1) if summary[1] else 0,
            'healthy_count': summary[2] or 0,
            'diseased_count': summary[3] or 0,
            'avg_confidence': round(summary[4], 2) if summary[4] else 0,
            'top_crops': [{'crop': c[0], 'count': c[1]} for c in top_crops],
            'top_diseases': [{'crop': d[0], 'disease': d[1], 'count': d[2]} for d in top_diseases],
            'daily_trend': [{'date': str(d[0]), 'total': d[1], 'diseased': d[2]} for d in daily_trend]
        }
    
    def export_monthly_report(self, year, month):
        """Export monthly report for ministry"""
        start_date = datetime(year, month, 1).date()
        
        # Calculate last day of month
        if month == 12:
            end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)
        
        report = self.get_period_report(start_date, end_date)
        report['month'] = f"{year}-{month:02d}"
        report['generated_at'] = datetime.now().isoformat()
        
        return report
    
    def get_statistics(self):
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT user_hash) as users,
                SUM(CASE WHEN is_healthy = 1 THEN 1 ELSE 0 END) as healthy,
                SUM(CASE WHEN is_healthy = 0 THEN 1 ELSE 0 END) as diseased,
                AVG(confidence) as avg_conf
            FROM detections
        ''')
        
        stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_detections': stats[0],
            'unique_users': stats[1],
            'healthy_plants': stats[2],
            'diseased_plants': stats[3],
            'avg_confidence': round(stats[4], 2) if stats[4] else 0,
            'health_ratio': round(stats[2] / stats[0] * 100, 1) if stats[0] > 0 else 0
        }

# Singleton instance
_analytics = None

def get_analytics(db_path='analytics.db'):
    """Get or create analytics instance"""
    global _analytics
    if _analytics is None:
        _analytics = BotAnalytics(db_path)
    return _analytics
