"""
Trading Bot Notification System
Umfassendes Benachrichtigungssystem für Telegram, E-Mail, Discord und andere Kanäle
"""

import asyncio
import smtplib
import ssl
import requests
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from pathlib import Path
import sqlite3
import schedule
from dataclasses import dataclass, asdict
import traceback

class NotificationLevel(Enum):
    """Prioritätslevel für Benachrichtigungen"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class NotificationChannel(Enum):
    """Verfügbare Benachrichtigungskanäle"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    FILE = "file"
    CONSOLE = "console"

@dataclass
class NotificationMessage:
    """Struktur für eine Benachrichtigung"""
    title: str
    message: str
    level: NotificationLevel
    category: str
    timestamp: datetime
    data: Dict[str, Any] = None
    channels: List[NotificationChannel] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.channels is None:
            self.channels = []

class TelegramNotifier:
    """Telegram-Benachrichtigungen"""
    
    def __init__(self, bot_token: str, chat_id: str, logger=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logger
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Rate Limiting
        self.last_message_time = 0
        self.min_interval = 1  # Mindestabstand zwischen Nachrichten in Sekunden
        
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Sendet eine Nachricht über Telegram"""
        try:
            # Rate Limiting
            current_time = time.time()
            if current_time - self.last_message_time < self.min_interval:
                time.sleep(self.min_interval - (current_time - self.last_message_time))
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            
            self.last_message_time = time.time()
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Telegram-Fehler: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Sendet ein Foto über Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                }
                
                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Telegram Photo-Fehler: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Testet die Telegram-Verbindung"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return True
        except Exception:
            return False

class EmailNotifier:
    """E-Mail-Benachrichtigungen"""
    
    def __init__(self, smtp_server: str, smtp_port: int, email: str, password: str,
                 recipient_email: str, logger=None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
        self.recipient_email = recipient_email
        self.logger = logger
        
    def send_email(self, subject: str, body: str, attachments: List[str] = None) -> bool:
        """Sendet eine E-Mail"""
        try:
            # E-Mail erstellen
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            # Text hinzufügen
            msg.attach(MIMEText(body, 'html'))
            
            # Anhänge hinzufügen
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(file_path)}'
                        )
                        msg.attach(part)
            
            # E-Mail senden
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email, self.password)
                server.sendmail(self.email, self.recipient_email, msg.as_string())
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"E-Mail-Fehler: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Testet die E-Mail-Verbindung"""
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email, self.password)
            return True
        except Exception:
            return False

class DiscordNotifier:
    """Discord-Benachrichtigungen über Webhook"""
    
    def __init__(self, webhook_url: str, logger=None):
        self.webhook_url = webhook_url
        self.logger = logger
        
    def send_message(self, message: str, username: str = "Trading Bot") -> bool:
        """Sendet eine Nachricht über Discord Webhook"""
        try:
            payload = {
                'content': message,
                'username': username
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Discord-Fehler: {e}")
            return False
    
    def send_embed(self, title: str, description: str, color: int = 0x00ff00,
                   fields: List[Dict] = None) -> bool:
        """Sendet eine eingebettete Nachricht"""
        try:
            embed = {
                'title': title,
                'description': description,
                'color': color,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if fields:
                embed['fields'] = fields
            
            payload = {
                'embeds': [embed],
                'username': 'Trading Bot'
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Discord Embed-Fehler: {e}")
            return False

class NotificationQueue:
    """Queue für Benachrichtigungen mit Priorität"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = []
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def add(self, notification: NotificationMessage):
        """Fügt eine Benachrichtigung zur Queue hinzu"""
        with self.lock:
            self.queue.append(notification)
            
            # Queue-Größe begrenzen
            if len(self.queue) > self.max_size:
                self.queue.pop(0)  # Älteste entfernen
            
            # Nach Priorität sortieren
            self.queue.sort(key=lambda x: self._get_priority_value(x.level), reverse=True)
    
    def get_next(self) -> Optional[NotificationMessage]:
        """Holt die nächste Benachrichtigung aus der Queue"""
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None
    
    def get_all(self) -> List[NotificationMessage]:
        """Holt alle Benachrichtigungen aus der Queue"""
        with self.lock:
            notifications = self.queue.copy()
            self.queue.clear()
            return notifications
    
    def size(self) -> int:
        """Gibt die aktuelle Queue-Größe zurück"""
        with self.lock:
            return len(self.queue)
    
    def _get_priority_value(self, level: NotificationLevel) -> int:
        """Konvertiert Prioritätslevel zu numerischem Wert"""
        priority_map = {
            NotificationLevel.LOW: 1,
            NotificationLevel.MEDIUM: 2,
            NotificationLevel.HIGH: 3,
            NotificationLevel.CRITICAL: 4
        }
        return priority_map.get(level, 1)

class NotificationHistory:
    """Speichert Benachrichtigungshistorie in SQLite"""
    
    def __init__(self, db_path: str = "notifications.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialisiert die Datenbank"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    data TEXT,
                    sent_successfully BOOLEAN NOT NULL DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON notifications(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_level ON notifications(level)
            """)
    
    def add(self, notification: NotificationMessage, sent_successfully: bool = False):
        """Fügt eine Benachrichtigung zur Historie hinzu"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO notifications 
                (timestamp, title, message, level, category, channels, data, sent_successfully)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                notification.timestamp.isoformat(),
                notification.title,
                notification.message,
                notification.level.value,
                notification.category,
                json.dumps([ch.value for ch in notification.channels]),
                json.dumps(notification.data),
                sent_successfully
            ))
    
    def get_recent(self, hours: int = 24) -> List[Dict]:
        """Holt aktuelle Benachrichtigungen"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM notifications 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff_time.isoformat(),))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old(self, days: int = 30):
        """Löscht alte Benachrichtigungen"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM notifications WHERE timestamp < ?
            """, (cutoff_time.isoformat(),))
            
            return cursor.rowcount

class NotificationSystem:
    """Hauptklasse für das Benachrichtigungssystem"""
    
    def __init__(self, config_manager=None, logger=None):
        self.config_manager = config_manager
        self.logger = logger
        
        # Komponenten
        self.queue = NotificationQueue()
        self.history = NotificationHistory()
        
        # Notifier
        self.telegram = None
        self.email = None
        self.discord = None
        
        # Konfiguration
        self.enabled_channels = []
        self.notification_rules = {}
        self.rate_limits = {}
        
        # Thread-Management
        self.worker_thread = None
        self.running = False
        
        self._load_config()
        self._init_notifiers()
        
    def _load_config(self):
        """Lädt Konfiguration"""
        if not self.config_manager:
            # Standard-Konfiguration
            self.config = {
                'telegram': {
                    'enabled': False,
                    'bot_token': '',
                    'chat_id': ''
                },
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'email': 'j.nijenhuis43@gmail.com',
                    'password': '',
                    'recipient': 'j.nijenhuis43@gmail.com'
                },
                'discord': {
                    'enabled': False,
                    'webhook_url': ''
                },
                'notification_rules': {
                    'trade': ['telegram', 'email'],
                    'error': ['telegram', 'email', 'discord'],
                    'system': ['telegram'],
                    'performance': ['email'],
                    'critical': ['telegram', 'email', 'discord']
                },
                'rate_limits': {
                    'telegram': 30,  # Max 30 Nachrichten pro Minute
                    'email': 10,     # Max 10 E-Mails pro Minute
                    'discord': 50    # Max 50 Nachrichten pro Minute
                }
            }
        else:
            self.config = self.config_manager.get_notification_config()
        
        # Regeln und Limits laden
        self.notification_rules = self.config.get('notification_rules', {})
        self.rate_limits = self.config.get('rate_limits', {})
    
    def _init_notifiers(self):
        """Initialisiert die verschiedenen Notifier"""
        try:
            # Telegram
            if self.config['telegram']['enabled']:
                self.telegram = TelegramNotifier(
                    self.config['telegram']['bot_token'],
                    self.config['telegram']['chat_id'],
                    self.logger
                )
                if self.telegram.test_connection():
                    self.enabled_channels.append(NotificationChannel.TELEGRAM)
                    if self.logger:
                        self.logger.info("Telegram-Notifier initialisiert", 'notification')
                else:
                    if self.logger:
                        self.logger.warning("Telegram-Verbindung fehlgeschlagen", 'notification')
            
            # E-Mail
            if self.config['email']['enabled']:
                self.email = EmailNotifier(
                    self.config['email']['smtp_server'],
                    self.config['email']['smtp_port'],
                    self.config['email']['email'],
                    self.config['email']['password'],
                    self.config['email']['recipient'],
                    self.logger
                )
                if self.email.test_connection():
                    self.enabled_channels.append(NotificationChannel.EMAIL)
                    if self.logger:
                        self.logger.info("E-Mail-Notifier initialisiert", 'notification')
                else:
                    if self.logger:
                        self.logger.warning("E-Mail-Verbindung fehlgeschlagen", 'notification')
            
            # Discord
            if self.config['discord']['enabled']:
                self.discord = DiscordNotifier(
                    self.config['discord']['webhook_url'],
                    self.logger
                )
                self.enabled_channels.append(NotificationChannel.DISCORD)
                if self.logger:
                    self.logger.info("Discord-Notifier initialisiert", 'notification')
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fehler bei Notifier-Initialisierung: {e}", 'notification')
    
    def start(self):
        """Startet das Benachrichtigungssystem"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            
            if self.logger:
                self.logger.info("Benachrichtigungssystem gestartet", 'notification')
            
            # Test-Nachricht senden
            self.notify(
                title="Trading Bot gestartet",
                message="Das Benachrichtigungssystem ist aktiv und bereit.",
                level=NotificationLevel.MEDIUM,
                category="system"
            )
    
    def stop(self):
        """Stoppt das Benachrichtigungssystem"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            
        if self.logger:
            self.logger.info("Benachrichtigungssystem gestoppt", 'notification')
    
    def _worker(self):
        """Worker-Thread für die Verarbeitung der Benachrichtigungen"""
        while self.running:
            try:
                notification = self.queue.get_next()
                if notification:
                    self._send_notification(notification)
                else:
                    time.sleep(1)  # Kurz warten wenn keine Nachrichten
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Fehler im Notification-Worker: {e}", 'notification')
                time.sleep(5)
    
    def notify(self, title: str, message: str, level: NotificationLevel,
               category: str, data: Dict[str, Any] = None,
               channels: List[NotificationChannel] = None):
        """Fügt eine neue Benachrichtigung hinzu"""
        
        notification = NotificationMessage(
            title=title,
            message=message,
            level=level,
            category=category,
            timestamp=datetime.now(),
            data=data or {},
            channels=channels or self._get_channels_for_category(category)
        )
        
        self.queue.add(notification)
        
        if self.logger:
            self.logger.debug(f"Benachrichtigung hinzugefügt: {title}", 'notification')
    
    def _get_channels_for_category(self, category: str) -> List[NotificationChannel]:
        """Ermittelt die Kanäle für eine Kategorie"""
        rule_channels = self.notification_rules.get(category, ['telegram'])
        channels = []
        
        for channel_name in rule_channels:
            try:
                channel = NotificationChannel(channel_name)
                if channel in self.enabled_channels:
                    channels.append(channel)
            except ValueError:
                continue
        
        return channels
    
    def _send_notification(self, notification: NotificationMessage):
        """Sendet eine Benachrichtigung über die konfigurierten Kanäle"""
        sent_successfully = False
        
        for channel in notification.channels:
            try:
                if channel == NotificationChannel.TELEGRAM and self.telegram:
                    message = self._format_telegram_message(notification)
                    if self.telegram.send_message(message):
                        sent_successfully = True
                
                elif channel == NotificationChannel.EMAIL and self.email:
                    subject = f"[Trading Bot] {notification.title}"
                    body = self._format_email_message(notification)
                    if self.email.send_email(subject, body):
                        sent_successfully = True
                
                elif channel == NotificationChannel.DISCORD and self.discord:
                    message = self._format_discord_message(notification)
                    if self.discord.send_message(message):
                        sent_successfully = True
                
                elif channel == NotificationChannel.CONSOLE:
                    print(f"\n[{notification.level.value}] {notification.title}")
                    print(f"{notification.message}")
                    print(f"Zeit: {notification.timestamp}")
                    sent_successfully = True
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Fehler beim Senden über {channel.value}: {e}", 'notification')
        
        # In Historie speichern
        self.history.add(notification, sent_successfully)
        
        if self.logger and sent_successfully:
            self.logger.debug(f"Benachrichtigung gesendet: {notification.title}", 'notification')
    
    def _format_telegram_message(self, notification: NotificationMessage) -> str:
        """Formatiert eine Nachricht für Telegram"""
        level_emoji = {
            NotificationLevel.LOW: "ℹ️",
            NotificationLevel.MEDIUM: "⚠️",
            NotificationLevel.HIGH: "🚨",
            NotificationLevel.CRITICAL: "🔥"
        }
        
        emoji = level_emoji.get(notification.level, "📢")
        
        message = f"{emoji} <b>{notification.title}</b>\n\n"
        message += f"{notification.message}\n\n"
        message += f"<i>Zeit: {notification.timestamp.strftime('%d.%m.%Y %H:%M:%S')}</i>\n"
        message += f"<i>Kategorie: {notification.category}</i>"
        
        # Zusätzliche Daten
        if notification.data:
            message += "\n\n<b>Details:</b>"
            for key, value in notification.data.items():
                if isinstance(value, (int, float)):
                    if key.lower() in ['price', 'amount', 'value', 'profit', 'loss']:
                        message += f"\n• {key}: {value:.8f}"
                    else:
                        message += f"\n• {key}: {value}"
                else:
                    message += f"\n• {key}: {str(value)[:100]}"
        
        # Telegram-Nachrichtenlimit beachten (4096 Zeichen)
        if len(message) > 4000:
            message = message[:3950] + "...\n\n<i>[Nachricht gekürzt]</i>"
        
        return message
    
    def _format_email_message(self, notification: NotificationMessage) -> str:
        """Formatiert eine Nachricht für E-Mail"""
        html_message = f"""
        <html>
        <body>
            <h2 style="color: {'red' if notification.level in [NotificationLevel.HIGH, NotificationLevel.CRITICAL] else 'orange' if notification.level == NotificationLevel.MEDIUM else 'blue'};">
                {notification.title}
            </h2>
            
            <p><strong>Level:</strong> {notification.level.value}</p>
            <p><strong>Kategorie:</strong> {notification.category}</p>
            <p><strong>Zeit:</strong> {notification.timestamp.strftime('%d.%m.%Y %H:%M:%S')}</p>
            
            <hr>
            
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                {notification.message.replace('\n', '<br>')}
            </div>
        """
        
        if notification.data:
            html_message += """
            <hr>
            <h3>Details:</h3>
            <table style="border-collapse: collapse; width: 100%;">
            """
            
            for key, value in notification.data.items():
                html_message += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{key}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{value}</td>
                </tr>
                """
            
            html_message += "</table>"
        
        html_message += """
            <hr>
            <p style="font-size: 12px; color: #666;">
                Diese Nachricht wurde automatisch vom Trading Bot gesendet.
            </p>
        </body>
        </html>
        """
        
        return html_message
    
    def _format_discord_message(self, notification: NotificationMessage) -> str:
        """Formatiert eine Nachricht für Discord"""
        level_emoji = {
            NotificationLevel.LOW: "🔵",
            NotificationLevel.MEDIUM: "🟡",
            NotificationLevel.HIGH: "🔴",
            NotificationLevel.CRITICAL: "💀"
        }
        
        emoji = level_emoji.get(notification.level, "📢")
        
        message = f"{emoji} **{notification.title}**\n\n"
        message += f"{notification.message}\n\n"
        message += f"*Zeit: {notification.timestamp.strftime('%d.%m.%Y %H:%M:%S')}*\n"
        message += f"*Kategorie: {notification.category}*"
        
        # Discord-Nachrichtenlimit beachten (2000 Zeichen)
        if len(message) > 1900:
            message = message[:1850] + "...\n\n*[Nachricht gekürzt]*"
        
        return message
    
    # Convenience-Methoden für verschiedene Nachrichtentypen
    def notify_trade(self, action: str, symbol: str, amount: float, price: float, **kwargs):
        """Benachrichtigung für Trades"""
        title = f"Trade ausgeführt: {action} {symbol}"
        message = f"Aktion: {action}\nSymbol: {symbol}\nMenge: {amount:.8f}\nPreis: {price:.8f}\nWert: {amount * price:.2f}"
        
        data = {
            'action': action,
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'value': amount * price,
            **kwargs
        }
        
        self.notify(title, message, NotificationLevel.MEDIUM, 'trade', data)
    
    def notify_error(self, error: Exception, context: str = "", **kwargs):
        """Benachrichtigung für Fehler"""
        title = f"Fehler aufgetreten: {type(error).__name__}"
        message = f"Fehler: {str(error)}\nKontext: {context}"
        
        data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stacktrace': traceback.format_exc()[:500],  # Gekürzt
            **kwargs
        }
        
        self.notify(title, message, NotificationLevel.HIGH, 'error', data)
    
    def notify_critical(self, title: str, message: str, **kwargs):
        """Kritische Benachrichtigung"""
        self.notify(title, message, NotificationLevel.CRITICAL, 'critical', kwargs)
    
    def notify_system_status(self, component: str, status: str, details: Dict = None):
        """System-Status-Benachrichtigung"""
        title = f"System-Status: {component}"
        message = f"Komponente: {component}\nStatus: {status}"
        
        data = {
            'component': component,
            'status': status,
            'details': details or {}
        }
        
        level = NotificationLevel.HIGH if status.lower() in ['error', 'failed', 'stopped'] else NotificationLevel.MEDIUM
        
        self.notify(title, message, level, 'system', data)
    
    def notify_performance(self, strategy: str, metrics: Dict[str, float], **kwargs):
        """Performance-Benachrichtigung"""
        title = f"Performance-Update: {strategy}"
        
        message_parts = [f"Strategie: {strategy}"]
        for key, value in metrics.items():
            if isinstance(value, float):
                message_parts.append(f"{key}: {value:.2f}")
        
        message = "\n".join(message_parts)
        
        data = {
            'strategy': strategy,
            'metrics': metrics,
            **kwargs
        }
        
        self.notify(title, message, NotificationLevel.LOW, 'performance', data)
    
    def send_daily_summary(self, summary_data: Dict[str, Any]):
        """Tägliche Zusammenfassung"""
        title = f"Tägliche Zusammenfassung - {datetime.now().strftime('%d.%m.%Y')}"
        
        message = f"""
Trades heute: {summary_data.get('trades_count', 0)}
Gewinn/Verlust: {summary_data.get('profit_loss', 0):.2f}
Aktive Positionen: {summary_data.get('active_positions', 0)}
Beste Strategie: {summary_data.get('best_strategy', 'N/A')}
Portfolio-Wert: {summary_data.get('portfolio_value', 0):.2f}
        """.strip()
        
        self.notify(title, message, NotificationLevel.MEDIUM, 'performance', summary_data)
    
    def send_chart(self, chart_path: str, title: str, description: str = ""):
        """Sendet ein Chart über Telegram"""
        if self.telegram and os.path.exists(chart_path):
            success = self.telegram.send_photo(chart_path, f"{title}\n\n{description}")
            if success and self.logger:
                self.logger.info(f"Chart gesendet: {title}", 'notification')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über das Benachrichtigungssystem zurück"""
        recent_notifications = self.history.get_recent(24)
        
        stats = {
            'total_notifications_24h': len(recent_notifications),
            'queue_size': self.queue.size(),
            'enabled_channels': [ch.value for ch in self.enabled_channels],
            'by_level': {},
            'by_category': {},
            'success_rate': 0
        }
        
        if recent_notifications:
            # Nach Level gruppieren
            for notif in recent_notifications:
                level = notif['level']
                stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
            
            # Nach Kategorie gruppieren
            for notif in recent_notifications:
                category = notif['category']
                stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # Erfolgsrate berechnen
            successful = sum(1 for n in recent_notifications if n['sent_successfully'])
            stats['success_rate'] = (successful / len(recent_notifications)) * 100
        
        return stats
    
    def test_all_channels(self) -> Dict[str, bool]:
        """Testet alle konfigurierten Kanäle"""
        results = {}
        
        if self.telegram:
            results['telegram'] = self.telegram.test_connection()
        
        if self.email:
            results['email'] = self.email.test_connection()
        
        if self.discord:
            # Discord Webhook hat keinen direkten Test, senden wir eine Test-Nachricht
            results['discord'] = self.discord.send_message("🔧 Test-Nachricht vom Trading Bot")
        
        return results
    
    def update_config(self, new_config: Dict[str, Any]):
        """Aktualisiert die Konfiguration zur Laufzeit"""
        self.config.update(new_config)
        self.notification_rules = self.config.get('notification_rules', {})
        self.rate_limits = self.config.get('rate_limits', {})
        
        # Notifier neu initialisieren
        self.enabled_channels.clear()
        self._init_notifiers()
        
        if self.logger:
            self.logger.info("Notification-Konfiguration aktualisiert", 'notification')
    
    def cleanup_history(self, days: int = 30) -> int:
        """Bereinigt alte Benachrichtigungen"""
        deleted_count = self.history.cleanup_old(days)
        
        if self.logger:
            self.logger.info(f"{deleted_count} alte Benachrichtigungen gelöscht", 'notification')
        
        return deleted_count

class AlertManager:
    """Manager für spezielle Alerts und Warnungen"""
    
    def __init__(self, notification_system: NotificationSystem, logger=None):
        self.notification_system = notification_system
        self.logger = logger
        
        # Alert-Konfiguration
        self.price_alerts = {}  # Symbol -> Alert-Konfiguration
        self.volume_alerts = {}
        self.portfolio_alerts = {}
        
        # Zustandsverfolgung
        self.last_prices = {}
        self.alert_cooldowns = {}  # Verhindert Spam
        
    def add_price_alert(self, symbol: str, target_price: float, 
                       condition: str = "above", one_time: bool = True):
        """Fügt einen Preisalert hinzu"""
        alert_id = f"{symbol}_{target_price}_{condition}"
        
        self.price_alerts[alert_id] = {
            'symbol': symbol,
            'target_price': target_price,
            'condition': condition,  # 'above', 'below', 'crosses_up', 'crosses_down'
            'one_time': one_time,
            'triggered': False,
            'created_at': datetime.now()
        }
        
        if self.logger:
            self.logger.info(f"Preisalert hinzugefügt: {symbol} {condition} {target_price}", 'alert')
    
    def add_volume_alert(self, symbol: str, volume_threshold: float, timeframe: str = "1h"):
        """Fügt einen Volumen-Alert hinzu"""
        alert_id = f"{symbol}_volume_{volume_threshold}_{timeframe}"
        
        self.volume_alerts[alert_id] = {
            'symbol': symbol,
            'volume_threshold': volume_threshold,
            'timeframe': timeframe,
            'created_at': datetime.now()
        }
    
    def add_portfolio_alert(self, alert_type: str, threshold: float):
        """Fügt einen Portfolio-Alert hinzu"""
        # alert_type: 'total_value', 'daily_pnl', 'drawdown', 'margin_ratio'
        alert_id = f"portfolio_{alert_type}_{threshold}"
        
        self.portfolio_alerts[alert_id] = {
            'type': alert_type,
            'threshold': threshold,
            'created_at': datetime.now()
        }
    
    def check_price_alerts(self, prices: Dict[str, float]):
        """Überprüft alle Preisalerts"""
        current_time = datetime.now()
        
        for alert_id, alert in list(self.price_alerts.items()):
            symbol = alert['symbol']
            
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            target_price = alert['target_price']
            condition = alert['condition']
            
            # Cooldown prüfen (5 Minuten zwischen gleichen Alerts)
            cooldown_key = f"price_{symbol}_{condition}"
            if cooldown_key in self.alert_cooldowns:
                if current_time - self.alert_cooldowns[cooldown_key] < timedelta(minutes=5):
                    continue
            
            triggered = False
            
            if condition == "above" and current_price > target_price:
                triggered = True
            elif condition == "below" and current_price < target_price:
                triggered = True
            elif condition == "crosses_up":
                if symbol in self.last_prices:
                    if self.last_prices[symbol] <= target_price < current_price:
                        triggered = True
            elif condition == "crosses_down":
                if symbol in self.last_prices:
                    if self.last_prices[symbol] >= target_price > current_price:
                        triggered = True
            
            if triggered and not alert['triggered']:
                # Alert senden
                title = f"Preisalert: {symbol}"
                message = f"Der Preis von {symbol} ist {condition} {target_price:.8f}\nAktueller Preis: {current_price:.8f}"
                
                data = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'target_price': target_price,
                    'condition': condition
                }
                
                self.notification_system.notify(
                    title, message, NotificationLevel.HIGH, 'price_alert', data
                )
                
                # Alert als ausgelöst markieren
                if alert['one_time']:
                    alert['triggered'] = True
                
                # Cooldown setzen
                self.alert_cooldowns[cooldown_key] = current_time
                
                if self.logger:
                    self.logger.info(f"Preisalert ausgelöst: {alert_id}", 'alert')
            
            # Aktuellen Preis speichern
            self.last_prices[symbol] = current_price
    
    def check_volume_alerts(self, volume_data: Dict[str, Dict[str, float]]):
        """Überprüft Volumen-Alerts"""
        # volume_data: {symbol: {timeframe: volume}}
        current_time = datetime.now()
        
        for alert_id, alert in self.volume_alerts.items():
            symbol = alert['symbol']
            timeframe = alert['timeframe']
            threshold = alert['volume_threshold']
            
            if symbol not in volume_data or timeframe not in volume_data[symbol]:
                continue
            
            current_volume = volume_data[symbol][timeframe]
            
            # Cooldown prüfen
            cooldown_key = f"volume_{symbol}_{timeframe}"
            if cooldown_key in self.alert_cooldowns:
                if current_time - self.alert_cooldowns[cooldown_key] < timedelta(minutes=15):
                    continue
            
            if current_volume > threshold:
                title = f"Volumen-Alert: {symbol}"
                message = f"Hohes Handelsvolumen bei {symbol}\nVolumen ({timeframe}): {current_volume:.2f}\nSchwellwert: {threshold:.2f}"
                
                data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'current_volume': current_volume,
                    'threshold': threshold
                }
                
                self.notification_system.notify(
                    title, message, NotificationLevel.MEDIUM, 'volume_alert', data
                )
                
                # Cooldown setzen
                self.alert_cooldowns[cooldown_key] = current_time
    
    def check_portfolio_alerts(self, portfolio_data: Dict[str, float]):
        """Überprüft Portfolio-Alerts"""
        current_time = datetime.now()
        
        for alert_id, alert in self.portfolio_alerts.items():
            alert_type = alert['type']
            threshold = alert['threshold']
            
            if alert_type not in portfolio_data:
                continue
            
            current_value = portfolio_data[alert_type]
            
            # Cooldown prüfen
            cooldown_key = f"portfolio_{alert_type}"
            if cooldown_key in self.alert_cooldowns:
                if current_time - self.alert_cooldowns[cooldown_key] < timedelta(minutes=30):
                    continue
            
            triggered = False
            message = ""
            level = NotificationLevel.MEDIUM
            
            if alert_type == 'total_value' and current_value < threshold:
                triggered = True
                message = f"Portfolio-Wert unter Schwellwert!\nAktuell: {current_value:.2f}\nSchwellwert: {threshold:.2f}"
                level = NotificationLevel.HIGH
            
            elif alert_type == 'daily_pnl' and current_value < threshold:
                triggered = True
                message = f"Täglicher P&L unter Schwellwert!\nAktuell: {current_value:.2f}\nSchwellwert: {threshold:.2f}"
                level = NotificationLevel.HIGH
            
            elif alert_type == 'drawdown' and current_value > threshold:
                triggered = True
                message = f"Drawdown über Schwellwert!\nAktuell: {current_value:.2f}%\nSchwellwert: {threshold:.2f}%"
                level = NotificationLevel.CRITICAL
            
            elif alert_type == 'margin_ratio' and current_value > threshold:
                triggered = True
                message = f"Margin-Ratio kritisch!\nAktuell: {current_value:.2f}%\nSchwellwert: {threshold:.2f}%"
                level = NotificationLevel.CRITICAL
            
            if triggered:
                title = f"Portfolio-Alert: {alert_type}"
                
                data = {
                    'alert_type': alert_type,
                    'current_value': current_value,
                    'threshold': threshold
                }
                
                self.notification_system.notify(
                    title, message, level, 'portfolio_alert', data
                )
                
                # Cooldown setzen
                self.alert_cooldowns[cooldown_key] = current_time
    
    def remove_alert(self, alert_id: str) -> bool:
        """Entfernt einen Alert"""
        removed = False
        
        if alert_id in self.price_alerts:
            del self.price_alerts[alert_id]
            removed = True
        elif alert_id in self.volume_alerts:
            del self.volume_alerts[alert_id]
            removed = True
        elif alert_id in self.portfolio_alerts:
            del self.portfolio_alerts[alert_id]
            removed = True
        
        if removed and self.logger:
            self.logger.info(f"Alert entfernt: {alert_id}", 'alert')
        
        return removed
    
    def get_active_alerts(self) -> Dict[str, List]:
        """Gibt alle aktiven Alerts zurück"""
        return {
            'price_alerts': list(self.price_alerts.values()),
            'volume_alerts': list(self.volume_alerts.values()),
            'portfolio_alerts': list(self.portfolio_alerts.values())
        }

class NotificationScheduler:
    """Scheduler für zeitgesteuerte Benachrichtigungen"""
    
    def __init__(self, notification_system: NotificationSystem, logger=None):
        self.notification_system = notification_system
        self.logger = logger
        self.running = False
        self.scheduler_thread = None
        
        # Scheduled Jobs
        self.scheduled_jobs = []
        
        self._setup_default_schedules()
    
    def _setup_default_schedules(self):
        """Richtet Standard-Zeitpläne ein"""
        # Tägliche Zusammenfassung um 18:00
        schedule.every().day.at("18:00").do(self._send_daily_summary)
        
        # Wöchentliche Zusammenfassung am Sonntag um 20:00
        schedule.every().sunday.at("20:00").do(self._send_weekly_summary)
        
        # Monatliche Bereinigung am 1. um 02:00
        schedule.every().month.do(self._monthly_cleanup)
        
        # Stündliche System-Checks
        schedule.every().hour.do(self._hourly_system_check)
    
    def start(self):
        """Startet den Scheduler"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
            self.scheduler_thread.start()
            
            if self.logger:
                self.logger.info("Notification-Scheduler gestartet", 'scheduler')
    
    def stop(self):
        """Stoppt den Scheduler"""
        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
            
        if self.logger:
            self.logger.info("Notification-Scheduler gestoppt", 'scheduler')
    
    def _scheduler_worker(self):
        """Worker-Thread für den Scheduler"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Jede Minute prüfen
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Scheduler-Fehler: {e}", 'scheduler')
                time.sleep(60)
    
    def _send_daily_summary(self):
        """Sendet tägliche Zusammenfassung"""
        try:
            # Hier würden normalerweise die aktuellen Daten abgerufen
            # Für jetzt verwenden wir Dummy-Daten
            summary_data = {
                'trades_count': 0,
                'profit_loss': 0.0,
                'active_positions': 0,
                'best_strategy': 'N/A',
                'portfolio_value': 0.0,
                'notifications_sent': len(self.notification_system.history.get_recent(24))
            }
            
            self.notification_system.send_daily_summary(summary_data)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fehler bei täglicher Zusammenfassung: {e}", 'scheduler')
    
    def _send_weekly_summary(self):
        """Sendet wöchentliche Zusammenfassung"""
        try:
            title = f"Wöchentliche Zusammenfassung - KW {datetime.now().isocalendar()[1]}"
            message = "Hier ist Ihre wöchentliche Trading-Bot Zusammenfassung..."
            
            self.notification_system.notify(
                title, message, NotificationLevel.LOW, 'weekly_summary'
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fehler bei wöchentlicher Zusammenfassung: {e}", 'scheduler')
    
    def _monthly_cleanup(self):
        """Monatliche Bereinigung"""
        try:
            deleted_count = self.notification_system.cleanup_history(30)
            
            if deleted_count > 0:
                self.notification_system.notify(
                    "Datenbereinigung",
                    f"{deleted_count} alte Benachrichtigungen wurden gelöscht.",
                    NotificationLevel.LOW,
                    'maintenance'
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fehler bei monatlicher Bereinigung: {e}", 'scheduler')
    
    def _hourly_system_check(self):
        """Stündlicher System-Check"""
        try:
            # Prüfe Queue-Größe
            queue_size = self.notification_system.queue.size()
            if queue_size > 100:
                self.notification_system.notify(
                    "System-Warnung",
                    f"Notification-Queue ist groß: {queue_size} Nachrichten",
                    NotificationLevel.MEDIUM,
                    'system'
                )
            
            # Prüfe Verbindungen
            channel_status = self.notification_system.test_all_channels()
            failed_channels = [ch for ch, status in channel_status.items() if not status]
            
            if failed_channels:
                self.notification_system.notify(
                    "Verbindungsproblem",
                    f"Fehlgeschlagene Kanäle: {', '.join(failed_channels)}",
                    NotificationLevel.HIGH,
                    'system'
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fehler bei System-Check: {e}", 'scheduler')

# Beispiel-Verwendung und Tests
if __name__ == "__main__":
    import logging
    
    # Logger einrichten
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Konfiguration für Tests
    test_config = {
        'telegram': {
            'enabled': True,
            'bot_token': 'YOUR_BOT_TOKEN',  # Hier Ihren Bot-Token eintragen
            'chat_id': '@joergnij'  # Ihr Telegram-Handle
        },
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email': 'j.nijenhuis43@gmail.com',
            'password': 'YOUR_APP_PASSWORD',  # App-Passwort für Gmail
            'recipient': 'j.nijenhuis43@gmail.com'
        },
        'discord': {
            'enabled': False,  # Discord optional
            'webhook_url': ''
        }
    }
    
    # Benachrichtigungssystem erstellen
    notification_system = NotificationSystem(logger=logger)
    notification_system.config = test_config
    notification_system._init_notifiers()
    
    # System starten
    notification_system.start()
    
    # Alert Manager erstellen
    alert_manager = AlertManager(notification_system, logger)
    
    # Scheduler erstellen
    scheduler = NotificationScheduler(notification_system, logger)
    scheduler.start()
    
    try:
        # Test-Benachrichtigungen
        print("Sende Test-Benachrichtigungen...")
        
        # Verschiedene Nachrichtentypen testen
        notification_system.notify_trade("BUY", "BTCUSDT", 0.001, 45000.0)
        
        notification_system.notify_system_status("Trading Engine", "Running")
        
        notification_system.notify_performance("GridStrategy", {
            'profit': 150.25,
            'trades': 12,
            'win_rate': 75.0
        })
        
        # Preisalert hinzufügen
        alert_manager.add_price_alert("BTCUSDT", 50000.0, "above")
        
        # Test-Preise für Alert-Prüfung
        test_prices = {"BTCUSDT": 51000.0}
        alert_manager.check_price_alerts(test_prices)
        
        # Statistiken anzeigen
        stats = notification_system.get_statistics()
        print(f"\nStatistiken: {stats}")
        
        # Aktive Alerts anzeigen
        alerts = alert_manager.get_active_alerts()
        print(f"\nAktive Alerts: {alerts}")
        
        # Kurz warten für die Verarbeitung
        time.sleep(5)
        
        print("\nTest abgeschlossen. Drücken Sie Ctrl+C zum Beenden...")
        
        # Dauerschleife für kontinuierlichen Betrieb
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nBeende Benachrichtigungssystem...")
        notification_system.stop()
        scheduler.stop()
        print("Beendet.")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
        notification_system.stop()
        scheduler.stop()