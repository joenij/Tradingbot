"""
Trading Bot Data Manager
Verwaltet historische und Echtzeit-Marktdaten von verschiedenen Exchanges
Caching, Speicherung und effiziente Bereitstellung für Analyse und Backtesting
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import time
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
from dataclasses import dataclass
from enum import Enum
import gzip
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import websocket
import schedule
from collections import defaultdict
import os
import sys

# Warnings für pandas unterdrücken
warnings.filterwarnings('ignore', category=FutureWarning)

class DataSource(Enum):
    BINANCE = "binance"
    KUCOIN = "kucoin"
    COMBINED = "combined"

class Timeframe(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m" 
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

@dataclass
class MarketData:
    """Container für Marktdaten"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades_count: int = 0
    source: str = ""

@dataclass
class DataRequest:
    """Container für Datenanfragen"""
    symbol: str
    timeframe: Timeframe
    start_time: datetime
    end_time: datetime
    source: DataSource = DataSource.BINANCE

class DataManager:
    def __init__(self, config_manager, logger, data_dir: str = "data"):
        self.config_manager = config_manager
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Datenbank-Setup
        self.db_path = self.data_dir / "market_data.db"
        self._init_database()
        
        # Cache-Verzeichnisse
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.raw_data_dir = self.data_dir / "raw"
        self.raw_data_dir.mkdir(exist_ok=True)
        
        # In-Memory Cache für schnellen Zugriff
        self.memory_cache = {}
        self.cache_max_size = self.config_manager.get_config_value('DATA', 'cache_size_mb', 500) * 1024 * 1024
        self.current_cache_size = 0
        
        # Thread-Safe Zugriff
        self.lock = threading.Lock()
        
        # API-Endpoints
        self.api_endpoints = {
            DataSource.BINANCE: {
                'base_url': 'https://api.binance.com/api/v3',
                'klines': '/klines',
                'ticker': '/ticker/24hr',
                'exchange_info': '/exchangeInfo',
                'ws_base': 'wss://stream.binance.com:9443/ws/'
            },
            DataSource.KUCOIN: {
                'base_url': 'https://api.kucoin.com/api/v1',
                'klines': '/market/candles',
                'ticker': '/market/stats',
                'symbols': '/symbols',
                'ws_token': '/bullet-public'
            }
        }
        
        # Rate Limiting
        self.rate_limits = {
            DataSource.BINANCE: {'requests_per_minute': 1200, 'weight_per_minute': 6000},
            DataSource.KUCOIN: {'requests_per_minute': 3000, 'concurrent_requests': 20}
        }
        
        self.request_counters = {source: {'count': 0, 'last_reset': time.time()} 
                               for source in DataSource}
        
        # Aktive Verbindungen
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/1.0.0',
            'Content-Type': 'application/json'
        })
        
        # Thread Pool für parallele Downloads
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # WebSocket Verbindungen für Live-Daten
        self.ws_connections = {}
        self.live_data_callbacks = defaultdict(list)
        self.ws_running = False
        
        # Automatische Updates
        self.auto_update_enabled = self.config_manager.get_config_value('DATA', 'auto_update', True)
        self.update_interval = self.config_manager.get_config_value('DATA', 'update_interval_minutes', 5)
        
        # Data Quality Monitoring
        self.data_quality_metrics = {}
        
        self.logger.info("DataManager initialisiert", 'data')
        
        # Startvalidierung
        self._validate_setup()
        
        # Starte automatische Updates wenn aktiviert
        if self.auto_update_enabled:
            self._start_auto_updates()
    
    def _init_database(self):
        """Initialisiert die SQLite-Datenbank für Marktdaten"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Haupttabelle für OHLCV-Daten
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        quote_volume REAL DEFAULT 0,
                        trades_count INTEGER DEFAULT 0,
                        source TEXT NOT NULL,
                        created_at INTEGER DEFAULT (strftime('%s', 'now')),
                        UNIQUE(symbol, timeframe, timestamp, source)
                    )
                """)
                
                # Indizes für bessere Performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                    ON market_data(symbol, timeframe, timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_symbol_source 
                    ON market_data(symbol, source)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON market_data(timestamp)
                """)
                
                # Metadaten-Tabelle
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_metadata (
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        source TEXT NOT NULL,
                        first_timestamp INTEGER,
                        last_timestamp INTEGER,
                        total_records INTEGER DEFAULT 0,
                        last_updated INTEGER DEFAULT (strftime('%s', 'now')),
                        data_quality_score REAL DEFAULT 1.0,
                        gaps_count INTEGER DEFAULT 0,
                        PRIMARY KEY(symbol, timeframe, source)
                    )
                """)
                
                # Symbol-Info Tabelle
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS symbol_info (
                        symbol TEXT NOT NULL,
                        source TEXT NOT NULL,
                        base_asset TEXT,
                        quote_asset TEXT,
                        price_precision INTEGER,
                        quantity_precision INTEGER,
                        min_order_size REAL,
                        status TEXT DEFAULT 'TRADING',
                        last_updated INTEGER DEFAULT (strftime('%s', 'now')),
                        PRIMARY KEY(symbol, source)
                    )
                """)
                
                # Live-Daten Tabelle für aktuelle Preise
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS live_data (
                        symbol TEXT NOT NULL,
                        source TEXT NOT NULL,
                        price REAL NOT NULL,
                        volume_24h REAL DEFAULT 0,
                        change_24h REAL DEFAULT 0,
                        high_24h REAL DEFAULT 0,
                        low_24h REAL DEFAULT 0,
                        timestamp INTEGER DEFAULT (strftime('%s', 'now')),
                        PRIMARY KEY(symbol, source)
                    )
                """)
                
                # Data Quality Logs
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        source TEXT NOT NULL,
                        issue_type TEXT NOT NULL,
                        issue_description TEXT,
                        timestamp INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                """)
                
                conn.commit()
                
            self.logger.info("Datenbank erfolgreich initialisiert", 'data')
            
        except Exception as e:
            self.logger.error(f"Fehler bei Datenbank-Initialisierung: {e}", 'data')
            raise
    
    def _validate_setup(self):
        """Validiert die DataManager-Konfiguration"""
        errors = []
        
        # Prüfe Verzeichnisse
        if not self.data_dir.exists():
            errors.append("Data-Verzeichnis konnte nicht erstellt werden")
        
        # Prüfe Datenbank-Verbindung
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
        except Exception as e:
            errors.append(f"Datenbank-Verbindung fehlgeschlagen: {e}")
        
        # Prüfe Internet-Verbindung
        try:
            response = self.session.get('https://api.binance.com/api/v3/time', timeout=5)
            if response.status_code != 200:
                errors.append("Binance API nicht erreichbar")
        except Exception as e:
            errors.append(f"Internet-Verbindung zu Binance fehlgeschlagen: {e}")
        
        if errors:
            for error in errors:
                self.logger.error(error, 'data')
            raise RuntimeError("DataManager-Validierung fehlgeschlagen")
        
        self.logger.info("DataManager-Setup erfolgreich validiert", 'data')
    
    def get_historical_data(self, symbol: str, timeframe: Timeframe, 
                           start_date: datetime, end_date: datetime = None,
                           source: DataSource = DataSource.BINANCE,
                           force_download: bool = False) -> pd.DataFrame:
        """
        Lädt historische Daten für ein Symbol und Timeframe
        Prüft zuerst Cache/DB, lädt bei Bedarf von Exchange
        """
        if end_date is None:
            end_date = datetime.now()
        
        timer_id = self.logger.start_performance_timer(
            'get_historical_data',
            f"{symbol}_{timeframe.value}_{source.value}"
        )
        
        try:
            # Cache-Key generieren
            cache_key = self._generate_cache_key(symbol, timeframe, start_date, end_date, source)
            
            # Prüfe Memory Cache
            if not force_download and cache_key in self.memory_cache:
                self.logger.debug(f"Daten aus Memory Cache geladen: {symbol} {timeframe.value}", 'data')
                return self.memory_cache[cache_key].copy()
            
            # Prüfe Datenbankabdeckung
            existing_data = self._get_data_from_db(symbol, timeframe, start_date, end_date, source)
            
            if not force_download and not existing_data.empty:
                coverage = self._check_data_coverage(existing_data, start_date, end_date, timeframe)
                
                if coverage >= 0.95:  # 95% Abdeckung ist ausreichend
                    self.logger.debug(f"Vollständige Daten aus Datenbank: {symbol} {timeframe.value}", 'data')
                    self._add_to_memory_cache(cache_key, existing_data)
                    return existing_data
            
            # Lade fehlende Daten von Exchange
            self.logger.info(f"Lade historische Daten: {symbol} {timeframe.value} von {start_date} bis {end_date}", 'data')
            
            new_data = self._download_historical_data(symbol, timeframe, start_date, end_date, source)
            
            if not new_data.empty:
                # Validiere Datenqualität
                self._validate_data_quality(new_data, symbol, timeframe, source)
                
                # Speichere in Datenbank
                self._save_data_to_db(new_data, symbol, timeframe, source)
                
                # Kombiniere mit existierenden Daten
                if not existing_data.empty:
                    combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['timestamp'])
                    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                else:
                    combined_data = new_data
                
                # Cache aktualisieren
                self._add_to_memory_cache(cache_key, combined_data)
                
                self.logger.info(f"Historische Daten erfolgreich geladen: {len(combined_data)} Datensätze", 'data')
                return combined_data
            else:
                self.logger.warning(f"Keine Daten für {symbol} {timeframe.value} verfügbar", 'data')
                return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Fehler beim Laden historischer Daten: {e}", 'data')
            return pd.DataFrame()
        
        finally:
            self.logger.end_performance_timer(timer_id)
    
    def get_initial_training_data(self, symbols: List[str], 
                                 timeframes: List[Timeframe] = None,
                                 start_date: datetime = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Lädt initiale Trainingsdaten ab Januar 2017 für alle Symbole und Timeframes
        Für ML-Training und Backtesting
        """
        if start_date is None:
            start_date = datetime(2017, 1, 1)
        
        if timeframes is None:
            timeframes = [Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAY_1]
        
        end_date = datetime.now()
        
        self.logger.info(f"Lade initiale Trainingsdaten für {len(symbols)} Symbole ab {start_date}", 'data')
        
        training_data = {}
        
        for symbol in symbols:
            training_data[symbol] = {}
            
            for timeframe in timeframes:
                try:
                    self.logger.info(f"Lade {symbol} {timeframe.value} Daten...", 'data')
                    
                    # Versuche beide Datenquellen
                    data_binance = self.get_historical_data(
                        symbol, timeframe, start_date, end_date, DataSource.BINANCE
                    )
                    
                    if not data_binance.empty:
                        training_data[symbol][timeframe.value] = data_binance
                        self.logger.info(f"Binance Daten geladen: {len(data_binance)} Datensätze", 'data')
                    else:
                        # Fallback zu KuCoin
                        data_kucoin = self.get_historical_data(
                            symbol, timeframe, start_date, end_date, DataSource.KUCOIN
                        )
                        
                        if not data_kucoin.empty:
                            training_data[symbol][timeframe.value] = data_kucoin
                            self.logger.info(f"KuCoin Daten geladen: {len(data_kucoin)} Datensätze", 'data')
                        else:
                            self.logger.warning(f"Keine Daten für {symbol} {timeframe.value} verfügbar", 'data')
                    
                    # Kurze Pause um APIs zu schonen
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Fehler beim Laden von {symbol} {timeframe.value}: {e}", 'data')
                    continue
        
        self.logger.info("Initiale Trainingsdaten erfolgreich geladen", 'data')
        return training_data
    
    def _download_historical_data(self, symbol: str, timeframe: Timeframe,
                                 start_date: datetime, end_date: datetime,
                                 source: DataSource) -> pd.DataFrame:
        """Lädt historische Daten von der Exchange"""
        
        if source == DataSource.BINANCE:
            return self._download_binance_data(symbol, timeframe, start_date, end_date)
        elif source == DataSource.KUCOIN:
            return self._download_kucoin_data(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"Unbekannte Datenquelle: {source}")
    
    def _download_binance_data(self, symbol: str, timeframe: Timeframe,
                              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Lädt Daten von Binance API"""
        
        url = f"{self.api_endpoints[DataSource.BINANCE]['base_url']}{self.api_endpoints[DataSource.BINANCE]['klines']}"
        
        all_data = []
        current_start = start_date
        
        # Binance hat ein Limit von 1000 Kerzen pro Anfrage
        max_candles = 1000
        
        while current_start < end_date:
            # Rate Limiting prüfen
            self._check_rate_limit(DataSource.BINANCE)
            
            # Berechne End-Zeit für diese Anfrage
            if timeframe == Timeframe.MINUTE_1:
                request_end = current_start + timedelta(minutes=max_candles)
            elif timeframe == Timeframe.MINUTE_5:
                request_end = current_start + timedelta(minutes=max_candles * 5)
            elif timeframe == Timeframe.MINUTE_15:
                request_end = current_start + timedelta(minutes=max_candles * 15)
            elif timeframe == Timeframe.MINUTE_30:
                request_end = current_start + timedelta(minutes=max_candles * 30)
            elif timeframe == Timeframe.HOUR_1:
                request_end = current_start + timedelta(hours=max_candles)
            elif timeframe == Timeframe.HOUR_4:
                request_end = current_start + timedelta(hours=max_candles * 4)
            elif timeframe == Timeframe.HOUR_12:
                request_end = current_start + timedelta(hours=max_candles * 12)
            elif timeframe == Timeframe.DAY_1:
                request_end = current_start + timedelta(days=max_candles)
            else:
                request_end = current_start + timedelta(days=max_candles)
            
            request_end = min(request_end, end_date)
            
            params = {
                'symbol': symbol,
                'interval': timeframe.value,
                'startTime': int(current_start.timestamp() * 1000),
                'endTime': int(request_end.timestamp() * 1000),
                'limit': max_candles
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data:
                    # Konvertiere zu DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 
                        'volume', 'close_time', 'quote_volume', 'trades_count',
                        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    # Datentypen konvertieren
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['trades_count'] = pd.to_numeric(df['trades_count'], errors='coerce')
                    
                    # Nur benötigte Spalten behalten
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades_count']]
                    df['source'] = DataSource.BINANCE.value
                    
                    all_data.append(df)
                    
                    self.logger.debug(f"Binance Daten geladen: {len(df)} Kerzen von {current_start} bis {request_end}", 'data')
                
                # Nächste Iteration
                current_start = request_end + timedelta(seconds=1)
                
                # Kurze Pause um Rate Limits zu respektieren
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Fehler beim Laden von Binance Daten: {e}", 'data')
                break
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            return result
        else:
            return pd.DataFrame()
    
    def _download_kucoin_data(self, symbol: str, timeframe: Timeframe,
                             start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Lädt Daten von KuCoin API"""
        
        url = f"{self.api_endpoints[DataSource.KUCOIN]['base_url']}{self.api_endpoints[DataSource.KUCOIN]['klines']}"
        
        # KuCoin verwendet unterschiedliche Timeframe-Bezeichnungen
        kucoin_timeframes = {
            Timeframe.MINUTE_1: '1min',
            Timeframe.MINUTE_5: '5min',
            Timeframe.MINUTE_15: '15min',
            Timeframe.MINUTE_30: '30min',
            Timeframe.HOUR_1: '1hour',
            Timeframe.HOUR_4: '4hour',
            Timeframe.HOUR_12: '12hour',
            Timeframe.DAY_1: '1day',
            Timeframe.WEEK_1: '1week'
        }
        
        kucoin_interval = kucoin_timeframes.get(timeframe, '1hour')
        
        all_data = []
        current_start = start_date
        
        # KuCoin hat ein Limit von 1500 Kerzen pro Anfrage
        max_candles = 1500
        
        while current_start < end_date:
            # Rate Limiting prüfen
            self._check_rate_limit(DataSource.KUCOIN)
            
            # Berechne End-Zeit basierend auf Timeframe
            if timeframe == Timeframe.MINUTE_1:
                request_end = current_start + timedelta(minutes=max_candles)
            elif timeframe == Timeframe.MINUTE_5:
                request_end = current_start + timedelta(minutes=max_candles * 5)
            elif timeframe == Timeframe.MINUTE_15:
                request_end = current_start + timedelta(minutes=max_candles * 15)
            elif timeframe == Timeframe.MINUTE_30:
                request_end = current_start + timedelta(minutes=max_candles * 30)
            elif timeframe == Timeframe.HOUR_1:
                request_end = current_start + timedelta(hours=max_candles)
            elif timeframe == Timeframe.HOUR_4:
                request_end = current_start + timedelta(hours=max_candles * 4)
            elif timeframe == Timeframe.HOUR_12:
                request_end = current_start + timedelta(hours=max_candles * 12)
            elif timeframe == Timeframe.DAY_1:
                request_end = current_start + timedelta(days=max_candles)
            else:
                request_end = current_start + timedelta(days=max_candles)
            
            request_end = min(request_end, end_date)
            
            params = {
                'symbol': symbol,
                'type': kucoin_interval,
                'startAt': int(current_start.timestamp()),
                'endAt': int(request_end.timestamp())
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('code') == '200000' and result.get('data'):
                    data = result['data']
                    
                    # KuCoin Datenformat: [timestamp, open, close, high, low, volume, quote_volume]
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'close', 'high', 'low', 'volume', 'quote_volume'
                    ])
                    
                    # Datentypen konvertieren
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Spalten neu ordnen
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
                    df['trades_count'] = 0  # KuCoin liefert diese Info nicht in Kerzen
                    df['source'] = DataSource.KUCOIN.value
                    
                    all_data.append(df)
                    
                    self.logger.debug(f"KuCoin Daten geladen: {len(df)} Kerzen von {current_start} bis {request_end}", 'data')
                
                # Nächste Iteration
                current_start = request_end + timedelta(seconds=1)
                
                # Kurze Pause um Rate Limits zu respektieren
                time.sleep(0.2)
                
            except Exception as e:
                self.logger.error(f"Fehler beim Laden von KuCoin Daten: {e}", 'data')
                break
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            return result
        else:
            return pd.DataFrame()
    
    def _check_rate_limit(self, source: DataSource):
        """Prüft und respektiert Rate Limits"""
        current_time = time.time()
        counter = self.request_counters[source]
        
        # Reset Counter nach einer Minute
        if current_time - counter['last_reset'] >= 60:
            counter['count'] = 0
            counter['last_reset'] = current_time
        
        # Prüfe Limits
        if source == DataSource.BINANCE:
            if counter['count'] >= self.rate_limits[source]['requests_per_minute']:
                wait_time = 60 - (current_time - counter['last_reset'])
                if wait_time > 0:
                    self.logger.warning(f"Binance Rate Limit erreicht. Warte {wait_time:.1f} Sekunden", 'data')
                    time.sleep(wait_time)
                    counter['count'] = 0
                    counter['last_reset'] = time.time()
        
        elif source == DataSource.KUCOIN:
            if counter['count'] >= self.rate_limits[source]['requests_per_minute']:
                wait_time = 60 - (current_time - counter['last_reset'])
                if wait_time > 0:
                    self.logger.warning(f"KuCoin Rate Limit erreicht. Warte {wait_time:.1f} Sekunden", 'data')
                    time.sleep(wait_time)
                    counter['count'] = 0
                    counter['last_reset'] = time.time()
        
        counter['count'] += 1
    
    def _get_data_from_db(self, symbol: str, timeframe: Timeframe,
                         start_date: datetime, end_date: datetime,
                         source: DataSource) -> pd.DataFrame:
        """Lädt Daten aus der lokalen Datenbank"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume, quote_volume, trades_count, source
                    FROM market_data
                    WHERE symbol = ? AND timeframe = ? AND source = ?
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                """
                
                params = (
                    symbol, 
                    timeframe.value, 
                    source.value,
                    int(start_date.timestamp()),
                    int(end_date.timestamp())
                )
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Fehler beim Laden aus Datenbank: {e}", 'data')
            return pd.DataFrame()
    
   def _save_data_to_db(self, data: pd.DataFrame, symbol: str, 
                        timeframe: Timeframe, source: DataSource):
        """Speichert Daten in die lokale Datenbank"""
        
        if data.empty:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Daten für Datenbank vorbereiten
                db_data = data.copy()
                db_data['symbol'] = symbol
                db_data['timeframe'] = timeframe.value
                db_data['timestamp'] = db_data['timestamp'].astype('int64') // 10**9  # Konvertiere zu Unix timestamp
                
                # Spalten in richtiger Reihenfolge
                columns = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 
                          'close', 'volume', 'quote_volume', 'trades_count', 'source']
                db_data = db_data[columns]
                
                # Verwende INSERT OR REPLACE um Duplikate zu vermeiden
                db_data.to_sql('market_data', conn, if_exists='append', index=False, method='multi')
                
                # Metadaten aktualisieren
                self._update_metadata(conn, symbol, timeframe, source, data)
                
                conn.commit()
                
            self.logger.debug(f"Daten in DB gespeichert: {len(data)} Datensätze für {symbol} {timeframe.value}", 'data')
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern in DB: {e}", 'data')
    
    def _update_metadata(self, conn, symbol: str, timeframe: Timeframe, 
                        source: DataSource, data: pd.DataFrame):
        """Aktualisiert Metadaten für Symbol/Timeframe/Source Kombination"""
        
        if data.empty:
            return
        
        cursor = conn.cursor()
        
        first_timestamp = int(data['timestamp'].min().timestamp())
        last_timestamp = int(data['timestamp'].max().timestamp())
        total_records = len(data)
        
        # Data Quality Score berechnen
        quality_score = self._calculate_data_quality_score(data)
        
        # Gaps zählen
        gaps_count = self._count_data_gaps(data, timeframe)
        
        cursor.execute("""
            INSERT OR REPLACE INTO data_metadata 
            (symbol, timeframe, source, first_timestamp, last_timestamp, 
             total_records, last_updated, data_quality_score, gaps_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, timeframe.value, source.value,
            first_timestamp, last_timestamp, total_records,
            int(time.time()), quality_score, gaps_count
        ))
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Berechnet einen Data Quality Score zwischen 0 und 1"""
        
        if data.empty:
            return 0.0
        
        score = 1.0
        
        # Prüfe auf fehlende Werte
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 0.3
        
        # Prüfe auf unrealistische Preise (0 oder negative)
        price_columns = ['open', 'high', 'low', 'close']
        invalid_prices = 0
        for col in price_columns:
            if col in data.columns:
                invalid_prices += (data[col] <= 0).sum()
        
        invalid_ratio = invalid_prices / (len(data) * len(price_columns))
        score -= invalid_ratio * 0.4
        
        # Prüfe OHLC Konsistenz
        if all(col in data.columns for col in price_columns):
            # High sollte höchster Wert sein, Low niedrigster
            consistency_issues = 0
            consistency_issues += (data['high'] < data['open']).sum()
            consistency_issues += (data['high'] < data['close']).sum()
            consistency_issues += (data['low'] > data['open']).sum()
            consistency_issues += (data['low'] > data['close']).sum()
            
            consistency_ratio = consistency_issues / (len(data) * 4)
            score -= consistency_ratio * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _count_data_gaps(self, data: pd.DataFrame, timeframe: Timeframe) -> int:
        """Zählt Datenlücken basierend auf erwarteter Zeitintervalle"""
        
        if len(data) < 2:
            return 0
        
        # Erwartete Zeitdifferenz basierend auf Timeframe
        expected_diff = {
            Timeframe.MINUTE_1: timedelta(minutes=1),
            Timeframe.MINUTE_5: timedelta(minutes=5),
            Timeframe.MINUTE_15: timedelta(minutes=15),
            Timeframe.MINUTE_30: timedelta(minutes=30),
            Timeframe.HOUR_1: timedelta(hours=1),
            Timeframe.HOUR_4: timedelta(hours=4),
            Timeframe.HOUR_12: timedelta(hours=12),
            Timeframe.DAY_1: timedelta(days=1),
            Timeframe.WEEK_1: timedelta(weeks=1)
        }
        
        expected_delta = expected_diff.get(timeframe, timedelta(hours=1))
        
        # Berechne tatsächliche Zeitdifferenzen
        time_diffs = data['timestamp'].diff().dropna()
        
        # Zähle Gaps (Differenzen größer als 1.5x erwartete Differenz)
        tolerance = expected_delta * 1.5
        gaps = (time_diffs > tolerance).sum()
        
        return gaps
    
    def _check_data_coverage(self, data: pd.DataFrame, start_date: datetime, 
                           end_date: datetime, timeframe: Timeframe) -> float:
        """Prüft die Datenabdeckung für einen Zeitraum"""
        
        if data.empty:
            return 0.0
        
        # Erwartete Anzahl Datenpunkte
        time_diff = end_date - start_date
        
        expected_points = {
            Timeframe.MINUTE_1: int(time_diff.total_seconds() / 60),
            Timeframe.MINUTE_5: int(time_diff.total_seconds() / 300),
            Timeframe.MINUTE_15: int(time_diff.total_seconds() / 900),
            Timeframe.MINUTE_30: int(time_diff.total_seconds() / 1800),
            Timeframe.HOUR_1: int(time_diff.total_seconds() / 3600),
            Timeframe.HOUR_4: int(time_diff.total_seconds() / 14400),
            Timeframe.HOUR_12: int(time_diff.total_seconds() / 43200),
            Timeframe.DAY_1: time_diff.days,
            Timeframe.WEEK_1: int(time_diff.days / 7)
        }
        
        expected = expected_points.get(timeframe, int(time_diff.total_seconds() / 3600))
        actual = len(data)
        
        return min(1.0, actual / max(1, expected))
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str, 
                             timeframe: Timeframe, source: DataSource):
        """Validiert Datenqualität und loggt Probleme"""
        
        issues = []
        
        if data.empty:
            issues.append("Keine Daten erhalten")
        else:
            # Prüfe auf fehlende Werte
            missing = data.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    issues.append(f"Fehlende Werte in {col}: {count}")
            
            # Prüfe auf unrealistische Preise
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    invalid = (data[col] <= 0).sum()
                    if invalid > 0:
                        issues.append(f"Ungültige Preise in {col}: {invalid}")
            
            # Prüfe Volume
            if 'volume' in data.columns:
                zero_volume = (data['volume'] == 0).sum()
                if zero_volume > len(data) * 0.1:  # Mehr als 10% Zero Volume
                    issues.append(f"Verdächtig viele Zero-Volume Perioden: {zero_volume}")
        
        # Logge Probleme
        if issues:
            for issue in issues:
                self.logger.warning(f"Datenqualitätsproblem {symbol} {timeframe.value}: {issue}", 'data')
                
                # In DB loggen
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO data_quality_log 
                            (symbol, timeframe, source, issue_type, issue_description)
                            VALUES (?, ?, ?, ?, ?)
                        """, (symbol, timeframe.value, source.value, "DATA_QUALITY", issue))
                        conn.commit()
                except Exception as e:
                    self.logger.error(f"Fehler beim Loggen von Datenqualitätsproblemen: {e}", 'data')
    
    def _generate_cache_key(self, symbol: str, timeframe: Timeframe, 
                           start_date: datetime, end_date: datetime, 
                           source: DataSource) -> str:
        """Generiert einen eindeutigen Cache-Key"""
        
        key_data = f"{symbol}_{timeframe.value}_{start_date.isoformat()}_{end_date.isoformat()}_{source.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _add_to_memory_cache(self, key: str, data: pd.DataFrame):
        """Fügt Daten zum Memory Cache hinzu"""
        
        with self.lock:
            # Schätze Speicherbedarf
            data_size = data.memory_usage(deep=True).sum()
            
            # Prüfe Cache-Größe
            while self.current_cache_size + data_size > self.cache_max_size and self.memory_cache:
                # Entferne ältesten Eintrag (FIFO)
                oldest_key = next(iter(self.memory_cache))
                oldest_data = self.memory_cache.pop(oldest_key)
                self.current_cache_size -= oldest_data.memory_usage(deep=True).sum()
            
            # Füge neue Daten hinzu
            self.memory_cache[key] = data.copy()
            self.current_cache_size += data_size
    
    def get_available_symbols(self, source: DataSource = DataSource.BINANCE) -> List[str]:
        """Lädt verfügbare Trading-Symbole von der Exchange"""
        
        try:
            if source == DataSource.BINANCE:
                url = f"{self.api_endpoints[source]['base_url']}{self.api_endpoints[source]['exchange_info']}"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                symbols = []
                
                for symbol_info in data.get('symbols', []):
                    if symbol_info.get('status') == 'TRADING':
                        symbols.append(symbol_info['symbol'])
                
                self.logger.info(f"Binance Symbole geladen: {len(symbols)}", 'data')
                return symbols
            
            elif source == DataSource.KUCOIN:
                url = f"{self.api_endpoints[source]['base_url']}{self.api_endpoints[source]['symbols']}"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                symbols = []
                
                if data.get('code') == '200000':
                    for symbol_info in data.get('data', []):
                        if symbol_info.get('enableTrading'):
                            symbols.append(symbol_info['symbol'])
                
                self.logger.info(f"KuCoin Symbole geladen: {len(symbols)}", 'data')
                return symbols
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Symbole von {source.value}: {e}", 'data')
            return []
    
    def get_live_price(self, symbol: str, source: DataSource = DataSource.BINANCE) -> Optional[float]:
        """Lädt aktuellen Preis für ein Symbol"""
        
        try:
            if source == DataSource.BINANCE:
                url = f"{self.api_endpoints[source]['base_url']}{self.api_endpoints[source]['ticker']}"
                params = {'symbol': symbol}
                
                response = self.session.get(url, params=params, timeout=5)
                response.raise_for_status()
                
                data = response.json()
                return float(data['lastPrice'])
            
            elif source == DataSource.KUCOIN:
                url = f"{self.api_endpoints[source]['base_url']}{self.api_endpoints[source]['ticker']}"
                params = {'symbol': symbol}
                
                response = self.session.get(url, params=params, timeout=5)
                response.raise_for_status()
                
                data = response.json()
                if data.get('code') == '200000':
                    return float(data['data']['last'])
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Live-Preises für {symbol}: {e}", 'data')
            return None
    
    def get_24h_ticker(self, symbol: str, source: DataSource = DataSource.BINANCE) -> Optional[Dict]:
        """Lädt 24h Ticker-Daten für ein Symbol"""
        
        try:
            if source == DataSource.BINANCE:
                url = f"{self.api_endpoints[source]['base_url']}{self.api_endpoints[source]['ticker']}"
                params = {'symbol': symbol}
                
                response = self.session.get(url, params=params, timeout=5)
                response.raise_for_status()
                
                data = response.json()
                return {
                    'symbol': data['symbol'],
                    'price': float(data['lastPrice']),
                    'change': float(data['priceChange']),
                    'change_percent': float(data['priceChangePercent']),
                    'high': float(data['highPrice']),
                    'low': float(data['lowPrice']),
                    'volume': float(data['volume']),
                    'quote_volume': float(data['quoteVolume']),
                    'trades_count': int(data['count'])
                }
            
            elif source == DataSource.KUCOIN:
                url = f"{self.api_endpoints[source]['base_url']}{self.api_endpoints[source]['ticker']}"
                params = {'symbol': symbol}
                
                response = self.session.get(url, params=params, timeout=5)
                response.raise_for_status()
                
                data = response.json()
                if data.get('code') == '200000':
                    ticker_data = data['data']
                    return {
                        'symbol': ticker_data['symbol'],
                        'price': float(ticker_data['last']),
                        'change': float(ticker_data['changePrice'] or 0),
                        'change_percent': float(ticker_data['changeRate'] or 0) * 100,
                        'high': float(ticker_data['high']),
                        'low': float(ticker_data['low']),
                        'volume': float(ticker_data['vol']),
                        'quote_volume': float(ticker_data['volValue']),
                        'trades_count': 0  # KuCoin liefert diese Info nicht
                    }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Ticker-Daten für {symbol}: {e}", 'data')
            return None
    
    def update_live_data(self, symbol: str, source: DataSource = DataSource.BINANCE):
        """Aktualisiert Live-Daten in der Datenbank"""
        
        ticker = self.get_24h_ticker(symbol, source)
        if not ticker:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO live_data 
                    (symbol, source, price, volume_24h, change_24h, high_24h, low_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, source.value, ticker['price'], ticker['volume'],
                    ticker['change_percent'], ticker['high'], ticker['low']
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der Live-Daten: {e}", 'data')
    
    def start_live_data_stream(self, symbols: List[str], 
                              callback: callable = None,
                              source: DataSource = DataSource.BINANCE):
        """Startet WebSocket-Stream für Live-Daten"""
        
        if source == DataSource.BINANCE:
            self._start_binance_websocket(symbols, callback)
        elif source == DataSource.KUCOIN:
            self._start_kucoin_websocket(symbols, callback)
    
    def _start_binance_websocket(self, symbols: List[str], callback: callable = None):
        """Startet Binance WebSocket Stream"""
        
        if self.ws_running:
            self.stop_live_data_stream()
        
        # Erstelle Stream Namen
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        stream_url = f"{self.api_endpoints[DataSource.BINANCE]['ws_base']}{'@'.join(streams)}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Verarbeite Ticker-Daten
                if 'e' in data and data['e'] == '24hrTicker':
                    ticker = {
                        'symbol': data['s'],
                        'price': float(data['c']),
                        'change_percent': float(data['P']),
                        'volume': float(data['v']),
                        'high': float(data['h']),
                        'low': float(data['l']),
                        'timestamp': datetime.now()
                    }
                    
                    # Callback aufrufen wenn definiert
                    if callback:
                        callback(ticker)
                    
                    # In DB speichern
                    self.update_live_data(ticker['symbol'], DataSource.BINANCE)
                    
            except Exception as e:
                self.logger.error(f"Fehler bei WebSocket Nachricht: {e}", 'data')
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket Fehler: {error}", 'data')
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info("WebSocket Verbindung geschlossen", 'data')
            self.ws_running = False
        
        def on_open(ws):
            self.logger.info(f"WebSocket Verbindung geöffnet für {len(symbols)} Symbole", 'data')
            self.ws_running = True
        
        # WebSocket in separatem Thread starten
        def run_websocket():
            ws = websocket.WebSocketApp(
                stream_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        self.ws_connections[DataSource.BINANCE] = ws_thread
    
    def stop_live_data_stream(self):
        """Stoppt alle WebSocket-Streams"""
        
        self.ws_running = False
        
        for source, thread in self.ws_connections.items():
            if thread and thread.is_alive():
                # WebSocket schließen (Thread wird automatisch beendet)
                self.logger.info(f"Stoppe WebSocket Stream für {source.value}", 'data')
        
        self.ws_connections.clear()
    
    def _start_auto_updates(self):
        """Startet automatische Datenaktualisierung"""
        
        def update_job():
            self.logger.info("Starte automatische Datenaktualisierung", 'data')
            
            # Aktualisiere wichtigste Symbole
            important_symbols = self.config_manager.get_config_value('TRADING', 'symbols', ['BTCUSDT', 'ETHUSDT'])
            
            for symbol in important_symbols:
                try:
                    # Aktualisiere letzte 24h Daten
                    end_date = datetime.now()
                    start_date = end_date - timedelta(hours=24)
                    
                    self.get_historical_data(
                        symbol, Timeframe.HOUR_1, start_date, end_date,
                        DataSource.BINANCE, force_download=False
                    )
                    
                    # Kurze Pause zwischen Symbolen
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Fehler bei automatischer Aktualisierung für {symbol}: {e}", 'data')
        
        # Schedule Updates
        schedule.every(self.update_interval).minutes.do(update_job)
        
        def run_scheduler():
            while self.auto_update_enabled:
                schedule.run_pending()
                time.sleep(60)  # Prüfe jede Minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        self.logger.info(f"Automatische Updates alle {self.update_interval} Minuten gestartet", 'data')
    
    def get_data_statistics(self) -> Dict:
        """Liefert Statistiken über gespeicherte Daten"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Grundstatistiken
                cursor.execute("SELECT COUNT(*) FROM market_data")
                total_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
                unique_symbols = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT source) FROM market_data")
                unique_sources = cursor.fetchone()[0]
                
                # Speicherbedarf
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # Neueste Daten
                cursor.execute("""
                    SELECT symbol, timeframe, source, MAX(timestamp) as latest
                    FROM market_data
                    GROUP BY symbol, timeframe, source
                    ORDER BY latest DESC
                    LIMIT 10
                """)
                latest_data = cursor.fetchall()
                
                # Data Quality
                cursor.execute("SELECT AVG(data_quality_score) FROM data_metadata")
                avg_quality = cursor.fetchone()[0] or 0
                
                return {
                    'total_records': total_records,
                    'unique_symbols': unique_symbols,
                    'unique_sources': unique_sources,
                    'database_size_mb': db_size / (1024 * 1024),
                    'memory_cache_size_mb': self.current_cache_size / (1024 * 1024),
                    'cache_entries': len(self.memory_cache),
                    'average_data_quality': avg_quality,
                    'latest_data': latest_data,
                    'websocket_active': self.ws_running
                }
                
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Datenstatistiken: {e}", 'data')
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Bereinigt alte Daten aus der Datenbank"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_timestamp = int(cutoff_date.timestamp())
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Zähle zu löschende Datensätze
                cursor.execute("SELECT COUNT(*) FROM market_data WHERE timestamp < ?", (cutoff_timestamp,))
                records_to_delete = cursor.fetchone()[0]
                
                if records_to_delete > 0:
                    # Lösche alte Daten
                    cursor.execute("DELETE FROM market_data WHERE timestamp < ?", (cutoff_timestamp,))
                    
                    # Aktualisiere Metadaten
                    cursor.execute("DELETE FROM data_metadata WHERE last_updated < ?", (cutoff_timestamp,))
                    
                    # Bereinige Quality Logs
                    cursor.execute("DELETE FROM data_quality_log WHERE timestamp < ?", (cutoff_timestamp,))
                    
                    # Vacuum für Speicherplatz-Rückgewinnung
                    cursor.execute("VACUUM")
                    
                    conn.commit()
                    
                    self.logger.info(f"Datenbereinigung abgeschlossen: {records_to_delete} alte Datensätze gelöscht", 'data')
                else:
                    self.logger.info("Keine alten Daten zu bereinigen gefunden", 'data')
                    
        except Exception as e:
            self.logger.error(f"Fehler bei Datenbereinigung: {e}", 'data')
    
    def export_data(self, symbol: str, timeframe: Timeframe, 
                   start_date: datetime, end_date: datetime,
                   output_file: str, format: str = 'csv'):
        """Exportiert Daten in verschiedene Formate"""
        
        data = self.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            self.logger.warning(f"Keine Daten zum Exportieren für {symbol} {timeframe.value}", 'data')
            return False
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                data.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                data.to_json(output_path, orient='records', date_format='iso')
            elif format.lower() == 'parquet':
                data.to_parquet(output_path, index=False)
            elif format.lower() == 'pickle':
                data.to_pickle(output_path)
            else:
                raise ValueError(f"Unbekanntes Export-Format: {format}")
            
            self.logger.info(f"Daten exportiert: {len(data)} Datensätze nach {output_path}", 'data')
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Exportieren der Daten: {e}", 'data')
            return False
    
    def shutdown(self):
        """Sauberes Shutdown des DataManagers"""
        
        self.logger.info("Shutdown DataManager...", 'data')
        
        # Stoppe WebSocket Streams
        self.stop_live_data_stream()
        
        # Stoppe automatische Updates
        self.auto_update_enabled = False
        
        # Schließe Thread Pool
        self.thread_pool.shutdown(wait=True)
        
        # Schließe Session
        self.session.close()
        
        # Leere Memory Cache
        with self.lock:
            self.memory_cache.clear()
            self.current_cache_size = 0
        
        self.logger.info("DataManager erfolgreich heruntergefahren", 'data')
    
    def __del__(self):
        """Destruktor - stellt sicher, dass Ressourcen freigegeben werden"""
        try:
            self.shutdown()
        except:
            pass

# Zusätzliche Utility-Funktionen

def timeframe_to_seconds(timeframe: Timeframe) -> int:
    """Konvertiert Timeframe zu Sekunden"""
    
    mapping = {
        Timeframe.MINUTE_1: 60,
        Timeframe.MINUTE_5: 300,
        Timeframe.MINUTE_15: 900,
        Timeframe.MINUTE_30: 1800,
        Timeframe.HOUR_1: 3600,
        Timeframe.HOUR_4: 14400,
        Timeframe.HOUR_12: 43200,
        Timeframe.DAY_1: 86400,
        Timeframe.WEEK_1: 604800
    }
    
    return mapping.get(timeframe, 3600)

def resample_data(data: pd.DataFrame, from_timeframe: Timeframe, 
                 to_timeframe: Timeframe) -> pd.DataFrame:
    """Resampled Daten von einem Timeframe zu einem anderen"""
    
    if data.empty:
        return data
    
    # Zeitintervall für Resampling bestimmen
    resample_rules = {
        Timeframe.MINUTE_1: '1T',
        Timeframe.MINUTE_5: '5T',
        Timeframe.MINUTE_15: '15T',
        Timeframe.MINUTE_30: '30T',
        Timeframe.HOUR_1: '1H',
        Timeframe.HOUR_4: '4H',
        Timeframe.HOUR_12: '12H',
        Timeframe.DAY_1: '1D',
        Timeframe.WEEK_1: '1W'
    }

    rule = resample_rules.get(to_timeframe)
    if not rule:
        return data

    df = data.copy()
    df.set_index('timestamp', inplace=True)

    # Resample OHLCV
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades_count': 'sum' if 'trades_count' in df.columns else 'first'
    }).dropna().reset_index()

    resampled['source'] = df['source'].iloc[0]
    return resampled
