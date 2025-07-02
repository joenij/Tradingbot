"""
Trading Bot Configuration Manager
Sichere Verwaltung aller Bot-Konfigurationen und API-Credentials
"""

import os
import json
import configparser
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
from cryptography.fernet import Fernet
import base64

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Pfade für verschiedene Konfigurationsdateien
        self.main_config_file = self.config_dir / "bot_config.ini"
        self.credentials_file = self.config_dir / "credentials.enc"
        self.strategies_config_file = self.config_dir / "strategies.json"
        
        # Verschlüsselungsschlüssel für Credentials
        self.key_file = self.config_dir / ".key"
        self._encryption_key = self._get_or_create_key()
        
        # Standard-Konfiguration laden oder erstellen
        self._load_or_create_config()
    
    def _get_or_create_key(self) -> bytes:
        """Erstellt oder lädt den Verschlüsselungsschlüssel für API-Credentials"""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Verstecke die Key-Datei (Windows)
            if os.name == 'nt':
                os.system(f'attrib +h "{self.key_file}"')
            return key
    
    def _encrypt_data(self, data: str) -> str:
        """Verschlüsselt sensible Daten"""
        fernet = Fernet(self._encryption_key)
        encrypted_data = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Entschlüsselt sensible Daten"""
        fernet = Fernet(self._encryption_key)
        decoded_data = base64.b64decode(encrypted_data.encode())
        return fernet.decrypt(decoded_data).decode()
    
    def _load_or_create_config(self):
        """Lädt vorhandene Konfiguration oder erstellt Standard-Konfiguration"""
        self.config = configparser.ConfigParser()
        
        if self.main_config_file.exists():
            self.config.read(self.main_config_file)
        else:
            self._create_default_config()
            self.save_main_config()
    
    def _create_default_config(self):
        """Erstellt Standard-Konfiguration"""
        # Bot Grundeinstellungen
        self.config['BOT'] = {
            'name': 'Advanced Trading Bot',
            'version': '1.0.0',
            'log_level': 'INFO',
            'update_interval_seconds': '60',
            'max_concurrent_trades': '5',
            'emergency_stop_loss_percent': '10.0'
        }
        
        # Exchange Einstellungen
        self.config['EXCHANGE'] = {
            'primary_exchange': 'binance',
            'secondary_exchange': 'kucoin',
            'testnet_mode': 'true',
            'order_timeout_seconds': '30',
            'rate_limit_buffer': '0.1'
        }
        
        # Risikomanagement
        self.config['RISK'] = {
            'max_portfolio_risk_percent': '2.0',
            'max_single_trade_percent': '1.0',
            'stop_loss_percent': '3.0',
            'take_profit_percent': '6.0',
            'trailing_stop_percent': '2.0'
        }
        
        # Daten Management
        self.config['DATA'] = {
            'historical_data_start': '2017-01-01',
            'data_update_interval_minutes': '5',
            'cache_size_mb': '500',
            'backup_data': 'true'
        }
        
        # Machine Learning
        self.config['ML'] = {
            'retrain_interval_hours': '24',
            'min_training_samples': '1000',
            'model_validation_split': '0.2',
            'feature_importance_threshold': '0.05'
        }
        
        # Backtesting
        self.config['BACKTEST'] = {
            'initial_balance': '10000',
            'commission_percent': '0.1',
            'slippage_percent': '0.05',
            'walk_forward_periods': '12'
        }
        
        # Benachrichtigungen
        self.config['NOTIFICATIONS'] = {
            'enable_console_output': 'true',
            'enable_file_logging': 'true',
            'enable_telegram': 'false',
            'enable_email': 'false',
            'notify_on_trades': 'true',
            'notify_on_errors': 'true'
        }
    
    def save_main_config(self):
        """Speichert die Hauptkonfiguration"""
        with open(self.main_config_file, 'w') as f:
            self.config.write(f)
    
    def get_config_value(self, section: str, key: str, fallback: Any = None) -> Any:
        """Holt einen Konfigurationswert mit automatischer Typkonvertierung"""
        try:
            value = self.config.get(section, key)
            
            # Versuche automatische Typkonvertierung
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            
            # Versuche als Integer
            try:
                return int(value)
            except ValueError:
                pass
            
            # Versuche als Float
            try:
                return float(value)
            except ValueError:
                pass
            
            # Rückgabe als String
            return value
            
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def set_config_value(self, section: str, key: str, value: Any):
        """Setzt einen Konfigurationswert"""
        if section not in self.config:
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
        self.save_main_config()
    
    def set_api_credentials(self, exchange: str, api_key: str, api_secret: str, 
                          passphrase: str = None):
        """Speichert API-Credentials verschlüsselt"""
        credentials = {
            'api_key': api_key,
            'api_secret': api_secret
        }
        
        if passphrase:
            credentials['passphrase'] = passphrase
        
        # Alle existierenden Credentials laden
        all_credentials = self._load_credentials()
        all_credentials[exchange] = credentials
        
        # Verschlüsselt speichern
        encrypted_data = self._encrypt_data(json.dumps(all_credentials))
        with open(self.credentials_file, 'w') as f:
            f.write(encrypted_data)
    
    def get_api_credentials(self, exchange: str) -> Dict[str, str]:
        """Lädt API-Credentials für eine Exchange"""
        all_credentials = self._load_credentials()
        return all_credentials.get(exchange, {})
    
    def _load_credentials(self) -> Dict[str, Dict[str, str]]:
        """Lädt alle verschlüsselten Credentials"""
        if not self.credentials_file.exists():
            return {}
        
        try:
            with open(self.credentials_file, 'r') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._decrypt_data(encrypted_data)
            return json.loads(decrypted_data)
        except Exception:
            return {}
    
    def load_strategy_config(self) -> Dict[str, Any]:
        """Lädt Strategiekonfiguration"""
        if self.strategies_config_file.exists():
            with open(self.strategies_config_file, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_strategy_config()
    
    def _create_default_strategy_config(self) -> Dict[str, Any]:
        """Erstellt Standard-Strategiekonfiguration"""
        strategy_config = {
            "uptrend_strategy": {
                "name": "Hold Long Strategy",
                "enabled": True,
                "parameters": {
                    "trend_confirmation_periods": 3,
                    "min_trend_strength": 0.6,
                    "trailing_stop_percent": 2.0,
                    "take_profit_percent": 8.0,
                    "max_holding_days": 30
                },
                "indicators": {
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bb_period": 20,
                    "bb_std": 2
                }
            },
            "sideways_strategy": {
                "name": "Adaptive Grid Trading",
                "enabled": True,
                "parameters": {
                    "grid_levels": 10,
                    "grid_spacing_percent": 1.5,
                    "min_range_percent": 3.0,
                    "max_range_percent": 15.0,
                    "rebalance_threshold_percent": 0.5,
                    "profit_target_percent": 1.0
                },
                "adaptive_settings": {
                    "volatility_adjustment": True,
                    "volume_weighting": True,
                    "support_resistance_levels": True,
                    "dynamic_spacing": True
                }
            },
            "downtrend_strategy": {
                "name": "Bear Market Protection",
                "enabled": True,
                "parameters": {
                    "trend_confirmation_periods": 2,
                    "min_decline_percent": 3.0,
                    "reentry_confirmation_periods": 3,
                    "max_drawdown_percent": 5.0,
                    "cash_reserve_percent": 20.0
                },
                "signals": {
                    "rsi_threshold": 80,
                    "macd_bearish_cross": True,
                    "volume_spike_threshold": 1.5,
                    "support_break_confirmation": True
                }
            },
            "ml_settings": {
                "features": [
                    "rsi", "macd", "bb_upper", "bb_lower", "volume_ratio",
                    "price_change_1h", "price_change_4h", "price_change_24h",
                    "volatility_1h", "volatility_24h"
                ],
                "lookback_periods": [5, 10, 20, 50],
                "prediction_horizon": 24,
                "model_update_threshold": 0.05
            }
        }
        
        self.save_strategy_config(strategy_config)
        return strategy_config
    
    def save_strategy_config(self, config: Dict[str, Any]):
        """Speichert Strategiekonfiguration"""
        with open(self.strategies_config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def get_trading_pairs(self) -> list:
        """Gibt die zu handelnden Trading-Paare zurück"""
        pairs_str = self.get_config_value('TRADING', 'pairs', 'BTCUSDT,ETHUSDT,ADAUSDT')
        return [pair.strip() for pair in pairs_str.split(',')]
    
    def set_trading_pairs(self, pairs: list):
        """Setzt die zu handelnden Trading-Paare"""
        self.set_config_value('TRADING', 'pairs', ','.join(pairs))
    
    def is_testnet_mode(self) -> bool:
        """Prüft ob Testnet-Modus aktiv ist"""
        return self.get_config_value('EXCHANGE', 'testnet_mode', True)
    
    def get_log_level(self) -> str:
        """Gibt das Log-Level zurück"""
        return self.get_config_value('BOT', 'log_level', 'INFO')
    
    def get_max_concurrent_trades(self) -> int:
        """Gibt die maximale Anzahl gleichzeitiger Trades zurück"""
        return self.get_config_value('BOT', 'max_concurrent_trades', 5)
    
    def get_risk_parameters(self) -> Dict[str, float]:
        """Gibt alle Risikomanagement-Parameter zurück"""
        return {
            'max_portfolio_risk_percent': self.get_config_value('RISK', 'max_portfolio_risk_percent', 2.0),
            'max_single_trade_percent': self.get_config_value('RISK', 'max_single_trade_percent', 1.0),
            'stop_loss_percent': self.get_config_value('RISK', 'stop_loss_percent', 3.0),
            'take_profit_percent': self.get_config_value('RISK', 'take_profit_percent', 6.0),
            'trailing_stop_percent': self.get_config_value('RISK', 'trailing_stop_percent', 2.0)
        }
    
    def validate_config(self) -> list:
        """Validiert die Konfiguration und gibt Fehler zurück"""
        errors = []
        
        # Prüfe kritische Werte
        if self.get_config_value('RISK', 'max_portfolio_risk_percent', 0) <= 0:
            errors.append("max_portfolio_risk_percent muss größer als 0 sein")
        
        if self.get_config_value('RISK', 'stop_loss_percent', 0) <= 0:
            errors.append("stop_loss_percent muss größer als 0 sein")
        
        if self.get_config_value('BOT', 'max_concurrent_trades', 0) <= 0:
            errors.append("max_concurrent_trades muss größer als 0 sein")
        
        # Prüfe API-Credentials
        primary_exchange = self.get_config_value('EXCHANGE', 'primary_exchange', '')
        if primary_exchange:
            credentials = self.get_api_credentials(primary_exchange)
            if not credentials.get('api_key') or not credentials.get('api_secret'):
                errors.append(f"API-Credentials für {primary_exchange} fehlen")
        
        return errors

# Beispiel für die Nutzung
if __name__ == "__main__":
    config_manager = ConfigManager()
    
    # Beispiel: API-Credentials setzen (nur für Demo)
    # config_manager.set_api_credentials('binance', 'your_api_key', 'your_api_secret')
    # config_manager.set_api_credentials('kucoin', 'your_api_key', 'your_api_secret', 'your_passphrase')
    
    # Konfiguration validieren
    errors = config_manager.validate_config()
    if errors:
        print("Konfigurationsfehler gefunden:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Konfiguration ist valide!")
    
    print(f"Testnet-Modus: {config_manager.is_testnet_mode()}")
    print(f"Log-Level: {config_manager.get_log_level()}")
    print(f"Max concurrent trades: {config_manager.get_max_concurrent_trades()}")