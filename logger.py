"""
Trading Bot Logger System
Strukturiertes Logging mit Datei- und Konsolen-Ausgabe, automatischer Rotation
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import traceback
from enum import Enum
import threading
import time

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class TradingLogger:
    def __init__(self, config_manager=None, log_dir: str = "logs"):
        self.config_manager = config_manager
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Logger-Instanzen für verschiedene Kategorien
        self.loggers = {}
        self.log_formatters = {}
        
        # Thread-Lock für thread-sichere Logs
        self.lock = threading.Lock()
        
        # Performance-Tracking
        self.performance_data = {}
        
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Initialisiert alle Logger mit entsprechenden Handlers"""
        
        # Haupt-Logger
        self._create_logger(
            'main', 
            self.log_dir / 'trading_bot.log',
            console_output=True
        )
        
        # Trading-Logger für alle Handelsaktivitäten
        self._create_logger(
            'trading',
            self.log_dir / 'trading.log',
            console_output=True
        )
        
        # Strategy-Logger für Strategieentscheidungen
        self._create_logger(
            'strategy',
            self.log_dir / 'strategy.log',
            console_output=False
        )
        
        # ML-Logger für Machine Learning Aktivitäten
        self._create_logger(
            'ml',
            self.log_dir / 'ml_training.log',
            console_output=False
        )
        
        # Error-Logger für alle Fehler
        self._create_logger(
            'error',
            self.log_dir / 'errors.log',
            console_output=True,
            level=logging.ERROR
        )
        
        # Performance-Logger für Backtests und Performance-Metriken
        self._create_logger(
            'performance',
            self.log_dir / 'performance.log',
            console_output=False
        )
        
        # Data-Logger für Datenmanagement
        self._create_logger(
            'data',
            self.log_dir / 'data_management.log',
            console_output=False
        )
    
    def _create_logger(self, name: str, log_file: Path, 
                      console_output: bool = True, 
                      level: int = logging.INFO):
        """Erstellt einen Logger mit File- und Console-Handler"""
        
        logger = logging.getLogger(f"trading_bot.{name}")
        logger.setLevel(level)
        
        # Verhindere doppelte Handler
        if logger.handlers:
            return logger
        
        # File Handler mit Rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Console Handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
        
        # Formatter für strukturierte Logs
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Einfacher Formatter für Console
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        if console_output:
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)
        
        self.loggers[name] = logger
        return logger
    
    def get_logger(self, category: str = 'main') -> logging.Logger:
        """Gibt einen Logger für eine bestimmte Kategorie zurück"""
        return self.loggers.get(category, self.loggers['main'])
    
    def log_trade(self, action: str, symbol: str, amount: float, price: float, 
                  strategy: str, exchange: str, order_id: str = None, **kwargs):
        """Spezieller Log für Trades mit strukturierten Daten"""
        
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'value': amount * price,
            'strategy': strategy,
            'exchange': exchange,
            'order_id': order_id,
            **kwargs
        }
        
        # Strukturierter Log-Eintrag
        log_message = f"TRADE | {action.upper()} | {symbol} | {amount:.8f} @ {price:.8f} | Strategy: {strategy} | Exchange: {exchange}"
        
        if order_id:
            log_message += f" | Order ID: {order_id}"
        
        self.get_logger('trading').info(log_message)
        
        # Zusätzlich als JSON für maschinelle Auswertung
        self._log_json('trading_json.log', trade_data)
    
    def log_strategy_decision(self, strategy: str, symbol: str, decision: str, 
                            confidence: float, market_condition: str, **kwargs):
        """Log für Strategieentscheidungen"""
        
        decision_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'symbol': symbol,
            'decision': decision,
            'confidence': confidence,
            'market_condition': market_condition,
            **kwargs
        }
        
        log_message = f"STRATEGY | {strategy} | {symbol} | Decision: {decision} | Confidence: {confidence:.2f} | Market: {market_condition}"
        
        self.get_logger('strategy').info(log_message)
        self._log_json('strategy_decisions.log', decision_data)
    
    def log_ml_training(self, model_name: str, accuracy: float, loss: float, 
                       training_samples: int, validation_accuracy: float = None, **kwargs):
        """Log für ML-Training"""
        
        ml_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'accuracy': accuracy,
            'loss': loss,
            'training_samples': training_samples,
            'validation_accuracy': validation_accuracy,
            **kwargs
        }
        
        log_message = f"ML_TRAINING | {model_name} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f} | Samples: {training_samples}"
        
        if validation_accuracy:
            log_message += f" | Val_Acc: {validation_accuracy:.4f}"
        
        self.get_logger('ml').info(log_message)
        self._log_json('ml_training.log', ml_data)
    
    def log_performance(self, strategy: str, symbol: str, timeframe: str, 
                       metrics: Dict[str, float], **kwargs):
        """Log für Performance-Metriken"""
        
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'symbol': symbol,
            'timeframe': timeframe,
            'metrics': metrics,
            **kwargs
        }
        
        # Wichtige Metriken in der Log-Message
        log_message = f"PERFORMANCE | {strategy} | {symbol} | {timeframe}"
        
        if 'total_return' in metrics:
            log_message += f" | Return: {metrics['total_return']:.2f}%"
        if 'sharpe_ratio' in metrics:
            log_message += f" | Sharpe: {metrics['sharpe_ratio']:.2f}"
        if 'max_drawdown' in metrics:
            log_message += f" | MaxDD: {metrics['max_drawdown']:.2f}%"
        
        self.get_logger('performance').info(log_message)
        self._log_json('performance_metrics.log', perf_data)
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Spezieller Error-Log mit Stacktrace"""
        
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stacktrace': traceback.format_exc(),
            **kwargs
        }
        
        log_message = f"ERROR | {type(error).__name__}: {str(error)}"
        if context:
            log_message += f" | Context: {context}"
        
        self.get_logger('error').error(log_message)
        self.get_logger('error').error(f"Stacktrace:\n{traceback.format_exc()}")
        
        self._log_json('errors.log', error_data)
    
    def log_system_status(self, component: str, status: str, details: Dict[str, Any] = None):
        """Log für System-Status-Updates"""
        
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'status': status,
            'details': details or {}
        }
        
        log_message = f"SYSTEM | {component} | Status: {status}"
        if details:
            # Wichtigste Details in der Message
            detail_str = " | ".join([f"{k}: {v}" for k, v in list(details.items())[:3]])
            log_message += f" | {detail_str}"
        
        self.get_logger('main').info(log_message)
        self._log_json('system_status.log', status_data)
    
    def start_performance_timer(self, operation: str, context: str = "") -> str:
        """Startet einen Performance-Timer"""
        timer_id = f"{operation}_{datetime.now().timestamp()}"
        
        with self.lock:
            self.performance_data[timer_id] = {
                'operation': operation,
                'context': context,
                'start_time': time.time(),
                'start_timestamp': datetime.now().isoformat()
            }
        
        return timer_id
    
    def end_performance_timer(self, timer_id: str, **kwargs):
        """Beendet einen Performance-Timer und loggt das Ergebnis"""
        
        with self.lock:
            if timer_id not in self.performance_data:
                self.get_logger('main').warning(f"Performance timer {timer_id} not found")
                return
            
            timer_data = self.performance_data.pop(timer_id)
        
        end_time = time.time()
        duration = end_time - timer_data['start_time']
        
        perf_log = {
            'timestamp': datetime.now().isoformat(),
            'operation': timer_data['operation'],
            'context': timer_data['context'],
            'duration_seconds': duration,
            'duration_ms': duration * 1000,
            **kwargs
        }
        
        log_message = f"PERFORMANCE | {timer_data['operation']} | Duration: {duration:.3f}s"
        if timer_data['context']:
            log_message += f" | Context: {timer_data['context']}"
        
        self.get_logger('performance').info(log_message)
        self._log_json('performance_timing.log', perf_log)
        
        return duration
    
    def _log_json(self, filename: str, data: Dict[str, Any]):
        """Loggt strukturierte Daten als JSON"""
        json_file = self.log_dir / filename
        
        with self.lock:
            try:
                with open(json_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            except Exception as e:
                self.get_logger('error').error(f"Failed to write JSON log: {e}")
    
    def info(self, message: str, category: str = 'main', **kwargs):
        """Info-Log mit optionalen strukturierten Daten"""
        self.get_logger(category).info(message)
        if kwargs:
            self._log_json(f'{category}_structured.log', {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': message,
                **kwargs
            })
    
    def warning(self, message: str, category: str = 'main', **kwargs):
        """Warning-Log"""
        self.get_logger(category).warning(message)
        if kwargs:
            self._log_json(f'{category}_structured.log', {
                'timestamp': datetime.now().isoformat(),
                'level': 'WARNING',
                'message': message,
                **kwargs
            })
    
    def error(self, message: str, category: str = 'main', **kwargs):
        """Error-Log"""
        self.get_logger(category).error(message)
        if kwargs:
            self._log_json(f'{category}_structured.log', {
                'timestamp': datetime.now().isoformat(),
                'level': 'ERROR',
                'message': message,
                **kwargs
            })
    
    def debug(self, message: str, category: str = 'main', **kwargs):
        """Debug-Log"""
        self.get_logger(category).debug(message)
        if kwargs:
            self._log_json(f'{category}_structured.log', {
                'timestamp': datetime.now().isoformat(),
                'level': 'DEBUG',
                'message': message,
                **kwargs
            })
    
    def critical(self, message: str, category: str = 'main', **kwargs):
        """Critical-Log"""
        self.get_logger(category).critical(message)
        if kwargs:
            self._log_json(f'{category}_structured.log', {
                'timestamp': datetime.now().isoformat(),
                'level': 'CRITICAL',
                'message': message,
                **kwargs
            })
    
    def log_bot_startup(self, version: str, config_summary: Dict[str, Any]):
        """Spezieller Log für Bot-Start"""
        startup_data = {
            'timestamp': datetime.now().isoformat(),
            'event': 'BOT_STARTUP',
            'version': version,
            'config_summary': config_summary
        }
        
        self.info(f"Trading Bot gestartet - Version: {version}", 'main')
        self._log_json('bot_lifecycle.log', startup_data)
        
        # Log wichtige Konfigurationswerte
        for key, value in config_summary.items():
            self.info(f"Config - {key}: {value}", 'main')
    
    def log_bot_shutdown(self, reason: str = "Normal shutdown", **kwargs):
        """Spezieller Log für Bot-Shutdown"""
        shutdown_data = {
            'timestamp': datetime.now().isoformat(),
            'event': 'BOT_SHUTDOWN',
            'reason': reason,
            **kwargs
        }
        
        self.info(f"Trading Bot wird beendet - Grund: {reason}", 'main')
        self._log_json('bot_lifecycle.log', shutdown_data)
    
    def log_position_change(self, symbol: str, old_position: Dict[str, Any], 
                           new_position: Dict[str, Any], reason: str):
        """Log für Positionsänderungen"""
        position_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'old_position': old_position,
            'new_position': new_position,
            'reason': reason,
            'position_change': {
                'size_change': new_position.get('size', 0) - old_position.get('size', 0),
                'value_change': new_position.get('value', 0) - old_position.get('value', 0)
            }
        }
        
        log_message = f"POSITION | {symbol} | {reason} | Size: {old_position.get('size', 0):.8f} -> {new_position.get('size', 0):.8f}"
        
        self.get_logger('trading').info(log_message)
        self._log_json('position_changes.log', position_data)
    
    def log_backtest_result(self, strategy: str, symbol: str, timeframe: str, 
                           start_date: str, end_date: str, results: Dict[str, float]):
        """Log für Backtest-Ergebnisse"""
        backtest_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'results': results
        }
        
        log_message = f"BACKTEST | {strategy} | {symbol} | {start_date} to {end_date}"
        if 'total_return' in results:
            log_message += f" | Return: {results['total_return']:.2f}%"
        if 'win_rate' in results:
            log_message += f" | WinRate: {results['win_rate']:.1f}%"
        if 'trades' in results:
            log_message += f" | Trades: {results['trades']}"
        
        self.get_logger('performance').info(log_message)
        self._log_json('backtest_results.log', backtest_data)
    
    def log_market_analysis(self, symbol: str, timeframe: str, 
                           market_condition: str, confidence: float, 
                           indicators: Dict[str, float]):
        """Log für Marktanalyse"""
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'market_condition': market_condition,
            'confidence': confidence,
            'indicators': indicators
        }
        
        log_message = f"MARKET_ANALYSIS | {symbol} | {timeframe} | Condition: {market_condition} | Confidence: {confidence:.2f}"
        
        self.get_logger('strategy').info(log_message)
        self._log_json('market_analysis.log', analysis_data)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Bereinigt alte Log-Dateien"""
        try:
            import glob
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Alle .log Dateien im log_dir finden
            log_files = glob.glob(str(self.log_dir / "*.log*"))
            
            deleted_count = 0
            for log_file in log_files:
                file_path = Path(log_file)
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except OSError:
                        pass
            
            if deleted_count > 0:
                self.info(f"Bereinigung abgeschlossen: {deleted_count} alte Log-Dateien gelöscht", 'main')
                
        except Exception as e:
            self.error(f"Fehler bei Log-Bereinigung: {e}", 'main')
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über die Log-Dateien zurück"""
        stats = {
            'log_directory': str(self.log_dir),
            'total_log_files': 0,
            'total_size_mb': 0,
            'files': {}
        }
        
        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.is_file():
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    stats['files'][log_file.name] = {
                        'size_mb': round(size_mb, 2),
                        'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                    }
                    stats['total_size_mb'] += size_mb
                    stats['total_log_files'] += 1
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
        except Exception as e:
            self.error(f"Fehler beim Sammeln von Log-Statistiken: {e}", 'main')
        
        return stats
    
    def log_daily_summary(self, trades_today: int, profit_loss: float, 
                         active_positions: int, **kwargs):
        """Tägliche Zusammenfassung"""
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().date().isoformat(),
            'trades_today': trades_today,
            'profit_loss': profit_loss,
            'active_positions': active_positions,
            **kwargs
        }
        
        log_message = f"DAILY_SUMMARY | Trades: {trades_today} | P&L: {profit_loss:.2f} | Active Positions: {active_positions}"
        
        self.get_logger('main').info(log_message)
        self._log_json('daily_summaries.log', summary_data)
    
    def set_log_level(self, level: str, category: str = None):
        """Setzt das Log-Level für eine Kategorie oder alle"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        if category:
            if category in self.loggers:
                self.loggers[category].setLevel(log_level)
                self.info(f"Log-Level für {category} auf {level} gesetzt", 'main')
            else:
                self.warning(f"Logger-Kategorie {category} nicht gefunden", 'main')
        else:
            # Alle Logger
            for logger_name, logger in self.loggers.items():
                logger.setLevel(log_level)
            self.info(f"Log-Level für alle Logger auf {level} gesetzt", 'main')

# Globale Logger-Instanz
_logger_instance = None

def get_logger(config_manager=None) -> TradingLogger:
    """Singleton-Pattern für globalen Logger-Zugriff"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingLogger(config_manager)
    return _logger_instance

def init_logger(config_manager=None) -> TradingLogger:
    """Initialisiert den globalen Logger"""
    global _logger_instance
    _logger_instance = TradingLogger(config_manager)
    return _logger_instance

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Demo ohne ConfigManager
    logger = TradingLogger()
    
    # Verschiedene Log-Typen testen
    logger.info("Trading Bot wird initialisiert", 'main')
    
    logger.log_trade(
        action='BUY',
        symbol='BTCUSDT',
        amount=0.001,
        price=45000.0,
        strategy='uptrend_strategy',
        exchange='binance',
        order_id='123456'
    )
    
    logger.log_strategy_decision(
        strategy='grid_strategy',
        symbol='ETHUSDT',
        decision='HOLD',
        confidence=0.75,
        market_condition='sideways',
        rsi=55.2,
        macd=0.15
    )
    
    logger.log_performance(
        strategy='uptrend_strategy',
        symbol='BTCUSDT',
        timeframe='1h',
        metrics={
            'total_return': 5.2,
            'sharpe_ratio': 1.8,
            'max_drawdown': -2.1,
            'win_rate': 68.5
        }
    )
    
    # Performance-Timer testen
    timer_id = logger.start_performance_timer('backtest_calculation', 'BTC uptrend strategy')
    import time
    time.sleep(0.1)  # Simuliere Berechnung
    duration = logger.end_performance_timer(timer_id, trades_analyzed=1000)
    
    print(f"Demo abgeschlossen. Logs wurden in 'logs/' Verzeichnis gespeichert.")
    print(f"Log-Statistiken: {logger.get_log_stats()}")