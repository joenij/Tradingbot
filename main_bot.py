"""
Trading Bot - Main Bot Integration
Vollständige Integration aller Komponenten mit automatischer Strategie-Auswahl
und kontinuierlichem Lernen
"""

import asyncio
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import threading
from collections import defaultdict
import psutil
import os

# Eigene Module (diese würden in echten Dateien existieren)
from config_manager import ConfigManager
from logger import TradingLogger, get_logger
from data_manager import DataManager
from exchange_connector import ExchangeConnector
from market_analyzer import MarketAnalyzer
from ml_trainer import MLTrainer
from strategy_uptrend import UptrendStrategy
from strategy_sideways import SidewaysStrategy
from strategy_downtrend import DowntrendStrategy
from backtester import Backtester
from strategy_selector import StrategySelector
from position_manager import PositionManager
from risk_manager import RiskManager
from notification_system import NotificationSystem

class TradingBot:
    """
    Hauptklasse des Trading Bots
    Orchestriert alle Komponenten und führt das automatische Trading durch
    """
    
    def __init__(self, config_path: str = "config/bot_config.json"):
        self.config_path = config_path
        self.is_running = False
        self.shutdown_requested = False
        
        # Komponenten
        self.config_manager = None
        self.logger = None
        self.data_manager = None
        self.exchanges = {}
        self.market_analyzer = None
        self.ml_trainer = None
        self.strategies = {}
        self.backtester = None
        self.strategy_selector = None
        self.position_manager = None
        self.risk_manager = None
        self.notification_system = None
        
        # Status-Tracking
        self.current_market_condition = None
        self.active_strategy = None
        self.last_strategy_check = None
        self.last_training_update = None
        self.performance_metrics = {}
        self.daily_stats = defaultdict(float)
        
        # Threading
        self.trading_thread = None
        self.monitoring_thread = None
        self.training_thread = None
        
        # Crash Recovery
        self.state_file = Path("data/bot_state.json")
        self.recovery_data = {}
        
        # Performance Monitoring
        self.loop_times = []
        self.error_count = 0
        self.last_heartbeat = datetime.now()
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Registriert Signal-Handler für graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Signal {signum} empfangen. Initiiere graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    async def initialize(self):
        """Initialisiert alle Bot-Komponenten"""
        try:
            print("🤖 Trading Bot wird initialisiert...")
            
            # 1. Config Manager
            self.config_manager = ConfigManager(self.config_path)
            config = self.config_manager.get_config()
            
            # 2. Logger
            self.logger = TradingLogger(self.config_manager)
            
            self.logger.log_bot_startup(
                version="1.0.0",
                config_summary={
                    'trading_pairs': config.get('trading', {}).get('pairs', []),
                    'exchanges': list(config.get('exchanges', {}).keys()),
                    'risk_per_trade': config.get('risk_management', {}).get('max_risk_per_trade', 0.02),
                    'ml_enabled': config.get('ml', {}).get('enabled', True)
                }
            )
            
            # 3. Data Manager
            self.logger.info("Initialisiere Data Manager...", 'main')
            self.data_manager = DataManager(self.config_manager, self.logger)
            await self.data_manager.initialize()
            
            # 4. Exchange Connectors
            self.logger.info("Verbinde zu Exchanges...", 'main')
            await self._initialize_exchanges()
            
            # 5. Market Analyzer
            self.logger.info("Initialisiere Market Analyzer...", 'main')
            self.market_analyzer = MarketAnalyzer(
                self.data_manager, 
                self.config_manager, 
                self.logger
            )
            
            # 6. ML Trainer
            if config.get('ml', {}).get('enabled', True):
                self.logger.info("Initialisiere ML Trainer...", 'main')
                self.ml_trainer = MLTrainer(
                    self.data_manager,
                    self.config_manager,
                    self.logger
                )
                await self.ml_trainer.initialize()
            
            # 7. Strategien
            self.logger.info("Initialisiere Trading Strategien...", 'main')
            await self._initialize_strategies()
            
            # 8. Backtester
            self.logger.info("Initialisiere Backtester...", 'main')
            self.backtester = Backtester(
                self.data_manager,
                self.config_manager,
                self.logger
            )
            
            # 9. Strategy Selector
            self.logger.info("Initialisiere Strategy Selector...", 'main')
            self.strategy_selector = StrategySelector(
                self.strategies,
                self.backtester,
                self.market_analyzer,
                self.config_manager,
                self.logger
            )
            
            # 10. Position Manager
            self.logger.info("Initialisiere Position Manager...", 'main')
            self.position_manager = PositionManager(
                self.exchanges,
                self.config_manager,
                self.logger
            )
            await self.position_manager.initialize()
            
            # 11. Risk Manager
            self.logger.info("Initialisiere Risk Manager...", 'main')
            self.risk_manager = RiskManager(
                self.position_manager,
                self.config_manager,
                self.logger
            )
            
            # 12. Notification System
            self.logger.info("Initialisiere Notification System...", 'main')
            self.notification_system = NotificationSystem(
                self.config_manager,
                self.logger
            )
            
            # 13. Crash Recovery
            await self._load_recovery_state()
            
            # 14. Initiales Training und Strategie-Auswahl
            await self._perform_initial_setup()
            
            self.logger.info("✅ Trading Bot erfolgreich initialisiert!", 'main')
            await self.notification_system.send_notification(
                "🤖 Trading Bot gestartet",
                "Alle Komponenten erfolgreich initialisiert"
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Fehler bei Bot-Initialisierung: {e}"
            if self.logger:
                self.logger.log_error(e, "Bot initialization")
            else:
                print(f"KRITISCHER FEHLER: {error_msg}")
                print(traceback.format_exc())
            return False
    
    async def _initialize_exchanges(self):
        """Initialisiert alle konfigurierten Exchanges"""
        exchanges_config = self.config_manager.get_config().get('exchanges', {})
        
        for exchange_name, exchange_config in exchanges_config.items():
            if exchange_config.get('enabled', False):
                try:
                    self.logger.info(f"Verbinde zu {exchange_name}...", 'main')
                    
                    exchange = ExchangeConnector(
                        exchange_name,
                        exchange_config,
                        self.logger
                    )
                    
                    await exchange.initialize()
                    
                    # Test-Verbindung
                    balance = await exchange.get_balance()
                    self.logger.info(f"✅ {exchange_name} verbunden. Verfügbares Guthaben: {len(balance)} Assets", 'main')
                    
                    self.exchanges[exchange_name] = exchange
                    
                except Exception as e:
                    self.logger.log_error(e, f"Exchange {exchange_name} initialization")
                    # Nicht kritisch - Bot kann mit einem Exchange laufen
        
        if not self.exchanges:
            raise Exception("Keine Exchange-Verbindungen verfügbar!")
    
    async def _initialize_strategies(self):
        """Initialisiert alle Trading-Strategien"""
        try:
            # Uptrend Strategy
            self.strategies['uptrend'] = UptrendStrategy(
                self.data_manager,
                self.config_manager,
                self.logger
            )
            
            # Sideways Strategy
            self.strategies['sideways'] = SidewaysStrategy(
                self.data_manager,
                self.config_manager,
                self.logger
            )
            
            # Downtrend Strategy
            self.strategies['downtrend'] = DowntrendStrategy(
                self.data_manager,
                self.config_manager,
                self.logger
            )
            
            # Strategien initialisieren
            for name, strategy in self.strategies.items():
                await strategy.initialize()
                self.logger.info(f"✅ Strategie '{name}' initialisiert", 'main')
            
        except Exception as e:
            self.logger.log_error(e, "Strategy initialization")
            raise
    
    async def _perform_initial_setup(self):
        """Führt initiales Training und Setup durch"""
        try:
            self.logger.info("Führe initiales Setup durch...", 'main')
            
            # 1. Historische Daten laden
            trading_pairs = self.config_manager.get_config().get('trading', {}).get('pairs', [])
            
            for pair in trading_pairs:
                self.logger.info(f"Lade historische Daten für {pair}...", 'data')
                await self.data_manager.update_historical_data(
                    pair, 
                    '1h', 
                    start_date='2017-01-01'
                )
            
            # 2. ML-Modelle trainieren (falls aktiviert)
            if self.ml_trainer:
                self.logger.info("Starte initiales ML-Training...", 'ml')
                await self.ml_trainer.train_all_models()
            
            # 3. Backtests durchführen
            self.logger.info("Führe initiale Backtests durch...", 'performance')
            await self._run_strategy_backtests()
            
            # 4. Beste Strategie auswählen
            await self._select_initial_strategy()
            
            # 5. Aktuelle Positionen wiederherstellen
            await self._restore_positions()
            
        except Exception as e:
            self.logger.log_error(e, "Initial setup")
            raise
    
    async def _run_strategy_backtests(self):
        """Führt Backtests für alle Strategien durch"""
        try:
            trading_pairs = self.config_manager.get_config().get('trading', {}).get('pairs', [])
            
            for pair in trading_pairs:
                for strategy_name, strategy in self.strategies.items():
                    self.logger.info(f"Backteste {strategy_name} für {pair}...", 'performance')
                    
                    # 6 Monate Backtest
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=180)
                    
                    result = await self.backtester.run_backtest(
                        strategy=strategy,
                        symbol=pair,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        timeframe='1h'
                    )
                    
                    self.logger.log_backtest_result(
                        strategy=strategy_name,
                        symbol=pair,
                        timeframe='1h',
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        results=result
                    )
                    
                    # Performance in internem Tracking speichern
                    if pair not in self.performance_metrics:
                        self.performance_metrics[pair] = {}
                    self.performance_metrics[pair][strategy_name] = result
            
        except Exception as e:
            self.logger.log_error(e, "Strategy backtesting")
    
    async def _select_initial_strategy(self):
        """Wählt die initiale Strategie basierend auf aktueller Marktlage"""
        try:
            trading_pairs = self.config_manager.get_config().get('trading', {}).get('pairs', [])
            
            if not trading_pairs:
                return
            
            # Hauptpaar für Strategieauswahl
            main_pair = trading_pairs[0]
            
            # Aktuelle Marktbedingungen analysieren
            market_condition = await self.market_analyzer.analyze_market_condition(
                main_pair, '1h'
            )
            
            self.current_market_condition = market_condition['condition']
            
            # Beste Strategie auswählen
            selected_strategy = await self.strategy_selector.select_best_strategy(
                main_pair,
                market_condition,
                self.performance_metrics.get(main_pair, {})
            )
            
            self.active_strategy = selected_strategy
            self.last_strategy_check = datetime.now()
            
            self.logger.info(
                f"🎯 Initiale Strategie gewählt: {selected_strategy} für Marktlage: {self.current_market_condition}",
                'strategy'
            )
            
            await self.notification_system.send_notification(
                "🎯 Strategie gewählt",
                f"Aktive Strategie: {selected_strategy}\nMarktlage: {self.current_market_condition}"
            )
            
        except Exception as e:
            self.logger.log_error(e, "Initial strategy selection")
    
    async def start(self):
        """Startet den Trading Bot"""
        if self.is_running:
            self.logger.warning("Bot läuft bereits!", 'main')
            return
        
        try:
            # Initialisierung
            if not await self.initialize():
                self.logger.error("Bot-Initialisierung fehlgeschlagen!", 'main')
                return False
            
            self.is_running = True
            self.logger.info("🚀 Trading Bot gestartet!", 'main')
            
            # Haupt-Threads starten
            self.trading_thread = threading.Thread(target=self._run_trading_loop, daemon=True)
            self.monitoring_thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
            
            if self.ml_trainer:
                self.training_thread = threading.Thread(target=self._run_training_loop, daemon=True)
                self.training_thread.start()
            
            self.trading_thread.start()
            self.monitoring_thread.start()
            
            # Haupt-Loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.log_error(e, "Bot startup")
            await self.shutdown()
            return False
    
    def _run_trading_loop(self):
        """Trading-Loop in separatem Thread"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        try:
            loop.run_until_complete(self._trading_loop())
        except Exception as e:
            self.logger.log_error(e, "Trading loop")
        finally:
            loop.close()
    
    def _run_monitoring_loop(self):
        """Monitoring-Loop in separatem Thread"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        try:
            loop.run_until_complete(self._monitoring_loop())
        except Exception as e:
            self.logger.log_error(e, "Monitoring loop")
        finally:
            loop.close()
    
    def _run_training_loop(self):
        """ML-Training-Loop in separatem Thread"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        try:
            loop.run_until_complete(self._training_loop())
        except Exception as e:
            self.logger.log_error(e, "Training loop")
        finally:
            loop.close()
    
    async def _main_loop(self):
        """Haupt-Event-Loop"""
        try:
            while self.is_running and not self.shutdown_requested:
                try:
                    # Heartbeat
                    self.last_heartbeat = datetime.now()
                    
                    # Status prüfen
                    await self._check_system_health()
                    
                    # Kurz warten
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.log_error(e, "Main loop iteration")
                    self.error_count += 1
                    
                    if self.error_count > 10:
                        self.logger.critical("Zu viele Fehler in Main Loop! Beende Bot...", 'main')
                        break
                    
                    await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt - Beende Bot...", 'main')
        except Exception as e:
            self.logger.log_error(e, "Main loop")
        finally:
            await self.shutdown()
    
    async def _trading_loop(self):
        """Haupt-Trading-Loop"""
        loop_interval = self.config_manager.get_config().get('trading', {}).get('loop_interval', 60)
        
        while self.is_running and not self.shutdown_requested:
            loop_start = time.time()
            
            try:
                # 1. Marktdaten aktualisieren
                await self._update_market_data()
                
                # 2. Marktbedingungen analysieren
                await self._analyze_market_conditions()
                
                # 3. Strategie-Wechsel prüfen
                await self._check_strategy_switch()
                
                # 4. Trading-Signale generieren
                await self._generate_trading_signals()
                
                # 5. Risikomanagement
                await self._apply_risk_management()
                
                # 6. Orders ausführen
                await self._execute_orders()
                
                # 7. Positionen überwachen
                await self._monitor_positions()
                
                # 8. Status speichern
                await self._save_state()
                
                # Loop-Zeit tracken
                loop_time = time.time() - loop_start
                self.loop_times.append(loop_time)
                if len(self.loop_times) > 100:
                    self.loop_times.pop(0)
                
                # Nächste Iteration warten
                sleep_time = max(0, loop_interval - loop_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    self.logger.warning(f"Trading loop überzogen: {loop_time:.2f}s", 'performance')
                
            except Exception as e:
                self.logger.log_error(e, "Trading loop iteration")
                await asyncio.sleep(30)  # Längere Pause bei Fehlern
    
    async def _monitoring_loop(self):
        """System-Monitoring-Loop"""
        while self.is_running and not self.shutdown_requested:
            try:
                # System-Ressourcen überwachen
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                if cpu_percent > 80:
                    self.logger.warning(f"Hohe CPU-Auslastung: {cpu_percent}%", 'system')
                
                if memory.percent > 80:
                    self.logger.warning(f"Hoher Speicherverbrauch: {memory.percent}%", 'system')
                
                if disk.percent > 90:
                    self.logger.warning(f"Wenig Festplattenspeicher: {disk.percent}%", 'system')
                
                # Performance-Metriken loggen
                if self.loop_times:
                    avg_loop_time = sum(self.loop_times) / len(self.loop_times)
                    self.logger.debug(f"Durchschnittliche Loop-Zeit: {avg_loop_time:.2f}s", 'performance')
                
                # Tägliche Zusammenfassung
                await self._generate_daily_summary()
                
                # Log-Bereinigung
                await self._cleanup_logs()
                
                await asyncio.sleep(300)  # Alle 5 Minuten
                
            except Exception as e:
                self.logger.log_error(e, "Monitoring loop")
                await asyncio.sleep(60)
    
    async def _training_loop(self):
        """ML-Training-Loop"""
        if not self.ml_trainer:
            return
        
        training_interval = self.config_manager.get_config().get('ml', {}).get('training_interval_hours', 24)
        
        while self.is_running and not self.shutdown_requested:
            try:
                # Prüfen ob Training nötig ist
                if (self.last_training_update is None or 
                    datetime.now() - self.last_training_update > timedelta(hours=training_interval)):
                    
                    self.logger.info("Starte ML-Modell Update...", 'ml')
                    
                    # Neue Daten sammeln
                    await self.data_manager.update_recent_data()
                    
                    # Modelle neu trainieren
                    await self.ml_trainer.incremental_training()
                    
                    # Backtests mit neuen Modellen
                    await self._run_strategy_backtests()
                    
                    self.last_training_update = datetime.now()
                    
                    self.logger.info("✅ ML-Modell Update abgeschlossen", 'ml')
                
                await asyncio.sleep(3600)  # Stündlich prüfen
                
            except Exception as e:
                self.logger.log_error(e, "Training loop")
                await asyncio.sleep(1800)  # 30 Min bei Fehlern
    
    async def _update_market_data(self):
        """Aktualisiert Marktdaten für alle Trading-Paare"""
        try:
            trading_pairs = self.config_manager.get_config().get('trading', {}).get('pairs', [])
            
            for pair in trading_pairs:
                # Aktuelle Daten von allen Exchanges
                for exchange_name, exchange in self.exchanges.items():
                    try:
                        # Ticker-Daten
                        ticker = await exchange.get_ticker(pair)
                        await self.data_manager.store_ticker_data(exchange_name, pair, ticker)
                        
                        # Neueste Kerzen
                        klines = await exchange.get_klines(pair, '1h', limit=100)
                        await self.data_manager.store_kline_data(exchange_name, pair, '1h', klines)
                        
                    except Exception as e:
                        self.logger.log_error(e, f"Market data update {exchange_name} {pair}")
            
        except Exception as e:
            self.logger.log_error(e, "Market data update")
    
    async def _analyze_market_conditions(self):
        """Analysiert aktuelle Marktbedingungen"""
        try:
            trading_pairs = self.config_manager.get_config().get('trading', {}).get('pairs', [])
            
            if not trading_pairs:
                return
            
            # Hauptpaar für Marktanalyse
            main_pair = trading_pairs[0]
            
            # Marktbedingungen analysieren
            market_analysis = await self.market_analyzer.analyze_market_condition(
                main_pair, '1h'
            )
            
            self.current_market_condition = market_analysis['condition']
            
            # Detaillierte Analyse loggen
            self.logger.log_market_analysis(
                symbol=main_pair,
                timeframe='1h',
                market_condition=market_analysis['condition'],
                confidence=market_analysis['confidence'],
                indicators=market_analysis['indicators']
            )
            
        except Exception as e:
            self.logger.log_error(e, "Market conditions analysis")
    
    async def _check_strategy_switch(self):
        """Prüft ob Strategie-Wechsel nötig ist"""
        try:
            # Nur alle 30 Minuten prüfen
            if (self.last_strategy_check and 
                datetime.now() - self.last_strategy_check < timedelta(minutes=30)):
                return
            
            trading_pairs = self.config_manager.get_config().get('trading', {}).get('pairs', [])
            if not trading_pairs:
                return
            
            main_pair = trading_pairs[0]
            
            # Aktuelle Marktbedingungen
            market_condition = await self.market_analyzer.analyze_market_condition(
                main_pair, '1h'
            )
            
            # Strategie-Auswahl basierend auf Marktlage
            recommended_strategy = await self.strategy_selector.select_best_strategy(
                main_pair,
                market_condition,
                self.performance_metrics.get(main_pair, {})
            )
            
            # Strategie-Wechsel nötig?
            if recommended_strategy != self.active_strategy:
                self.logger.info(
                    f"🔄 Strategie-Wechsel: {self.active_strategy} -> {recommended_strategy}",
                    'strategy'
                )
                
                # Alte Positionen schließen (falls konfiguriert)
                close_on_switch = self.config_manager.get_config().get('trading', {}).get('close_positions_on_strategy_switch', False)
                if close_on_switch:
                    await self.position_manager.close_all_positions("Strategy switch")
                
                # Neue Strategie aktivieren
                self.active_strategy = recommended_strategy
                
                await self.notification_system.send_notification(
                    "🔄 Strategie-Wechsel",
                    f"Neue Strategie: {recommended_strategy}\nMarktlage: {market_condition['condition']}"
                )
            
            self.last_strategy_check = datetime.now()
            
        except Exception as e:
            self.logger.log_error(e, "Strategy switch check")
    
    async def _generate_trading_signals(self):
        """Generiert Trading-Signale mit aktiver Strategie"""
        try:
            if not self.active_strategy or self.active_strategy not in self.strategies:
                return
            
            strategy = self.strategies[self.active_strategy]
            trading_pairs = self.config_manager.get_config().get('trading', {}).get('pairs', [])
            
            for pair in trading_pairs:
                # Signal generieren
                signal = await strategy.generate_signal(pair, '1h')
                
                if signal and signal['action'] != 'HOLD':
                    self.logger.log_strategy_decision(
                        strategy=self.active_strategy,
                        symbol=pair,
                        decision=signal['action'],
                        confidence=signal['confidence'],
                        market_condition=self.current_market_condition,
                        price=signal.get('price', 0),
                        reason=signal.get('reason', '')
                    )
                    
                    # Signal für Ausführung vormerken
                    await self._queue_signal_for_execution(pair, signal)
            
        except Exception as e:
            self.logger.log_error(e, "Trading signal generation")
    
    async def _queue_signal_for_execution(self, symbol: str, signal: Dict[str, Any]):
        """Fügt Signal zur Ausführungsqueue hinzu"""
        try:
            # Risiko-Check vor Ausführung
            risk_approved = await self.risk_manager.check_trade_risk(
                symbol=symbol,
                action=signal['action'],
                amount=signal.get('amount', 0),
                price=signal.get('price', 0)
            )
            
            if risk_approved:
                # Signal zur Ausführung vormerken (hier würde eine Queue implementiert)
                self.logger.info(f"📊 Signal für Ausführung vorgemerkt: {signal['action']} {symbol}", 'trading')
            else:
                self.logger.warning(f"⚠️ Signal abgelehnt (Risiko): {signal['action']} {symbol}", 'trading')
                
        except Exception as e:
            self.logger.log_error(e, "Signal queueing")
    
    async def _apply_risk_management(self):
        """Wendet Risikomanagement-Regeln an"""
        try:
            # Tägliche Verlustgrenze prüfen
            daily_pnl = self.daily_stats.get('pnl', 0)
            max_daily_loss = self.config_manager.get_config().get('risk_management', {}).get('max_daily_loss', -0.05)
            
            if daily_pnl < max_daily_loss:
                self.logger.warning(f"⚠️ Tägliche Verlustgrenze erreicht: {daily_pnl:.2%}", 'risk')
                # Alle offenen Positionen schließen
                await self.position_manager.close_all_positions("Daily loss limit reached")
                return
            
            # Positions-spezifisches Risikomanagement
            await self.risk_manager.monitor_all_positions()
            
        except Exception as e:
            self.logger.log_error(e, "Risk management")
    
    async def _execute_orders(self):
        """Führt geplante Orders aus"""
        try:
            # Hier würde die Order-Ausführung implementiert
            # Für jetzt nur ein Platzhalter
            pass
            
        except Exception as e:
            self.logger.log_error(e, "Order execution")
    
    async def _monitor_positions(self):
        """Überwacht alle offenen Positionen"""
        try:
            positions = await self.position_manager.get_all_positions()
            
            for position in positions:
                # Position aktualisieren
                await self.position_manager.update_position(position['id'])
                
                # Stop-Loss/Take-Profit prüfen
                await self.position_manager.check_exit_conditions(position['id'])
                
                # Performance tracken
                if position['status'] == 'closed':
                    pnl = position.get('pnl', 0)
                    self.daily_stats['pnl'] += pnl
                    self.daily_stats['trades'] += 1
                    
                    if pnl > 0:
                        self.daily_stats['winning_trades'] += 1
                    else:
                        self.daily_stats['losing_trades'] += 1
            
        except Exception as e:
            self.logger.log_error(e, "Position monitoring")
    
    async def _save_state(self):
        """Speichert aktuellen Bot-Status für Crash-Recovery"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'current_market_condition': self.current_market_condition,
                'active_strategy': self.active_strategy,
                'last_strategy_check': self.last_strategy_check.isoformat() if self.last_strategy_check else None,
                'last_training_update': self.last_training_update.isoformat() if self.last_training_update else None,
                'daily_stats': dict(self.daily_stats),
                'performance_metrics': self.performance_metrics,
                'error_count': self.error_count
            }
            
            # Atomisches Schreiben
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            temp_file.replace(self.state_file)
            
        except Exception as e:
            self.logger.log_error(e, "State saving")
    
    async def _load_recovery_state(self):
        """Lädt gespeicherten Status für Crash-Recovery"""
        try:
            if not self.state_file.exists():
                self.logger.info("Keine Recovery-Daten gefunden - Neustart", 'main')
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.recovery_data = state
            
            # Status wiederherstellen
            if 'current_market_condition' in state:
                self.current_market_condition = state['current_market_condition']
            
            if 'active_strategy' in state:
                self.active_strategy = state['active_strategy']
            
            if 'last_strategy_check' in state and state['last_strategy_check']:
                self.last_strategy_check = datetime.fromisoformat(state['last_strategy_check'])
            
            if 'last_training_update' in state and state['last_training_update']:
                self.last_training_update = datetime.fromisoformat(state['last_training_update'])
            
            if 'daily_stats' in state:
                self.daily_stats.update(state['daily_stats'])
            
            if 'performance_metrics' in state:
                self.performance_metrics = state['performance_metrics']
            
            self.logger.info(f"✅ Recovery-Status geladen: {state['timestamp']}", 'main')
            
        except Exception as e:
            self.logger.log_error(e, "Recovery state loading")
            self.logger.warning("Recovery fehlgeschlagen - Neustart", 'main')
    
    async def _restore_positions(self):
        """Stellt offene Positionen nach Neustart wieder her"""
        try:
            # Positionen von allen Exchanges abrufen
            for exchange_name, exchange in self.exchanges.items():
                try:
                    positions = await exchange.get_open_positions()
                    
                    for position in positions:
                        # Position im Position Manager registrieren
                        await self.position_manager.register_existing_position(
                            exchange_name, position
                        )
                        
                        self.logger.info(
                            f"📈 Position wiederhergestellt: {position['symbol']} {position['side']} {position['size']}",
                            'position'
                        )
                
                except Exception as e:
                    self.logger.log_error(e, f"Position restore {exchange_name}")
                    
        except Exception as e:
            self.logger.log_error(e, "Position restoration")
    
    async def _check_system_health(self):
        """Prüft System-Gesundheit"""
        try:
            # Heartbeat-Timeout prüfen
            if datetime.now() - self.last_heartbeat > timedelta(minutes=5):
                self.logger.warning("Heartbeat-Timeout erkannt!", 'system')
                self.last_heartbeat = datetime.now()
            
            # Exchange-Verbindungen prüfen
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # Ping-Test
                    await exchange.ping()
                except Exception as e:
                    self.logger.warning(f"Exchange {exchange_name} nicht erreichbar: {e}", 'system')
                    # Reconnect versuchen
                    await exchange.reconnect()
            
            # Speicher-Leaks prüfen
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if current_memory > 1000:  # 1GB
                self.logger.warning(f"Hoher Speicherverbrauch: {current_memory:.1f} MB", 'system')
            
        except Exception as e:
            self.logger.log_error(e, "System health check")
    
    async def _generate_daily_summary(self):
        """Generiert tägliche Zusammenfassung"""
        try:
            now = datetime.now()
            
            # Nur einmal pro Tag um Mitternacht
            if now.hour != 0 or now.minute > 5:
                return
            
            # Tägliche Statistiken
            total_trades = self.daily_stats.get('trades', 0)
            winning_trades = self.daily_stats.get('winning_trades', 0)
            losing_trades = self.daily_stats.get('losing_trades', 0)
            total_pnl = self.daily_stats.get('pnl', 0)
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            summary = f"""
📊 TÄGLICHE ZUSAMMENFASSUNG - {now.strftime('%Y-%m-%d')}
{'='*50}
📈 Gesamte Trades: {total_trades}
✅ Gewinn-Trades: {winning_trades}
❌ Verlust-Trades: {losing_trades}
📊 Win-Rate: {win_rate:.1f}%
💰 Gesamt P&L: {total_pnl:.2f}
🎯 Aktive Strategie: {self.active_strategy}
🌊 Marktlage: {self.current_market_condition}
⚡ Durchschn. Loop-Zeit: {sum(self.loop_times)/len(self.loop_times):.2f}s
🔄 Fehleranzahl: {self.error_count}
"""
            
            self.logger.info(summary, 'daily_summary')
            
            # Notification senden
            await self.notification_system.send_notification(
                "📊 Tägliche Zusammenfassung",
                f"Trades: {total_trades} | Win-Rate: {win_rate:.1f}% | P&L: {total_pnl:.2f}"
            )
            
            # Statistiken zurücksetzen
            self.daily_stats.clear()
            
        except Exception as e:
            self.logger.log_error(e, "Daily summary generation")
    
    async def _cleanup_logs(self):
        """Bereinigt alte Log-Dateien"""
        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                return
            
            # Logs älter als 30 Tage löschen
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for log_file in log_dir.glob("*.log"):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        log_file.unlink()
                        self.logger.debug(f"Alte Log-Datei gelöscht: {log_file.name}", 'system')
                except Exception as e:
                    self.logger.debug(f"Fehler beim Löschen von {log_file.name}: {e}", 'system')
            
        except Exception as e:
            self.logger.log_error(e, "Log cleanup")
    
    async def shutdown(self):
        """Fährt den Bot ordnungsgemäß herunter"""
        if not self.is_running:
            return
        
        try:
            self.logger.info("🛑 Bot-Shutdown initiiert...", 'main')
            self.is_running = False
            
            # 1. Neue Trades stoppen
            self.logger.info("Stoppe neue Trades...", 'main')
            
            # 2. Offene Orders canceln (falls konfiguriert)
            cancel_orders = self.config_manager.get_config().get('trading', {}).get('cancel_orders_on_shutdown', True)
            if cancel_orders and self.position_manager:
                self.logger.info("Cancele offene Orders...", 'main')
                await self.position_manager.cancel_all_orders()
            
            # 3. Positionen schließen (falls konfiguriert)
            close_positions = self.config_manager.get_config().get('trading', {}).get('close_positions_on_shutdown', False)
            if close_positions and self.position_manager:
                self.logger.info("Schließe offene Positionen...", 'main')
                await self.position_manager.close_all_positions("Bot shutdown")
            
            # 4. Finalen Status speichern
            await self._save_state()
            
            # 5. Threads beenden
            self.logger.info("Beende Threads...", 'main')
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=10)
            
            # 6. Exchange-Verbindungen schließen
            for exchange_name, exchange in self.exchanges.items():
                try:
                    await exchange.close()
                    self.logger.info(f"✅ Exchange {exchange_name} getrennt", 'main')
                except Exception as e:
                    self.logger.log_error(e, f"Exchange {exchange_name} disconnect")
            
            # 7. Finale Notification
            if self.notification_system:
                await self.notification_system.send_notification(
                    "🛑 Bot gestoppt",
                    "Trading Bot wurde ordnungsgemäß heruntergefahren"
                )
            
            # 8. Finale Statistiken
            if self.daily_stats:
                total_pnl = self.daily_stats.get('pnl', 0)
                total_trades = self.daily_stats.get('trades', 0)
                self.logger.info(f"📊 Session-Statistiken: {total_trades} Trades, P&L: {total_pnl:.2f}", 'main')
            
            self.logger.info("✅ Bot-Shutdown abgeschlossen!", 'main')
            
        except Exception as e:
            self.logger.log_error(e, "Bot shutdown")
            print(f"FEHLER beim Shutdown: {e}")
    
    # Öffentliche Methoden für externe Kontrolle
    async def get_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Bot-Status zurück"""
        return {
            'is_running': self.is_running,
            'current_market_condition': self.current_market_condition,
            'active_strategy': self.active_strategy,
            'last_strategy_check': self.last_strategy_check.isoformat() if self.last_strategy_check else None,
            'last_training_update': self.last_training_update.isoformat() if self.last_training_update else None,
            'daily_stats': dict(self.daily_stats),
            'error_count': self.error_count,
            'connected_exchanges': list(self.exchanges.keys()),
            'avg_loop_time': sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0,
            'uptime': (datetime.now() - self.last_heartbeat).total_seconds() if self.last_heartbeat else 0
        }
    
    async def force_strategy_switch(self, strategy_name: str) -> bool:
        """Erzwingt Wechsel zu bestimmter Strategie"""
        try:
            if strategy_name not in self.strategies:
                self.logger.error(f"Unbekannte Strategie: {strategy_name}", 'main')
                return False
            
            old_strategy = self.active_strategy
            self.active_strategy = strategy_name
            self.last_strategy_check = datetime.now()
            
            self.logger.info(f"🔄 Manueller Strategie-Wechsel: {old_strategy} -> {strategy_name}", 'strategy')
            
            await self.notification_system.send_notification(
                "🔄 Manueller Strategie-Wechsel",
                f"Strategie geändert zu: {strategy_name}"
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, "Force strategy switch")
            return False
    
    async def force_training_update(self) -> bool:
        """Erzwingt ML-Training Update"""
        try:
            if not self.ml_trainer:
                self.logger.warning("ML-Training nicht aktiviert", 'ml')
                return False
            
            self.logger.info("🔄 Manuelles Training-Update gestartet...", 'ml')
            
            # Neue Daten sammeln
            await self.data_manager.update_recent_data()
            
            # Modelle neu trainieren
            await self.ml_trainer.incremental_training()
            
            # Backtests aktualisieren
            await self._run_strategy_backtests()
            
            self.last_training_update = datetime.now()
            
            self.logger.info("✅ Manuelles Training-Update abgeschlossen", 'ml')
            
            await self.notification_system.send_notification(
                "🔄 Training Update",
                "ML-Modelle wurden manuell aktualisiert"
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, "Force training update")
            return False
    
    async def emergency_stop(self) -> bool:
        """Notfall-Stopp: Schließt alle Positionen und stoppt Bot"""
        try:
            self.logger.warning("🚨 NOTFALL-STOPP AKTIVIERT!", 'main')
            
            # Alle Positionen sofort schließen
            if self.position_manager:
                await self.position_manager.close_all_positions("Emergency stop")
            
            # Alle Orders canceln
            for exchange_name, exchange in self.exchanges.items():
                try:
                    await exchange.cancel_all_orders()
                except Exception as e:
                    self.logger.log_error(e, f"Emergency cancel orders {exchange_name}")
            
            # Bot stoppen
            self.shutdown_requested = True
            
            await self.notification_system.send_notification(
                "🚨 NOTFALL-STOPP",
                "Bot wurde notfallmäßig gestoppt. Alle Positionen geschlossen."
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, "Emergency stop")
            return False


# Hauptfunktion
async def main():
    """Hauptfunktion zum Starten des Trading Bots"""
    print("🤖 Trading Bot wird gestartet...")
    print("=" * 50)
    
    # Konfigurationspfad
    config_path = "config/bot_config.json"
    
    # Prüfen ob Konfiguration existiert
    if not Path(config_path).exists():
        print(f"❌ Konfigurationsdatei nicht gefunden: {config_path}")
        print("Bitte erstelle die Konfigurationsdatei oder passe den Pfad an.")
        return
    
    # Bot initialisieren
    bot = TradingBot(config_path)
    
    try:
        # Bot starten
        await bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Strg+C erkannt - Bot wird gestoppt...")
        await bot.shutdown()
        
    except Exception as e:
        print(f"❌ Kritischer Fehler: {e}")
        traceback.print_exc()
        await bot.shutdown()
    
    print("👋 Bot beendet.")


# CLI-Interface für direkte Kontrolle
class BotCLI:
    """Kommandozeilen-Interface für Bot-Kontrolle"""
    
    def __init__(self, bot: TradingBot):
        self.bot = bot
        self.commands = {
            'status': self._status,
            'strategy': self._strategy,
            'train': self._train,
            'stop': self._stop,
            'emergency': self._emergency,
            'help': self._help
        }
    
    async def _status(self, args):
        """Zeigt Bot-Status"""
        status = await self.bot.get_status()
        print("\n📊 BOT STATUS")
        print("=" * 30)
        for key, value in status.items():
            print(f"{key}: {value}")
    
    async def _strategy(self, args):
        """Wechselt Strategie"""
        if not args:
            print("Verfügbare Strategien: uptrend, sideways, downtrend")
            return
        
        strategy = args[0]
        success = await self.bot.force_strategy_switch(strategy)
        print(f"✅ Strategie-Wechsel: {'Erfolgreich' if success else 'Fehlgeschlagen'}")
    
    async def _train(self, args):
        """Startet Training"""
        success = await self.bot.force_training_update()
        print(f"✅ Training-Update: {'Erfolgreich' if success else 'Fehlgeschlagen'}")
    
    async def _stop(self, args):
        """Stoppt Bot"""
        await self.bot.shutdown()
        print("🛑 Bot gestoppt")
    
    async def _emergency(self, args):
        """Notfall-Stopp"""
        success = await self.bot.emergency_stop()
        print(f"🚨 Notfall-Stopp: {'Aktiviert' if success else 'Fehlgeschlagen'}")
    
    async def _help(self, args):
        """Zeigt Hilfe"""
        print("\n🤖 BOT KOMMANDOS")
        print("=" * 30)
        print("status     - Bot-Status anzeigen")
        print("strategy X - Strategie wechseln (uptrend/sideways/downtrend)")
        print("train      - ML-Training starten")
        print("stop       - Bot stoppen")
        print("emergency  - Notfall-Stopp")
        print("help       - Diese Hilfe")
    
    async def run_command(self, command_line: str):
        """Führt Kommando aus"""
        parts = command_line.strip().split()
        if not parts:
            return
        
        command = parts[0].lower()
        args = parts[1:]
        
        if command in self.commands:
            await self.commands[command](args)
        else:
            print(f"❌ Unbekanntes Kommando: {command}")
            await self._help([])


# Startroutine mit CLI-Option
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--cli":
            # CLI-Modus
            print("🖥️  CLI-Modus aktiviert")
            print("Verwende 'help' für verfügbare Kommandos")
            
            async def cli_mode():
                config_path = "config/bot_config.json"
                bot = TradingBot(config_path)
                cli = BotCLI(bot)
                
                # Bot im Hintergrund starten
                bot_task = asyncio.create_task(bot.start())
                
                # CLI-Loop
                while bot.is_running:
                    try:
                        command = input("Bot> ").strip()
                        if command.lower() in ['quit', 'exit']:
                            break
                        await cli.run_command(command)
                    except EOFError:
                        break
                    except KeyboardInterrupt:
                        break
                
                # Bot stoppen
                await bot.shutdown()
                bot_task.cancel()
            
            asyncio.run(cli_mode())
        else:
            # Normaler Modus
            asyncio.run(main())
            
    except KeyboardInterrupt:
        print("\n👋 Auf Wiedersehen!")
    except Exception as e:
        print(f"❌ Fataler Fehler: {e}")
        traceback.print_exc()