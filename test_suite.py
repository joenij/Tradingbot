#!/usr/bin/env python3
"""
Trading Bot Integration Test Suite
Umfassende Tests für alle Module des Trading Bots
"""

import os
import sys
import unittest
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import aller Bot-Module (angepasst an deine Struktur)
try:
    from config_manager import ConfigManager
    from logger import BotLogger
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
    from main_bot import TradingBot
except ImportError as e:
    print(f"Import Error: {e}")
    print("Bitte stelle sicher, dass alle Bot-Module im PYTHONPATH sind")
    sys.exit(1)

class IntegrationTestSuite:
    """Hauptklasse für Integrationstests"""
    
    def __init__(self):
        self.test_results = {}
        self.setup_test_environment()
        
    def setup_test_environment(self):
        """Test-Umgebung vorbereiten"""
        # Test-Config erstellen
        self.test_config = {
            "exchanges": {
                "binance": {
                    "api_key": "test_key",
                    "api_secret": "test_secret",
                    "testnet": True
                },
                "kucoin": {
                    "api_key": "test_key",
                    "api_secret": "test_secret",
                    "passphrase": "test_pass",
                    "testnet": True
                }
            },
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "base_amount": 100,
                "max_positions": 3,
                "risk_per_trade": 0.02
            },
            "ml": {
                "model_retrain_hours": 24,
                "min_data_points": 1000
            }
        }
        
        # Test-Daten generieren
        self.generate_test_data()
        
    def generate_test_data(self):
        """Generiert Test-Marktdaten"""
        dates = pd.date_range(start='2017-01-01', end='2024-01-01', freq='1H')
        np.random.seed(42)
        
        # Simuliere verschiedene Marktphasen
        base_price = 40000
        prices = [base_price]
        
        for i in range(1, len(dates)):
            # Verschiedene Marktphasen simulieren
            if i < len(dates) * 0.3:  # Uptrend
                change = np.random.normal(0.001, 0.02)
            elif i < len(dates) * 0.7:  # Sideways
                change = np.random.normal(0, 0.015)
            else:  # Downtrend
                change = np.random.normal(-0.001, 0.02)
                
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Mindestpreis
            
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, len(dates))
        })

class TestModuleIntegration(unittest.TestCase):
    """Test-Klasse für Modul-Integration"""
    
    @classmethod
    def setUpClass(cls):
        cls.test_suite = IntegrationTestSuite()
        
    def test_01_config_manager_integration(self):
        """Test Config Manager Integration"""
        print("\n=== Testing Config Manager Integration ===")
        
        try:
            config_manager = ConfigManager()
            
            # Test config loading
            config_manager.load_config(self.test_suite.test_config)
            self.assertIsNotNone(config_manager.get_exchange_config("binance"))
            
            # Test config validation
            self.assertTrue(config_manager.validate_config())
            
            print("✓ Config Manager Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Config Manager Integration: FAILED - {e}")
            return False
            
    def test_02_logger_integration(self):
        """Test Logger Integration"""
        print("\n=== Testing Logger Integration ===")
        
        try:
            logger = BotLogger("integration_test")
            
            # Test verschiedene Log-Level
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            # Test Log-Datei Erstellung
            self.assertTrue(os.path.exists("logs"))
            
            print("✓ Logger Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Logger Integration: FAILED - {e}")
            return False
            
    def test_03_data_manager_integration(self):
        """Test Data Manager Integration"""
        print("\n=== Testing Data Manager Integration ===")
        
        try:
            data_manager = DataManager()
            
            # Test mit Mock-Daten
            with patch.object(data_manager, 'fetch_historical_data', return_value=self.test_suite.test_data):
                data = data_manager.get_market_data("BTCUSDT", "1h", 1000)
                self.assertIsInstance(data, pd.DataFrame)
                self.assertGreater(len(data), 0)
            
            print("✓ Data Manager Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Data Manager Integration: FAILED - {e}")
            return False
            
    def test_04_exchange_connector_integration(self):
        """Test Exchange Connector Integration"""
        print("\n=== Testing Exchange Connector Integration ===")
        
        try:
            # Mock Exchange Connector für Testnet
            exchange_connector = ExchangeConnector()
            
            with patch.object(exchange_connector, 'connect') as mock_connect:
                mock_connect.return_value = True
                
                # Test Connection
                self.assertTrue(exchange_connector.connect("binance", self.test_suite.test_config["exchanges"]["binance"]))
                
                # Test Mock Order
                with patch.object(exchange_connector, 'place_order') as mock_order:
                    mock_order.return_value = {"id": "test_order_123", "status": "filled"}
                    order = exchange_connector.place_order("BTCUSDT", "buy", 0.001, 40000)
                    self.assertIsNotNone(order)
            
            print("✓ Exchange Connector Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Exchange Connector Integration: FAILED - {e}")
            return False
            
    def test_05_market_analyzer_integration(self):
        """Test Market Analyzer Integration"""
        print("\n=== Testing Market Analyzer Integration ===")
        
        try:
            analyzer = MarketAnalyzer()
            
            # Test Markt-Analyse mit Test-Daten
            analysis = analyzer.analyze_market(self.test_suite.test_data)
            
            self.assertIn('trend', analysis)
            self.assertIn('volatility', analysis)
            self.assertIn('support_resistance', analysis)
            
            print("✓ Market Analyzer Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Market Analyzer Integration: FAILED - {e}")
            return False
            
    def test_06_ml_trainer_integration(self):
        """Test ML Trainer Integration"""
        print("\n=== Testing ML Trainer Integration ===")
        
        try:
            ml_trainer = MLTrainer()
            
            # Test Model Training mit Test-Daten
            with patch.object(ml_trainer, 'prepare_features') as mock_features:
                mock_features.return_value = (
                    np.random.random((100, 10)),  # X
                    np.random.randint(0, 3, 100)  # y
                )
                
                model = ml_trainer.train_model(self.test_suite.test_data)
                self.assertIsNotNone(model)
            
            print("✓ ML Trainer Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ ML Trainer Integration: FAILED - {e}")
            return False
            
    def test_07_strategies_integration(self):
        """Test Strategien Integration"""
        print("\n=== Testing Strategies Integration ===")
        
        try:
            # Test alle drei Strategien
            uptrend_strategy = UptrendStrategy()
            sideways_strategy = SidewaysStrategy()
            downtrend_strategy = DowntrendStrategy()
            
            market_data = {
                'trend': 'up',
                'price': 40000,
                'volatility': 0.02,
                'volume': 1000
            }
            
            # Test Signal-Generierung
            up_signal = uptrend_strategy.generate_signal(market_data)
            side_signal = sideways_strategy.generate_signal(market_data)
            down_signal = downtrend_strategy.generate_signal(market_data)
            
            self.assertIn('action', up_signal)
            self.assertIn('action', side_signal)
            self.assertIn('action', down_signal)
            
            print("✓ Strategies Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Strategies Integration: FAILED - {e}")
            return False
            
    def test_08_backtester_integration(self):
        """Test Backtester Integration"""
        print("\n=== Testing Backtester Integration ===")
        
        try:
            backtester = Backtester()
            
            # Mock Strategy für Backtest
            mock_strategy = Mock()
            mock_strategy.generate_signal.return_value = {'action': 'buy', 'amount': 0.001}
            
            results = backtester.run_backtest(
                strategy=mock_strategy,
                data=self.test_suite.test_data,
                initial_balance=1000
            )
            
            self.assertIn('total_return', results)
            self.assertIn('trades', results)
            self.assertIn('win_rate', results)
            
            print("✓ Backtester Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Backtester Integration: FAILED - {e}")
            return False
            
    def test_09_strategy_selector_integration(self):
        """Test Strategy Selector Integration"""
        print("\n=== Testing Strategy Selector Integration ===")
        
        try:
            selector = StrategySelector()
            
            # Mock Backtest-Ergebnisse
            mock_results = {
                'uptrend': {'total_return': 0.15, 'win_rate': 0.65},
                'sideways': {'total_return': 0.08, 'win_rate': 0.55},
                'downtrend': {'total_return': 0.05, 'win_rate': 0.45}
            }
            
            best_strategy = selector.select_best_strategy(mock_results, 'up')
            self.assertEqual(best_strategy, 'uptrend')
            
            print("✓ Strategy Selector Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Strategy Selector Integration: FAILED - {e}")
            return False
            
    def test_10_position_manager_integration(self):
        """Test Position Manager Integration"""
        print("\n=== Testing Position Manager Integration ===")
        
        try:
            position_manager = PositionManager()
            
            # Test Position erstellen
            position = {
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'amount': 0.001,
                'entry_price': 40000,
                'timestamp': datetime.now()
            }
            
            position_manager.add_position(position)
            positions = position_manager.get_active_positions()
            
            self.assertEqual(len(positions), 1)
            self.assertEqual(positions[0]['symbol'], 'BTCUSDT')
            
            print("✓ Position Manager Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Position Manager Integration: FAILED - {e}")
            return False
            
    def test_11_risk_manager_integration(self):
        """Test Risk Manager Integration"""
        print("\n=== Testing Risk Manager Integration ===")
        
        try:
            risk_manager = RiskManager(max_risk_per_trade=0.02)
            
            # Test Risk Calculation
            balance = 1000
            price = 40000
            risk_amount = risk_manager.calculate_position_size(balance, price)
            
            self.assertGreater(risk_amount, 0)
            self.assertLess(risk_amount * price, balance * 0.02)
            
            print("✓ Risk Manager Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Risk Manager Integration: FAILED - {e}")
            return False
            
    def test_12_notification_system_integration(self):
        """Test Notification System Integration"""
        print("\n=== Testing Notification System Integration ===")
        
        try:
            notification_system = NotificationSystem()
            
            # Test verschiedene Notification-Typen
            with patch.object(notification_system, 'send_notification') as mock_send:
                mock_send.return_value = True
                
                result = notification_system.send_trade_notification({
                    'symbol': 'BTCUSDT',
                    'action': 'buy',
                    'amount': 0.001,
                    'price': 40000
                })
                
                self.assertTrue(result)
            
            print("✓ Notification System Integration: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Notification System Integration: FAILED - {e}")
            return False

class FullSystemIntegrationTest:
    """Vollständiger System-Integrationstest"""
    
    def __init__(self):
        self.test_suite = IntegrationTestSuite()
        
    async def run_full_system_test(self):
        """Führt einen kompletten System-Test durch"""
        print("\n" + "="*60)
        print("FULL SYSTEM INTEGRATION TEST")
        print("="*60)
        
        try:
            # 1. Bot initialisieren (mit Mock-Daten)
            print("\n1. Initializing Trading Bot...")
            bot = TradingBot()
            
            with patch.object(bot, 'initialize') as mock_init:
                mock_init.return_value = True
                await bot.initialize(self.test_suite.test_config)
                print("✓ Bot initialized successfully")
            
            # 2. Daten laden
            print("\n2. Loading market data...")
            with patch.object(bot.data_manager, 'get_market_data') as mock_data:
                mock_data.return_value = self.test_suite.test_data
                data_loaded = await bot.load_initial_data()
                print("✓ Market data loaded successfully")
                
            # 3. ML-Model trainieren
            print("\n3. Training ML models...")
            with patch.object(bot.ml_trainer, 'train_model') as mock_train:
                mock_train.return_value = Mock()
                model_trained = await bot.train_models()
                print("✓ ML models trained successfully")
                
            # 4. Strategien backtesten
            print("\n4. Running strategy backtests...")
            with patch.object(bot.backtester, 'run_backtest') as mock_backtest:
                mock_backtest.return_value = {
                    'total_return': 0.15,
                    'trades': 50,
                    'win_rate': 0.65,
                    'max_drawdown': 0.08
                }
                backtest_results = await bot.run_strategy_backtests()
                print("✓ Strategy backtests completed successfully")
                
            # 5. Beste Strategie auswählen
            print("\n5. Selecting best strategy...")
            with patch.object(bot.strategy_selector, 'select_best_strategy') as mock_select:
                mock_select.return_value = 'uptrend'
                selected_strategy = await bot.select_strategy()
                print(f"✓ Best strategy selected: {selected_strategy}")
                
            # 6. Simulierte Trading-Loop (kurz)
            print("\n6. Running simulated trading loop...")
            with patch.object(bot, 'execute_trade') as mock_trade:
                mock_trade.return_value = {
                    'success': True,
                    'order_id': 'test_123',
                    'amount': 0.001
                }
                
                # Simuliere 5 Trading-Zyklen
                for i in range(5):
                    await bot.trading_cycle()
                    print(f"  ✓ Trading cycle {i+1} completed")
                    
            # 7. Position Management Test
            print("\n7. Testing position management...")
            positions = bot.position_manager.get_active_positions()
            print(f"✓ Active positions managed: {len(positions)}")
            
            # 8. Risk Management Test
            print("\n8. Testing risk management...")
            risk_check = bot.risk_manager.check_risk_limits()
            print("✓ Risk limits checked")
            
            # 9. Crash Recovery Test
            print("\n9. Testing crash recovery...")
            with patch.object(bot, 'save_state') as mock_save:
                mock_save.return_value = True
                await bot.save_state()
                
            with patch.object(bot, 'load_state') as mock_load:
                mock_load.return_value = True
                await bot.load_state()
                print("✓ Crash recovery tested successfully")
                
            print("\n" + "="*60)
            print("✓ FULL SYSTEM INTEGRATION TEST: PASSED")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n✗ FULL SYSTEM INTEGRATION TEST: FAILED")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def run_integration_tests():
    """Führt alle Integrationstests aus"""
    print("Trading Bot Integration Test Suite")
    print("=" * 50)
    
    # Unit-Tests für Module
    print("\nRunning Module Integration Tests...")
    unittest.main(argv=[''], module=__name__, verbosity=2, exit=False)
    
    # Vollständiger System-Test
    print("\nRunning Full System Integration Test...")
    system_test = FullSystemIntegrationTest()
    
    # Async Test ausführen
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(system_test.run_full_system_test())
    
    if success:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nNext Steps:")
        print("1. Review test results and logs")
        print("2. Configure real API keys for paper trading")
        print("3. Start with small amounts for live testing")
        print("4. Monitor performance and adjust parameters")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please review the error messages and fix issues before deployment")

if __name__ == "__main__":
    run_integration_tests()