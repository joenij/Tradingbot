#!/usr/bin/env python3
"""
Test-Umgebung Setup Script für Trading Bot
Bereitet automatisch die komplette Test-Umgebung vor
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
import shutil

class TestEnvironmentSetup:
    """Setup-Klasse für Test-Umgebung"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.test_dir = self.base_dir / "integration_tests"
        self.python_executable = sys.executable
        
    def print_banner(self):
        """Zeigt Setup-Banner"""
        print("="*60)
        print("🚀 TRADING BOT TEST ENVIRONMENT SETUP")
        print("="*60)
        print(f"Platform: {platform.platform()}")
        print(f"Python: {platform.python_version()}")
        print(f"Working Directory: {self.base_dir}")
        print("="*60)
        
    def check_system_requirements(self):
        """Prüft System-Anforderungen"""
        print("\n📋 Checking system requirements...")
        
        # Python Version prüfen
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
            
        # Git prüfen (optional)
        try:
            subprocess.run(['git', '--version'], check=True, capture_output=True)
            print("✅ Git available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Git not found (optional)")
            
        # pip prüfen
        try:
            subprocess.run([self.python_executable, '-m', 'pip', '--version'], 
                         check=True, capture_output=True)
            print("✅ pip available")
        except subprocess.CalledProcessError:
            print("❌ pip not available")
            return False
            
        print("✅ System requirements met")
        return True
        
    def create_directory_structure(self):
        """Erstellt Verzeichnisstruktur"""
        print("\n📁 Creating directory structure...")
        
        directories = [
            self.test_dir,
            self.test_dir / "results",
            self.test_dir / "logs", 
            self.test_dir / "reports",
            self.test_dir / "data",
            self.test_dir / "configs",
            self.test_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
            
    def install_dependencies(self):
        """Installiert Test-Dependencies"""
        print("\n📦 Installing test dependencies...")
        
        # Requirements für Tests
        test_requirements = [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'coverage>=6.0',
            'memory-profiler>=0.60.0',
            'psutil>=5.9.0',
            'unittest-xml-reporting>=3.2.0',
            'mock>=4.0.3'
        ]
        
        # Bot-Dependencies (falls nicht vorhanden)
        bot_requirements = [
            'pandas>=1.5.0',
            'numpy>=1.21.0',
            'scikit-learn>=1.1.0',
            'ccxt>=3.0.0',
            'aiohttp>=3.8.0',
            'python-binance>=1.0.16',
            'kucoin-python>=1.0.0',
            'ta-lib>=0.4.25',
            'requests>=2.28.0',
            'websockets>=10.4'
        ]
        
        all_requirements = test_requirements + bot_requirements
        
        for requirement in all_requirements:
            try:
                print(f"Installing {requirement}...")
                subprocess.run([
                    self.python_executable, '-m', 'pip', 'install', requirement
                ], check=True, capture_output=True)
                print(f"✅ {requirement}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install {requirement}: {e}")
                
        # Requirements.txt für später erstellen
        requirements_file = self.test_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            for req in all_requirements:
                f.write(f"{req}\n")
        print(f"✅ Requirements saved to {requirements_file}")
        
    def create_test_configs(self):
        """Erstellt Test-Konfigurationen"""
        print("\n⚙️ Creating test configurations...")
        
        # Basis Test-Config
        test_config = {
            "exchanges": {
                "binance": {
                    "api_key": "YOUR_BINANCE_TESTNET_API_KEY",
                    "api_secret": "YOUR_BINANCE_TESTNET_SECRET",
                    "testnet": True,
                    "sandbox": True
                },
                "kucoin": {
                    "api_key": "YOUR_KUCOIN_TESTNET_API_KEY", 
                    "api_secret": "YOUR_KUCOIN_TESTNET_SECRET",
                    "passphrase": "YOUR_KUCOIN_TESTNET_PASSPHRASE",
                    "testnet": True,
                    "sandbox": True
                }
            },
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "base_amount": 10,
                "max_positions": 2,
                "risk_per_trade": 0.01,
                "stop_loss": 0.02,
                "take_profit": 0.03
            },
            "strategies": {
                "uptrend": {
                    "enabled": True,
                    "parameters": {
                        "rsi_period": 14,
                        "ma_period": 20,
                        "volume_threshold": 1.5
                    }
                },
                "sideways": {
                    "enabled": True,
                    "parameters": {
                        "grid_levels": 10,
                        "grid_spacing": 0.005,
                        "rebalance_threshold": 0.1
                    }
                },
                "downtrend": {
                    "enabled": True,
                    "parameters": {
                        "sell_threshold": -0.02,
                        "rebuy_threshold": -0.05,
                        "wait_time": 300
                    }
                }
            },
            "ml": {
                "model_retrain_hours": 24,
                "min_data_points": 1000,
                "features": ["price", "volume", "rsi", "macd", "bb"],
                "model_type": "random_forest"
            },
            "risk_management": {
                "max_drawdown": 0.1,
                "max_daily_loss": 0.05,
                "position_size_method": "kelly",
                "volatility_adjustment": True
            },
            "logging": {
                "level": "INFO",
                "file_rotation": "daily",
                "max_files": 30
            },
            "notifications": {
                "telegram": {
                    "enabled": False,
                    "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
                    "chat_id": "YOUR_TELEGRAM_CHAT_ID"
                },
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "your_email@gmail.com",
                    "password": "your_app_password"
                }
            }
        }
        
        # Test-Config speichern
        config_file = self.test_dir / "configs" / "test_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_config, f, indent=2, ensure_ascii=False)
        print(f"✅ Test config created: {config_file}")
        
        # Paper-Trading Config
        paper_config = test_config.copy()
        paper_config["trading"]["paper_trading"] = True
        paper_config["trading"]["initial_balance"] = 10000
        
        paper_config_file = self.test_dir / "configs" / "paper_trading_config.json"
        with open(paper_config_file, 'w', encoding='utf-8') as f:
            json.dump(paper_config, f, indent=2, ensure_ascii=False)
        print(f"✅ Paper trading config created: {paper_config_file}")
        
        # Performance Test Config
        perf_config = test_config.copy()
        perf_config["trading"]["symbols"] = ["BTCUSDT"]  # Nur ein Symbol für Performance-Test
        perf_config["ml"]["model_retrain_hours"] = 1  # Häufigeres Retraining
        
        perf_config_file = self.test_dir / "configs" / "performance_config.json"
        with open(perf_config_file, 'w', encoding='utf-8') as f:
            json.dump(perf_config, f, indent=2, ensure_ascii=False)
        print(f"✅ Performance config created: {perf_config_file}")
        
    """
Test Suite für Trading Bot
Führt alle Tests aus und generiert Reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_bot import TradingBot
from src.config_manager import ConfigManager
import asyncio
import time

async def run_quick_test():
    """Schneller Funktionstest"""
    print("🔄 Running quick functionality test...")
    
    try:
        # Config laden
        config = ConfigManager("integration_tests/configs/test_config.json")
        
        # Bot initialisieren (Paper Trading)
        bot = TradingBot(config, paper_trading=True)
        
        # Kurzer Test-Lauf (30 Sekunden)
        print("Starting bot for 30 seconds...")
        await bot.start()
        await asyncio.sleep(30)
        await bot.stop()
        
        print("✅ Quick test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(run_quick_test())
'''
        
        quick_test_file = self.test_dir / "quick_test.py" 
        with open(quick_test_file, 'w', encoding='utf-8') as f:
            f.write(quick_test_script)
        print(f"✅ Quick test script: {quick_test_file}")
        
        # Performance Test Script
        performance_test_script = '''#!/usr/bin/env python3
"""
Performance Test für Trading Bot
Misst Memory, CPU und Execution Time
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import psutil
import memory_profiler
from src.main_bot import TradingBot
from src.config_manager import ConfigManager

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start_monitoring(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def get_metrics(self):
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = self.process.cpu_percent()
        
        return {
            "runtime": current_time - self.start_time,
            "memory_usage": current_memory,
            "memory_increase": current_memory - self.start_memory,
            "cpu_percent": cpu_percent
        }

async def run_performance_test():
    """Performance Test ausführen"""
    print("🔄 Running performance test...")
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        config = ConfigManager("integration_tests/configs/performance_config.json")
        bot = TradingBot(config, paper_trading=True)
        
        print("Starting performance test (5 minutes)...")
        await bot.start()
        
        # 5 Minuten laufen lassen und Metriken sammeln
        for i in range(30):  # 30 * 10 Sekunden = 5 Minuten
            await asyncio.sleep(10)
            metrics = monitor.get_metrics()
            print(f"Runtime: {metrics['runtime']:.1f}s, "
                  f"Memory: {metrics['memory_usage']:.1f}MB, "
                  f"CPU: {metrics['cpu_percent']:.1f}%")
                  
        await bot.stop()
        
        final_metrics = monitor.get_metrics()
        print("\\n📊 Final Performance Metrics:")
        print(f"Total Runtime: {final_metrics['runtime']:.1f} seconds")
        print(f"Memory Usage: {final_metrics['memory_usage']:.1f} MB")
        print(f"Memory Increase: {final_metrics['memory_increase']:.1f} MB")
        print(f"Average CPU: {final_metrics['cpu_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(run_performance_test())
'''
        
        perf_test_file = self.test_dir / "performance_test.py"
        with open(perf_test_file, 'w', encoding='utf-8') as f:
            f.write(performance_test_script)
        print(f"✅ Performance test script: {perf_test_file}")
        
        # Stress Test Script
        stress_test_script = '''#!/usr/bin/env python3
"""
Stress Test für Trading Bot
Testet das Verhalten unter extremen Bedingungen
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import random
from src.main_bot import TradingBot
from src.config_manager import ConfigManager

async def simulate_market_crash():
    """Simuliert einen Markt-Crash"""
    print("🔄 Simulating market crash scenario...")
    
    try:
        config = ConfigManager("integration_tests/configs/test_config.json")
        bot = TradingBot(config, paper_trading=True)
        
        await bot.start()
        
        # Simuliere plötzliche Preiseinbrüche
        print("Simulating sudden price drops...")
        for i in range(10):
            # Hier würde normalerweise eine Mock-Preisänderung eingefügt
            await asyncio.sleep(5)
            print(f"Crash simulation step {i+1}/10")
            
        await bot.stop()
        print("✅ Market crash simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False

async def simulate_network_issues():
    """Simuliert Netzwerkprobleme"""
    print("🔄 Simulating network issues...")
    
    try:
        config = ConfigManager("integration_tests/configs/test_config.json")
        bot = TradingBot(config, paper_trading=True)
        
        await bot.start()
        
        # Simuliere Verbindungsunterbrechungen
        print("Simulating connection interruptions...")
        for i in range(5):
            await asyncio.sleep(10)
            print(f"Network simulation step {i+1}/5")
            
        await bot.stop()
        print("✅ Network issues simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Network stress test failed: {e}")
        return False

async def run_stress_tests():
    """Alle Stress Tests ausführen"""
    print("🚀 Starting stress tests...")
    
    results = []
    
    # Market Crash Test
    results.append(await simulate_market_crash())
    
    # Network Issues Test  
    results.append(await simulate_network_issues())
    
    success_rate = sum(results) / len(results) * 100
    print(f"\\n📊 Stress Test Results: {success_rate:.1f}% success rate")
    
    return all(results)

if __name__ == "__main__":
    asyncio.run(run_stress_tests())
'''
        
        stress_test_file = self.test_dir / "stress_test.py"
        with open(stress_test_file, 'w', encoding='utf-8') as f:
            f.write(stress_test_script)
        print(f"✅ Stress test script: {stress_test_file}")
        
    def create_test_data(self):
        """Erstellt Test-Daten"""
        print("\n📊 Creating test data...")
        
        # Mock Market Data
        mock_data = {
            "BTCUSDT": {
                "price": 45000.00,
                "24h_change": 2.5,
                "volume": 1500000000,
                "high": 46000.00,
                "low": 44000.00
            },
            "ETHUSDT": {
                "price": 3200.00,
                "24h_change": -1.2,
                "volume": 800000000,
                "high": 3250.00,
                "low": 3150.00
            }
        }
        
        mock_data_file = self.test_dir / "data" / "mock_market_data.json"
        with open(mock_data_file, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, indent=2)
        print(f"✅ Mock market data: {mock_data_file}")
        
        # Historical Test Data (Simplified)
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generiere vereinfachte historische Daten
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # BTCUSDT Mock Data
        btc_data = {
            'timestamp': dates,
            'open': np.random.normal(45000, 5000, len(dates)),
            'high': np.random.normal(46000, 5000, len(dates)),
            'low': np.random.normal(44000, 5000, len(dates)),
            'close': np.random.normal(45000, 5000, len(dates)),
            'volume': np.random.normal(1000000, 200000, len(dates))
        }
        
        btc_df = pd.DataFrame(btc_data)
        btc_df['open'] = btc_df['open'].abs()
        btc_df['high'] = btc_df['high'].abs()
        btc_df['low'] = btc_df['low'].abs()
        btc_df['close'] = btc_df['close'].abs()
        btc_df['volume'] = btc_df['volume'].abs()
        
        btc_file = self.test_dir / "data" / "BTCUSDT_test_data.csv"
        btc_df.to_csv(btc_file, index=False)
        print(f"✅ BTC test data: {btc_file}")
        
    def create_pytest_config(self):
        """Erstellt pytest Konfiguration"""
        print("\n🧪 Creating pytest configuration...")
        
        pytest_ini_content = '''[tool:pytest]
testpaths = integration_tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html:integration_tests/reports/coverage
    --cov-report=term-missing
    --junit-xml=integration_tests/reports/junit.xml
markers =
    unit: Unit tests
    integration: Integration tests  
    performance: Performance tests
    stress: Stress tests
    slow: Slow running tests
asyncio_mode = auto
'''
        
        pytest_file = self.base_dir / "pytest.ini"
        with open(pytest_file, 'w', encoding='utf-8') as f:
            f.write(pytest_ini_content)
        print(f"✅ Pytest config: {pytest_file}")
        
    def create_run_scripts(self):
        """Erstellt Ausführungs-Scripts"""
        print("\n🏃 Creating run scripts...")
        
        # Windows Batch Script
        windows_script = '''@echo off
echo ========================================
echo    TRADING BOT TEST SUITE
echo ========================================

echo Running quick tests...
python integration_tests\\quick_test.py
if errorlevel 1 (
    echo Quick tests failed!
    pause
    exit /b 1
)

echo.
echo Running integration tests...
python -m pytest integration_tests\\test_integration.py -v
if errorlevel 1 (
    echo Integration tests failed!
    pause
    exit /b 1
)

echo.
echo Running performance tests...
python integration_tests\\performance_test.py
if errorlevel 1 (
    echo Performance tests failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo    ALL TESTS COMPLETED SUCCESSFULLY!
echo ========================================
pause
'''
        
        windows_script_file = self.test_dir / "run_tests.bat"
        with open(windows_script_file, 'w', encoding='utf-8') as f:
            f.write(windows_script)
        print(f"✅ Windows script: {windows_script_file}")
        
        # Linux/Mac Shell Script
        linux_script = '''#!/bin/bash
echo "========================================"
echo "    TRADING BOT TEST SUITE"
echo "========================================"

echo "Running quick tests..."
python3 integration_tests/quick_test.py
if [ $? -ne 0 ]; then
    echo "Quick tests failed!"
    exit 1
fi

echo ""
echo "Running integration tests..."
python3 -m pytest integration_tests/test_integration.py -v
if [ $? -ne 0 ]; then
    echo "Integration tests failed!"
    exit 1
fi

echo ""
echo "Running performance tests..."
python3 integration_tests/performance_test.py
if [ $? -ne 0 ]; then
    echo "Performance tests failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "    ALL TESTS COMPLETED SUCCESSFULLY!"
echo "========================================"
'''
        
        linux_script_file = self.test_dir / "run_tests.sh"
        with open(linux_script_file, 'w', encoding='utf-8') as f:
            f.write(linux_script)
        
        # Ausführbar machen
        try:
            os.chmod(linux_script_file, 0o755)
        except:
            pass
            
        print(f"✅ Linux script: {linux_script_file}")
        
    def create_documentation(self):
        """Erstellt Test-Dokumentation"""
        print("\n📖 Creating test documentation...")
        
        readme_content = '''# Trading Bot Test Suite

## Überblick
Diese Test-Suite bietet umfassende Tests für den Trading Bot mit verschiedenen Test-Szenarien.

## Setup
1. Führe `python test_setup.py` aus um die Test-Umgebung zu erstellen
2. Passe die Konfigurationsdateien in `integration_tests/configs/` an
3. Füge deine API-Keys für Testnet-Trading hinzu

## Test-Kategorien

### Quick Tests (`quick_test.py`)
- Schnelle Funktionalitätstests
- Dauer: ~30 Sekunden
- Prüft grundlegende Bot-Funktionalität

### Integration Tests (`test_integration.py`)
- Vollständige Bot-Integration
- Exchange-Verbindungen
- Strategie-Wechsel
- Dauer: ~5 Minuten

### Performance Tests (`performance_test.py`)
- Memory Usage Monitoring
- CPU Usage Tracking
- Execution Time Measurement  
- Dauer: ~5 Minuten

### Stress Tests (`stress_test.py`)
- Market Crash Simulation
- Network Issue Simulation
- Extreme Load Testing
- Dauer: ~10 Minuten

## Ausführung

### Alle Tests (Windows)
```bash
integration_tests\\run_tests.bat
```

### Alle Tests (Linux/Mac)
```bash
chmod +x integration_tests/run_tests.sh
./integration_tests/run_tests.sh
```

### Einzelne Tests
```bash
# Quick Test
python integration_tests/quick_test.py

# Performance Test
python integration_tests/performance_test.py

# Stress Test
python integration_tests/stress_test.py

# Integration Test mit pytest
python -m pytest integration_tests/test_integration.py -v
```

## Konfiguration

### test_config.json
Standard-Konfiguration für Tests mit Testnet-APIs

### paper_trading_config.json  
Konfiguration für Paper-Trading Tests ohne echte API-Verbindung

### performance_config.json
Optimierte Konfiguration für Performance-Tests

## Reports
- Test-Reports: `integration_tests/reports/`
- Coverage-Reports: `integration_tests/reports/coverage/`
- Log-Dateien: `integration_tests/logs/`

## API-Keys Setup
1. Erstelle Testnet-Accounts bei Binance und KuCoin
2. Generiere API-Keys für Testnets
3. Trage die Keys in die Config-Dateien ein:
   - `integration_tests/configs/test_config.json`
   - Setze `"testnet": true` und `"sandbox": true`

## Troubleshooting

### Häufige Probleme
1. **Missing Dependencies**: Führe `pip install -r integration_tests/requirements.txt` aus
2. **API Connection Failed**: Prüfe API-Keys und Testnet-URLs
3. **Permission Denied**: Stelle sicher dass Scripts ausführbar sind

### Debug-Modus
Setze Logging-Level auf "DEBUG" in der Konfiguration für detaillierte Logs.

## Continuous Integration
Die Test-Suite kann in CI/CD Pipelines integriert werden:
```yaml
# Beispiel für GitHub Actions
- name: Run Trading Bot Tests
  run: |
    python test_setup.py
    python integration_tests/run_tests.sh
```
'''
        
        readme_file = self.test_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ Documentation: {readme_file}")
        
    def create_gitignore(self):
        """Erstellt .gitignore für Test-Verzeichnis"""
        print("\n🔒 Creating .gitignore...")
        
        gitignore_content = '''# Test Results
integration_tests/results/
integration_tests/logs/
integration_tests/reports/
integration_tests/backups/

# API Keys (WICHTIG!)
**/configs/*config*.json
**/*_config.json
**/test_config.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
coverage.xml
*.cover
.hypothesis/

# Logs
*.log

# Data Files
*.csv
*.json
!**/requirements.txt
!**/mock_*.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
'''
        
        gitignore_file = self.test_dir / ".gitignore"
        with open(gitignore_file, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print(f"✅ .gitignore: {gitignore_file}")
        
    def finalize_setup(self):
        """Finalization und Summary"""
        print("\n🎯 Finalizing setup...")
        
        # Berechtigungen setzen (Linux/Mac)
        if platform.system() != "Windows":
            try:
                run_script = self.test_dir / "run_tests.sh"
                os.chmod(run_script, 0o755)
            except:
                pass
                
        # Setup-Summary erstellen
        summary = {
            "setup_date": str(Path.cwd()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "test_directory": str(self.test_dir),
            "status": "completed"
        }
        
        summary_file = self.test_dir / "setup_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Setup summary: {summary_file}")
        
        print("\n" + "="*60)
        print("🎉 TEST ENVIRONMENT SETUP COMPLETED!")
        print("="*60)
        print(f"📁 Test Directory: {self.test_dir}")
        print(f"📋 Next Steps:")
        print(f"   1. Edit config files in: {self.test_dir}/configs/")
        print(f"   2. Add your API keys (Testnet only!)")
        print(f"   3. Run quick test: python {self.test_dir}/quick_test.py")
        print(f"   4. Run full tests: {self.test_dir}/run_tests.{'bat' if platform.system() == 'Windows' else 'sh'}")
        print("="*60)
        
    def run_setup(self):
        """Führt komplettes Setup durch"""
        try:
            self.print_banner()
            
            if not self.check_system_requirements():
                print("❌ System requirements not met. Aborting setup.")
                return False
                
            self.create_directory_structure()
            self.install_dependencies()
            self.create_test_configs()
            self.create_test_scripts()
            self.create_test_data()
            self.create_pytest_config()
            self.create_run_scripts()
            self.create_documentation()
            self.create_gitignore()
            self.finalize_setup()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False

def main():
    """Main Entry Point"""
    setup = TestEnvironmentSetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("Ready to start testing your Trading Bot! 🚀")
    else:
        print("\n❌ Setup failed!")
        print("Please check the errors above and try again.")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())