#!/usr/bin/env python3
"""
Automatisierter Test-Runner für Trading Bot Integration
Führt alle Tests aus und generiert detaillierte Berichte
"""

import os
import sys
import json
import time
import subprocess
import platform
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import logging
from typing import Dict, List, Any
import asyncio
import signal

class TestRunner:
    """Hauptklasse für automatisierte Tests"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.test_results = {}
        self.system_info = self.get_system_info()
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Test-Verzeichnisse erstellen"""
        self.base_dir = Path("integration_tests")
        self.base_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        
        for dir_path in [self.results_dir, self.logs_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def setup_logging(self):
        """Logging für Test-Runner einrichten"""
        log_file = self.logs_dir / f"test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_system_info(self) -> Dict[str, Any]:
        """System-Informationen sammeln"""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_free': shutil.disk_usage('.').free,
            'timestamp': datetime.now().isoformat()
        }
        
    def check_prerequisites(self) -> bool:
        """Prüft ob alle Voraussetzungen erfüllt sind"""
        self.logger.info("Checking prerequisites...")
        
        # Python-Module prüfen
        required_modules = [
            'pandas', 'numpy', 'scikit-learn', 'ccxt', 
            'asyncio', 'aiohttp', 'pytest', 'unittest'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
                
        if missing_modules:
            self.logger.error(f"Missing required modules: {missing_modules}")
            return False
            
        # Bot-Module prüfen
        required_bot_files = [
            'config_manager.py', 'logger.py', 'data_manager.py',
            'exchange_connector.py', 'market_analyzer.py', 'ml_trainer.py',
            'strategy_uptrend.py', 'strategy_sideways.py', 'strategy_downtrend.py',
            'backtester.py', 'strategy_selector.py', 'position_manager.py',
            'risk_manager.py', 'notification_system.py', 'main_bot.py'
        ]
        
        missing_files = []
        for file_name in required_bot_files:
            if not Path(file_name).exists():
                missing_files.append(file_name)
                
        if missing_files:
            self.logger.error(f"Missing bot files: {missing_files}")
            return False
            
        self.logger.info("✓ All prerequisites met")
        return True
        
    def run_unit_tests(self) -> Dict[str, Any]:
        """Führt Unit-Tests aus"""
        self.logger.info("Running unit tests...")
        
        try:
            # Integration Test Suite ausführen
            result = subprocess.run([
                sys.executable, 'integration_test_suite.py'
            ], capture_output=True, text=True, timeout=300)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': 0  # TODO: Measure actual duration
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error("Unit tests timed out")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            self.logger.error(f"Unit tests failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def run_paper_trading_test(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """Führt Paper-Trading-Test aus"""
        self.logger.info(f"Running paper trading test for {duration_minutes} minutes...")
        
        try:
            # Bot im Paper-Trading-Modus starten
            process = subprocess.Popen([
                sys.executable, 'main_bot.py', 
                '--paper-trading', '--config', 'test_config.json'
            ])
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            # System-Metriken sammeln
            cpu_usage = []
            memory_usage = []
            
            while time.time() < end_time:
                if process.poll() is not None:
                    # Bot ist beendet
                    break
                    
                # Metriken sammeln
                try:
                    bot_process = psutil.Process(process.pid)
                    cpu_usage.append(bot_process.cpu_percent())
                    memory_usage.append(bot_process.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                    
                time.sleep(10)  # Alle 10 Sekunden messen
                
            # Bot beenden
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
                
            return {
                'success': True,
                'duration_minutes': duration_minutes,
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'max_memory_usage': max(memory_usage) if memory_usage else 0
            }
            
        except Exception as e:
            self.logger.error(f"Paper trading test failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def run_crash_recovery_test(self) -> Dict[str, Any]:
        """Testet Crash-Recovery-Mechanismus"""
        self.logger.info("Running crash recovery test...")
        
        try:
            # Bot starten
            process = subprocess.Popen([
                sys.executable, 'main_bot.py', 
                '--config', 'test_config.json'
            ])
            
            # 2 Minuten laufen lassen
            time.sleep(120)
            
            # State-Datei prüfen vor Crash
            state_file_exists_before = Path('bot_state.json').exists()
            
            # Bot "crashen" (SIGKILL)
            process.kill()
            process.wait()
            
            # Kurz warten
            time.sleep(5)
            
            # Bot neu starten
            recovery_process = subprocess.Popen([
                sys.executable, 'main_bot.py', 
                '--config', 'test_config.json'
            ])
            
            # 1 Minute laufen lassen für Recovery
            time.sleep(60)
            
            # Erfolgreich gestartet?
            recovery_success = recovery_process.poll() is None
            
            # Bot beenden
            if recovery_process.poll() is None:
                recovery_process.terminate()
                recovery_process.wait(timeout=10)
                
            return {
                'success': recovery_success,
                'state_file_existed': state_file_exists_before,
                'recovery_successful': recovery_success
            }
            
        except Exception as e:
            self.logger.error(f"Crash recovery test failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Führt Performance-Benchmark durch"""
        self.logger.info("Running performance benchmark...")
        
        try:
            # Bot für Benchmark starten
            process = subprocess.Popen([
                sys.executable, 'main_bot.py', 
                '--benchmark', '--config', 'test_config.json'
            ])
            
            # 10 Minuten Benchmark
            benchmark_duration = 600  # 10 Minuten
            start_time = time.time()
            
            cpu_readings = []
            memory_readings = []
            
            while time.time() - start_time < benchmark_duration:
                if process.poll() is not None:
                    break
                    
                try:
                    bot_process = psutil.Process(process.pid)
                    cpu_readings.append(bot_process.cpu_percent(interval=1))
                    memory_readings.append(bot_process.memory_info().rss / 1024 / 1024)  # MB
                except psutil.NoSuchProcess:
                    break
                    
            # Bot beenden
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
                
            return {
                'success': True,
                'duration_seconds': benchmark_duration,
                'avg_cpu_percent': sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0,
                'max_cpu_percent': max(cpu_readings) if cpu_readings else 0,
                'avg_memory_mb': sum(memory_readings) / len(memory_readings) if memory_readings else 0,
                'max_memory_mb': max(memory_readings) if memory_readings else 0,
                'cpu_stability': (max(cpu_readings) - min(cpu_readings)) if cpu_readings else 0
            }
            
        except Exception as e:
            self.logger.error(f"Performance benchmark failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def generate_test_report(self) -> str:
        """Generiert detaillierten Test-Report"""
        report_time = datetime.now()
        duration = report_time - self.start_time
        
        report = f"""
# Trading Bot Integration Test Report

**Datum**: {report_time.strftime('%Y-%m-%d %H:%M:%S')}
**Test-Dauer**: {duration}
**Version**: 1.0

## System-Information
- **Platform**: {self.system_info['platform']}
- **Python**: {self.system_info['python_version']}
- **CPU Kerne**: {self.system_info['cpu_count']}
- **RAM**: {self.system_info['memory_total'] / 1024**3:.1f} GB
- **Freier Speicher**: {self.system_info['disk_free'] / 1024**3:.1f} GB

## Test-Ergebnisse

### 1. Unit Tests
"""
        
        if 'unit_tests' in self.test_results:
            result = self.test_results['unit_tests']
            status = "✅ BESTANDEN" if result['success'] else "❌ FEHLGESCHLAGEN"
            report += f"**Status**: {status}\n\n"
            
            if not result['success']:
                report += f"**Fehler**: {result.get('error', 'Unbekannt')}\n\n"
        
        report += """
### 2. Paper Trading Test
"""
        
        if 'paper_trading' in self.test_results:
            result = self.test_results['paper_trading']
            status = "✅ BESTANDEN" if result['success'] else "❌ FEHLGESCHLAGEN"
            report += f"**Status**: {status}\n"
            
            if result['success']:
                report += f"- **Laufzeit**: {result['duration_minutes']} Minuten\n"
                report += f"- **Ø CPU-Auslastung**: {result['avg_cpu_usage']:.1f}%\n"
                report += f"- **Ø RAM-Verbrauch**: {result['avg_memory_usage'] / 1024**2:.0f} MB\n"
                report += f"- **Max RAM-Verbrauch**: {result['max_memory_usage'] / 1024**2:.0f} MB\n\n"
        
        report += """
### 3. Crash Recovery Test
"""
        
        if 'crash_recovery' in self.test_results:
            result = self.test_results['crash_recovery']
            status = "✅ BESTANDEN" if result['success'] else "❌ FEHLGESCHLAGEN"
            report += f"**Status**: {status}\n"
            report += f"- **State-Datei vorhanden**: {'Ja' if result.get('state_file_existed') else 'Nein'}\n"
            report += f"- **Recovery erfolgreich**: {'Ja' if result.get('recovery_successful') else 'Nein'}\n\n"
        
        report += """
### 4. Performance Benchmark
"""
        
        if 'performance' in self.test_results:
            result = self.test_results['performance']
            status = "✅ BESTANDEN" if result['success'] else "❌ FEHLGESCHLAGEN"
            report += f"**Status**: {status}\n"
            
            if result['success']:
                report += f"- **Benchmark-Dauer**: {result['duration_seconds']} Sekunden\n"
                report += f"- **Ø CPU**: {result['avg_cpu_percent']:.1f}%\n"
                report += f"- **Max CPU**: {result['max_cpu_percent']:.1f}%\n"
                report += f"- **Ø RAM**: {result['avg_memory_mb']:.0f} MB\n"
                report += f"- **Max RAM**: {result['max_memory_mb']:.0f} MB\n"
                report += f"- **CPU-Stabilität**: {result['cpu_stability']:.1f}%\n\n"
        
        # Gesamtbewertung
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        report += f"""
## Gesamtbewertung

**Tests bestanden**: {passed_tests}/{total_tests}
**Erfolgsrate**: {(passed_tests/total_tests*100):.1f}%

"""
        
        if passed_tests == total_tests:
            report += "🎉 **ALLE TESTS BESTANDEN!** Der Bot ist bereit für das Deployment.\n\n"
            report += """
### Nächste Schritte:
1. Konfiguriere echte API-Keys für Testnet
2. Führe 24h Paper-Trading durch
3. Starte mit kleinen Beträgen im Live-Trading
4. Überwache Performance kontinuierlich
"""
        else:
            report += "❌ **EINIGE TESTS FEHLGESCHLAGEN!** Bitte Probleme beheben vor Deployment.\n\n"
            report += """
### Empfohlene Maßnahmen:
1. Fehler-Logs analysieren
2. Fehlgeschlagene Module überarbeiten
3. Tests erneut durchführen
4. Bei Bedarf Support kontaktieren
"""
        
        return report
        
    def save_report(self, report: str):
        """Speichert Test-Report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"integration_test_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"Test report saved to: {report_file}")
        
        # Auch JSON-Version für maschinelle Auswertung
        json_file = self.reports_dir / f"integration_test_results_{timestamp}.json"
        results_json = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results.values() if r.get('success', False)),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds()
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
            
    async def run_all_tests(self):
        """Führt alle Tests sequenziell aus"""
        self.logger.info("Starting comprehensive integration test suite...")
        
        # 1. Voraussetzungen prüfen
        if not self.check_prerequisites():
            self.logger.error("Prerequisites not met. Aborting tests.")
            return False
            
        # 2. Unit Tests
        self.logger.info("Phase 1: Unit Tests")
        self.test_results['unit_tests'] = self.run_unit_tests()
        
        # 3. Paper Trading Test (wenn Unit Tests erfolgreich)
        if self.test_results['unit_tests']['success']:
            self.logger.info("Phase 2: Paper Trading Test")
            self.test_results['paper_trading'] = self.run_paper_trading_test(30)
        else:
            self.logger.warning("Skipping Paper Trading Test due to Unit Test failures")
            
        # 4. Crash Recovery Test
        self.logger.info("Phase 3: Crash Recovery Test")
        self.test_results['crash_recovery'] = self.run_crash_recovery_test()
        
        # 5. Performance Benchmark
        self.logger.info("Phase 4: Performance Benchmark")
        self.test_results['performance'] = self.run_performance_benchmark()
        
        # 6. Report generieren
        self.logger.info("Generating test report...")
        report = self.generate_test_report()
        self.save_report(report)
        
        # Report auch auf Console ausgeben
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Erfolg zurückgeben
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        return passed_tests == total_tests

def main():
    """Hauptfunktion"""
    print("Trading Bot Integration Test Runner")
    print("="*50)
    
    # Test-Runner initialisieren
    runner = TestRunner()
    
    try:
        # Alle Tests ausführen
        success = asyncio.run(runner.run_all_tests())
        
        if success:
            print("\n✅ ALL INTEGRATION TESTS PASSED!")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests abgebrochen durch Benutzer")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()