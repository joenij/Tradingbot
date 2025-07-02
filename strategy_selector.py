"""
Strategy Selector Module
Intelligente Auswahl der besten Trading-Strategie basierend auf:
- Aktuelle Marktsituation
- Performance-Metriken der Strategien
- Backtest-Ergebnisse
- Machine Learning Vorhersagen
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from pathlib import Path
import threading
import time

from config_manager import ConfigManager
from logger import TradingLogger
from market_analyzer import MarketAnalyzer
from backtester import Backtester


class MarketCondition(Enum):
    """Marktbedingungen für Strategieauswahl"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class StrategyType(Enum):
    """Verfügbare Trading-Strategien"""
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"


@dataclass
class StrategyPerformance:
    """Performance-Metriken einer Strategie"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    volatility: float
    trades_count: int
    profit_factor: float
    last_updated: datetime
    market_conditions: List[MarketCondition]
    confidence_score: float = 0.0


@dataclass
class StrategyRecommendation:
    """Empfehlung für Strategiewechsel"""
    recommended_strategy: StrategyType
    current_strategy: StrategyType
    confidence: float
    reason: str
    market_condition: MarketCondition
    expected_performance: Dict[str, float]
    switch_threshold_met: bool


class StrategySelector:
    """
    Intelligenter Strategy Selector
    Wählt die beste Strategie basierend auf aktueller Marktsituation
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = TradingLogger("StrategySelector")
        self.market_analyzer = MarketAnalyzer()
        self.backtester = Backtester()
        
        # Konfiguration
        self.selector_config = self.config.get('strategy_selector', {})
        self.min_confidence_threshold = self.selector_config.get('min_confidence_threshold', 0.7)
        self.performance_window_days = self.selector_config.get('performance_window_days', 30)
        self.rebalance_frequency_hours = self.selector_config.get('rebalance_frequency_hours', 6)
        self.min_trades_for_evaluation = self.selector_config.get('min_trades_for_evaluation', 10)
        
        # State Management
        self.current_strategy = StrategyType.UPTREND  # Default
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.last_selection_time = datetime.now()
        self.selection_history: List[Dict] = []
        
        # Threading
        self.running = False
        self.selector_thread = None
        self.lock = threading.Lock()
        
        # Persistenz
        self.data_dir = Path("data/strategy_selector")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.performance_file = self.data_dir / "strategy_performances.pkl"
        self.history_file = self.data_dir / "selection_history.json"
        
        # Strategien-Mapping
        self.strategy_mapping = {
            MarketCondition.UPTREND: StrategyType.UPTREND,
            MarketCondition.SIDEWAYS: StrategyType.SIDEWAYS,
            MarketCondition.DOWNTREND: StrategyType.DOWNTREND,
            MarketCondition.VOLATILE: StrategyType.SIDEWAYS,  # Grid Trading für volatile Märkte
            MarketCondition.UNKNOWN: StrategyType.UPTREND     # Default fallback
        }
        
        self._load_data()
        self.logger.info("StrategySelector initialisiert")
    
    def start(self):
        """Startet den Strategy Selector"""
        if self.running:
            self.logger.warning("StrategySelector läuft bereits")
            return
        
        self.running = True
        self.selector_thread = threading.Thread(target=self._selection_loop, daemon=True)
        self.selector_thread.start()
        self.logger.info("StrategySelector gestartet")
    
    def stop(self):
        """Stoppt den Strategy Selector"""
        self.running = False
        if self.selector_thread:
            self.selector_thread.join(timeout=10)
        self._save_data()
        self.logger.info("StrategySelector gestoppt")
    
    def _selection_loop(self):
        """Hauptschleife für Strategieauswahl"""
        while self.running:
            try:
                # Prüfe ob Rebalancing nötig ist
                if self._should_rebalance():
                    recommendation = self.get_strategy_recommendation()
                    
                    if recommendation.switch_threshold_met:
                        self._execute_strategy_switch(recommendation)
                
                # Performance-Update
                self._update_strategy_performances()
                
                # Speichere Daten periodisch
                self._save_data()
                
                time.sleep(300)  # 5 Minuten warten
                
            except Exception as e:
                self.logger.error(f"Fehler in selection_loop: {str(e)}")
                time.sleep(60)  # 1 Minute warten bei Fehler
    
    def get_strategy_recommendation(self) -> StrategyRecommendation:
        """
        Gibt Strategieempfehlung basierend auf aktueller Marktsituation
        """
        try:
            # Aktuelle Marktanalyse
            market_condition = self._analyze_current_market()
            
            # Beste Strategie für aktuelle Marktbedingung
            recommended_strategy = self._get_best_strategy_for_condition(market_condition)
            
            # Confidence Score berechnen
            confidence = self._calculate_confidence_score(recommended_strategy, market_condition)
            
            # Performance-Erwartung
            expected_performance = self._get_expected_performance(recommended_strategy, market_condition)
            
            # Switch-Schwelle prüfen
            switch_threshold_met = self._check_switch_threshold(recommended_strategy, confidence)
            
            # Begründung generieren
            reason = self._generate_recommendation_reason(
                recommended_strategy, market_condition, confidence
            )
            
            recommendation = StrategyRecommendation(
                recommended_strategy=recommended_strategy,
                current_strategy=self.current_strategy,
                confidence=confidence,
                reason=reason,
                market_condition=market_condition,
                expected_performance=expected_performance,
                switch_threshold_met=switch_threshold_met
            )
            
            self.logger.info(f"Strategieempfehlung: {recommendation.recommended_strategy.value} "
                           f"(Confidence: {confidence:.2f}, Market: {market_condition.value})")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Fehler bei Strategieempfehlung: {str(e)}")
            return self._get_fallback_recommendation()
    
    def _analyze_current_market(self) -> MarketCondition:
        """Analysiert aktuelle Marktbedingung"""
        try:
            # Marktdaten abrufen
            market_data = self.market_analyzer.get_current_market_state()
            
            if not market_data:
                return MarketCondition.UNKNOWN
            
            # Trend-Indikatoren
            trend_strength = market_data.get('trend_strength', 0)
            trend_direction = market_data.get('trend_direction', 0)
            volatility = market_data.get('volatility', 0)
            volume_trend = market_data.get('volume_trend', 0)
            
            # Marktbedingung bestimmen
            if abs(trend_strength) > 0.6 and volatility < 0.3:
                if trend_direction > 0:
                    return MarketCondition.UPTREND
                else:
                    return MarketCondition.DOWNTREND
            elif volatility > 0.5:
                return MarketCondition.VOLATILE
            elif abs(trend_strength) < 0.3:
                return MarketCondition.SIDEWAYS
            else:
                return MarketCondition.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"Fehler bei Marktanalyse: {str(e)}")
            return MarketCondition.UNKNOWN
    
    def _get_best_strategy_for_condition(self, condition: MarketCondition) -> StrategyType:
        """Ermittelt beste Strategie für Marktbedingung"""
        # Basis-Mapping
        base_strategy = self.strategy_mapping.get(condition, StrategyType.UPTREND)
        
        # Performance-basierte Verfeinerung
        if condition in [MarketCondition.UPTREND, MarketCondition.DOWNTREND, MarketCondition.SIDEWAYS]:
            condition_performances = self._get_performances_for_condition(condition)
            if condition_performances:
                # Beste Strategie basierend auf historischer Performance
                best_performance = max(condition_performances.items(), 
                                     key=lambda x: x[1].sharpe_ratio * x[1].win_rate)
                best_strategy_name = best_performance[0]
                
                # Mapping zurück zu StrategyType
                for strategy_type in StrategyType:
                    if strategy_type.value == best_strategy_name:
                        return strategy_type
        
        return base_strategy
    
    def _calculate_confidence_score(self, strategy: StrategyType, condition: MarketCondition) -> float:
        """Berechnet Confidence Score für Strategieempfehlung"""
        try:
            confidence_factors = []
            
            # 1. Historische Performance der Strategie unter dieser Marktbedingung
            performance = self.strategy_performances.get(strategy.value)
            if performance and condition in performance.market_conditions:
                perf_score = min(1.0, (performance.sharpe_ratio + 2) / 4)  # Normalisiert auf 0-1
                confidence_factors.append(perf_score * 0.4)
            else:
                confidence_factors.append(0.2)  # Niedrige Confidence ohne historische Daten
            
            # 2. Anzahl verfügbarer Trades für Bewertung
            if performance:
                trade_score = min(1.0, performance.trades_count / self.min_trades_for_evaluation)
                confidence_factors.append(trade_score * 0.2)
            else:
                confidence_factors.append(0.1)
            
            # 3. Aktualität der Performance-Daten
            if performance:
                days_old = (datetime.now() - performance.last_updated).days
                recency_score = max(0.1, 1.0 - (days_old / 30))  # Decay über 30 Tage
                confidence_factors.append(recency_score * 0.2)
            else:
                confidence_factors.append(0.1)
            
            # 4. Klarheit der Marktbedingung
            market_clarity = self._get_market_clarity_score(condition)
            confidence_factors.append(market_clarity * 0.2)
            
            return sum(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Confidence-Berechnung: {str(e)}")
            return 0.5  # Mittlere Confidence als Fallback
    
    def _get_market_clarity_score(self, condition: MarketCondition) -> float:
        """Bewertet Klarheit der Marktbedingung"""
        try:
            market_data = self.market_analyzer.get_current_market_state()
            if not market_data:
                return 0.3
            
            trend_strength = abs(market_data.get('trend_strength', 0))
            volatility = market_data.get('volatility', 0)
            
            if condition == MarketCondition.UPTREND or condition == MarketCondition.DOWNTREND:
                # Starker Trend = hohe Clarity
                return min(1.0, trend_strength * 1.5)
            elif condition == MarketCondition.SIDEWAYS:
                # Niedrige Volatility = hohe Clarity für Sideways
                return min(1.0, 1.0 - volatility)
            elif condition == MarketCondition.VOLATILE:
                # Hohe Volatility = hohe Clarity für Volatile
                return min(1.0, volatility * 1.2)
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Fehler bei Market Clarity Score: {str(e)}")
            return 0.3
    
    def _get_expected_performance(self, strategy: StrategyType, condition: MarketCondition) -> Dict[str, float]:
        """Berechnet erwartete Performance-Metriken"""
        performance = self.strategy_performances.get(strategy.value)
        
        if performance and condition in performance.market_conditions:
            return {
                'expected_return': performance.total_return,
                'expected_sharpe': performance.sharpe_ratio,
                'expected_drawdown': performance.max_drawdown,
                'expected_win_rate': performance.win_rate
            }
        else:
            # Default-Erwartungen basierend auf Strategietyp
            defaults = {
                StrategyType.UPTREND: {
                    'expected_return': 0.15,
                    'expected_sharpe': 1.2,
                    'expected_drawdown': -0.08,
                    'expected_win_rate': 0.55
                },
                StrategyType.SIDEWAYS: {
                    'expected_return': 0.08,
                    'expected_sharpe': 1.5,
                    'expected_drawdown': -0.05,
                    'expected_win_rate': 0.65
                },
                StrategyType.DOWNTREND: {
                    'expected_return': 0.05,
                    'expected_sharpe': 0.8,
                    'expected_drawdown': -0.12,
                    'expected_win_rate': 0.45
                }
            }
            return defaults.get(strategy, defaults[StrategyType.UPTREND])
    
    def _check_switch_threshold(self, recommended_strategy: StrategyType, confidence: float) -> bool:
        """Prüft ob Switch-Schwelle erreicht ist"""
        # Kein Switch wenn bereits die empfohlene Strategie läuft
        if recommended_strategy == self.current_strategy:
            return False
        
        # Confidence muss über Schwelle liegen
        if confidence < self.min_confidence_threshold:
            return False
        
        # Zeitbasierte Switches vermeiden (min. 1 Stunde zwischen Switches)
        time_since_last_switch = datetime.now() - self.last_selection_time
        if time_since_last_switch < timedelta(hours=1):
            return False
        
        # Performance-basierte Prüfung
        current_performance = self.strategy_performances.get(self.current_strategy.value)
        recommended_performance = self.strategy_performances.get(recommended_strategy.value)
        
        if current_performance and recommended_performance:
            # Switch nur wenn empfohlene Strategie signifikant besser ist
            performance_improvement = (
                recommended_performance.sharpe_ratio - current_performance.sharpe_ratio
            )
            if performance_improvement < 0.2:  # Mindest-Verbesserung
                return False
        
        return True
    
    def _generate_recommendation_reason(self, strategy: StrategyType, 
                                      condition: MarketCondition, confidence: float) -> str:
        """Generiert Begründung für Strategieempfehlung"""
        reasons = []
        
        # Marktbedingung
        condition_reasons = {
            MarketCondition.UPTREND: "Starker Aufwärtstrend erkannt",
            MarketCondition.DOWNTREND: "Abwärtstrend mit Verkaufssignalen",
            MarketCondition.SIDEWAYS: "Seitwärtsbewegung ideal für Grid Trading",
            MarketCondition.VOLATILE: "Hohe Volatilität begünstigt Grid-Strategien",
            MarketCondition.UNKNOWN: "Unklare Marktlage erfordert konservative Strategie"
        }
        reasons.append(condition_reasons.get(condition, "Marktanalyse durchgeführt"))
        
        # Performance
        performance = self.strategy_performances.get(strategy.value)
        if performance:
            if performance.sharpe_ratio > 1.0:
                reasons.append(f"Strategie zeigt starke Sharpe Ratio ({performance.sharpe_ratio:.2f})")
            if performance.win_rate > 0.6:
                reasons.append(f"Hohe Gewinnrate ({performance.win_rate:.1%})")
        
        # Confidence
        if confidence > 0.8:
            reasons.append("Hohe Confidence in Vorhersage")
        elif confidence > 0.6:
            reasons.append("Moderate Confidence in Vorhersage")
        
        return "; ".join(reasons)
    
    def _execute_strategy_switch(self, recommendation: StrategyRecommendation):
        """Führt Strategiewechsel durch"""
        try:
            with self.lock:
                old_strategy = self.current_strategy
                self.current_strategy = recommendation.recommended_strategy
                self.last_selection_time = datetime.now()
                
                # History speichern
                switch_record = {
                    'timestamp': datetime.now().isoformat(),
                    'from_strategy': old_strategy.value,
                    'to_strategy': recommendation.recommended_strategy.value,
                    'market_condition': recommendation.market_condition.value,
                    'confidence': recommendation.confidence,
                    'reason': recommendation.reason
                }
                self.selection_history.append(switch_record)
                
                # Log
                self.logger.info(f"Strategiewechsel: {old_strategy.value} → "
                               f"{recommendation.recommended_strategy.value} "
                               f"(Confidence: {recommendation.confidence:.2f})")
                self.logger.info(f"Grund: {recommendation.reason}")
                
                # Event für andere Module (falls implementiert)
                self._notify_strategy_change(old_strategy, recommendation.recommended_strategy)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Strategiewechsel: {str(e)}")
    
    def _notify_strategy_change(self, old_strategy: StrategyType, new_strategy: StrategyType):
        """Benachrichtigt andere Module über Strategiewechsel"""
        # Placeholder für Event-System
        # Könnte später für Integration mit anderen Modulen verwendet werden
        pass
    
    def update_strategy_performance(self, strategy_name: str, performance_data: Dict[str, Any]):
        """Aktualisiert Performance-Daten einer Strategie"""
        try:
            with self.lock:
                # Marktbedingungen aus Performance-Daten extrahieren
                market_conditions = performance_data.get('market_conditions', [MarketCondition.UNKNOWN])
                if isinstance(market_conditions, str):
                    market_conditions = [MarketCondition(market_conditions)]
                
                performance = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_return=performance_data.get('total_return', 0.0),
                    sharpe_ratio=performance_data.get('sharpe_ratio', 0.0),
                    max_drawdown=performance_data.get('max_drawdown', 0.0),
                    win_rate=performance_data.get('win_rate', 0.0),
                    avg_trade_duration=performance_data.get('avg_trade_duration', 0.0),
                    volatility=performance_data.get('volatility', 0.0),
                    trades_count=performance_data.get('trades_count', 0),
                    profit_factor=performance_data.get('profit_factor', 1.0),
                    last_updated=datetime.now(),
                    market_conditions=market_conditions
                )
                
                self.strategy_performances[strategy_name] = performance
                self.logger.info(f"Performance aktualisiert für Strategie: {strategy_name}")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Update der Performance: {str(e)}")
    
    def get_current_strategy(self) -> StrategyType:
        """Gibt aktuelle Strategie zurück"""
        return self.current_strategy
    
    def get_strategy_performances(self) -> Dict[str, StrategyPerformance]:
        """Gibt alle Strategy Performances zurück"""
        return self.strategy_performances.copy()
    
    def get_selection_history(self) -> List[Dict]:
        """Gibt History der Strategiewechsel zurück"""
        return self.selection_history.copy()
    
    def force_strategy_switch(self, strategy: StrategyType, reason: str = "Manual override"):
        """Erzwingt Strategiewechsel (für manuellen Eingriff)"""
        try:
            with self.lock:
                old_strategy = self.current_strategy
                self.current_strategy = strategy
                self.last_selection_time = datetime.now()
                
                switch_record = {
                    'timestamp': datetime.now().isoformat(),
                    'from_strategy': old_strategy.value,
                    'to_strategy': strategy.value,
                    'market_condition': 'manual',
                    'confidence': 1.0,
                    'reason': reason
                }
                self.selection_history.append(switch_record)
                
                self.logger.info(f"Manueller Strategiewechsel: {old_strategy.value} → {strategy.value}")
                
        except Exception as e:
            self.logger.error(f"Fehler beim manuellen Strategiewechsel: {str(e)}")
    
    def _should_rebalance(self) -> bool:
        """Prüft ob Rebalancing durchgeführt werden sollte"""
        time_since_last = datetime.now() - self.last_selection_time
        return time_since_last >= timedelta(hours=self.rebalance_frequency_hours)
    
    def _update_strategy_performances(self):
        """Aktualisiert Performance-Daten aller Strategien"""
        try:
            # Hier würde normalerweise die Integration mit dem Backtester erfolgen
            # Für jede Strategie würden aktuelle Performance-Daten abgerufen
            
            for strategy_type in StrategyType:
                strategy_name = strategy_type.value
                if strategy_name not in self.strategy_performances:
                    # Initialisiere mit Dummy-Daten falls nicht vorhanden
                    self._initialize_strategy_performance(strategy_name)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Performance-Update: {str(e)}")
    
    def _initialize_strategy_performance(self, strategy_name: str):
        """Initialisiert Performance-Daten für neue Strategie"""
        default_performance = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.5,
            'avg_trade_duration': 24.0,
            'volatility': 0.2,
            'trades_count': 0,
            'profit_factor': 1.0,
            'market_conditions': [MarketCondition.UNKNOWN]
        }
        self.update_strategy_performance(strategy_name, default_performance)
    
    def _get_performances_for_condition(self, condition: MarketCondition) -> Dict[str, StrategyPerformance]:
        """Gibt Performances für spezifische Marktbedingung zurück"""
        relevant_performances = {}
        for name, performance in self.strategy_performances.items():
            if condition in performance.market_conditions:
                relevant_performances[name] = performance
        return relevant_performances
    
    def _get_fallback_recommendation(self) -> StrategyRecommendation:
        """Fallback-Empfehlung bei Fehlern"""
        return StrategyRecommendation(
            recommended_strategy=StrategyType.UPTREND,
            current_strategy=self.current_strategy,
            confidence=0.3,
            reason="Fallback aufgrund Analysefehler",
            market_condition=MarketCondition.UNKNOWN,
            expected_performance={'expected_return': 0.05, 'expected_sharpe': 0.5, 
                                'expected_drawdown': -0.1, 'expected_win_rate': 0.5},
            switch_threshold_met=False
        )
    
    def _save_data(self):
        """Speichert Performance-Daten und History"""
        try:
            # Strategy Performances
            with open(self.performance_file, 'wb') as f:
                pickle.dump(self.strategy_performances, f)
            
            # Selection History
            with open(self.history_file, 'w') as f:
                json.dump(self.selection_history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Daten: {str(e)}")
    
    def _load_data(self):
        """Lädt gespeicherte Daten"""
        try:
            # Strategy Performances
            if self.performance_file.exists():
                with open(self.performance_file, 'rb') as f:
                    self.strategy_performances = pickle.load(f)
            
            # Selection History
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    self.selection_history = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Daten: {str(e)}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Gibt detaillierten Status-Report zurück"""
        try:
            current_market = self._analyze_current_market()
            recommendation = self.get_strategy_recommendation()
            
            return {
                'current_strategy': self.current_strategy.value,
                'current_market_condition': current_market.value,
                'recommendation': {
                    'strategy': recommendation.recommended_strategy.value,
                    'confidence': recommendation.confidence,
                    'reason': recommendation.reason,
                    'switch_recommended': recommendation.switch_threshold_met
                },
                'strategy_performances': {
                    name: {
                        'sharpe_ratio': perf.sharpe_ratio,
                        'win_rate': perf.win_rate,
                        'total_return': perf.total_return,
                        'trades_count': perf.trades_count,
                        'last_updated': perf.last_updated.isoformat()
                    }
                    for name, perf in self.strategy_performances.items()
                },
                'last_selection_time': self.last_selection_time.isoformat(),
                'total_switches': len(self.selection_history),
                'running': self.running
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Status-Report: {str(e)}")
            return {'error': str(e)}


# Utility Functions
def create_mock_performance_data() -> Dict[str, Dict[str, Any]]:
    """Erstellt Mock-Performance-Daten für Testing"""
    return {
        'uptrend': {
            'total_return': 0.18,
            'sharpe_ratio': 1.4,
            'max_drawdown': -0.08,
            'win_rate': 0.58,
            'avg_trade_duration': 36.0,
            'volatility': 0.15,
            'trades_count': 45,
            'profit_factor': 1.6,
            'market_conditions': [MarketCondition.UPTREND]
        },
        'sideways': {
            'total_return': 0.12,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.04,
            'win_rate': 0.72,
            'avg_trade_duration': 18.0,
            'volatility': 0.08,
            'trades_count': 120,
            'profit_factor': 2.1,
            'market_conditions': [MarketCondition.SIDEWAYS]
        },
        'downtrend': {
            'total_return': 0.08,
            'sharpe_ratio': 0.9,
            'max_drawdown': -0.15,
            'win_rate': 0.42,
            'avg_trade_duration': 48.0,
            'volatility': 0.22,
            'trades_count': 28,
            'profit_factor': 1.2,
            'market_conditions': [MarketCondition.DOWNTREND]
        }
    }


if __name__ == "__main__":
    # Test des Strategy Selectors
    selector = StrategySelector()
    
    # Mock-Daten laden
    mock_data = create_mock_performance_data()
    for strategy, data in mock_data.items():
        selector.update_strategy_performance(strategy, data)
    
    # Test der Strategieempfehlung
    recommendation = selector.get_strategy_recommendation()
    print(f"Empfohlene Strategie: {recommendation.recommended_strategy.value}")
    print(f"Confidence: {recommendation.confidence:.2f}")
    print(f"Grund: {recommendation.reason}")
    
    # Status-Report
    status = selector.get_status_report()
    print("\nStatus Report:")
    print(json.dumps(status, indent=2, default=str))