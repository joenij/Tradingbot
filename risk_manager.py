"""
Risk Manager für Trading Bot
Verwaltet Risiko, Position Sizing, Stop-Loss und Portfolio-Schutz
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Risk Metrics Datenklasse"""
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    max_position_size: float
    current_exposure: float
    risk_score: float
    
@dataclass
class PositionRisk:
    """Position Risk Datenklasse"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss_price: float
    risk_reward_ratio: float
    position_value: float

class RiskManager:
    def __init__(self, config_manager, logger_instance):
        """
        Initialisiert den Risk Manager
        
        Args:
            config_manager: ConfigManager Instanz
            logger_instance: Logger Instanz
        """
        self.config = config_manager
        self.logger = logger_instance
        
        # Risk Management Konfiguration
        self.risk_config = self.config.get_config().get('risk_management', {})
        
        # Basis Risk Parameter
        self.max_portfolio_risk = self.risk_config.get('max_portfolio_risk', 0.02)  # 2% max Portfolio Risk
        self.max_position_risk = self.risk_config.get('max_position_risk', 0.01)   # 1% max Position Risk
        self.max_daily_loss = self.risk_config.get('max_daily_loss', 0.05)         # 5% max täglicher Verlust
        self.max_drawdown_limit = self.risk_config.get('max_drawdown_limit', 0.15)  # 15% max Drawdown
        
        # Position Sizing Parameter
        self.min_position_size = self.risk_config.get('min_position_size', 10.0)
        self.max_position_size = self.risk_config.get('max_position_size', 1000.0)
        self.position_size_method = self.risk_config.get('position_size_method', 'fixed_risk')
        
        # Stop Loss Parameter
        self.default_stop_loss = self.risk_config.get('default_stop_loss', 0.02)  # 2%
        self.trailing_stop_enabled = self.risk_config.get('trailing_stop_enabled', True)
        self.trailing_stop_distance = self.risk_config.get('trailing_stop_distance', 0.015)  # 1.5%
        
        # Risk Tracking
        self.portfolio_value_history = []
        self.daily_pnl_history = []
        self.risk_metrics_history = []
        self.current_positions_risk = {}
        
        # Emergency Shutdown
        self.emergency_shutdown = False
        self.shutdown_reason = ""
        
        self.logger.info("Risk Manager initialisiert")
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float, portfolio_value: float,
                              strategy_confidence: float = 1.0) -> float:
        """
        Berechnet die optimale Positionsgröße basierend auf Risiko
        
        Args:
            symbol: Trading Symbol
            entry_price: Einstiegspreis
            stop_loss_price: Stop-Loss Preis
            portfolio_value: Aktueller Portfolio Wert
            strategy_confidence: Vertrauen in die Strategie (0-1)
            
        Returns:
            Positionsgröße in Quote-Währung
        """
        try:
            # Risk per Trade berechnen
            risk_per_trade = portfolio_value * self.max_position_risk
            
            # Price Risk berechnen
            price_risk = abs(entry_price - stop_loss_price) / entry_price
            
            if price_risk == 0:
                self.logger.warning(f"Price Risk ist 0 für {symbol}, verwende Default Stop Loss")
                price_risk = self.default_stop_loss
            
            # Position Size basierend auf Risk
            if self.position_size_method == 'fixed_risk':
                position_size = risk_per_trade / price_risk
            elif self.position_size_method == 'volatility_adjusted':
                # Volatilitäts-adjustierte Position Size (vereinfacht)
                volatility_factor = min(price_risk * 2, 0.1)  # Max 10% Volatility
                position_size = (risk_per_trade / volatility_factor) * strategy_confidence
            else:
                # Fixed Position Size
                position_size = portfolio_value * 0.1  # 10% des Portfolios
            
            # Confidence Adjustment
            position_size *= strategy_confidence
            
            # Min/Max Limits anwenden
            position_size = max(self.min_position_size, 
                              min(position_size, self.max_position_size))
            
            # Portfolio Exposure Check
            max_portfolio_exposure = portfolio_value * 0.8  # Max 80% des Portfolios
            position_size = min(position_size, max_portfolio_exposure)
            
            self.logger.info(f"Position Size für {symbol}: {position_size:.2f} "
                           f"(Risk: {price_risk:.4f}, Confidence: {strategy_confidence:.2f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Fehler bei Position Size Berechnung: {str(e)}")
            return self.min_position_size
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                          position_type: str, volatility: float = None) -> float:
        """
        Berechnet den Stop-Loss Preis
        
        Args:
            symbol: Trading Symbol
            entry_price: Einstiegspreis
            position_type: 'long' oder 'short'
            volatility: Aktuelle Volatilität (optional)
            
        Returns:
            Stop-Loss Preis
        """
        try:
            # Basis Stop Loss
            stop_loss_pct = self.default_stop_loss
            
            # Volatilitäts-Adjustment
            if volatility:
                # Höhere Volatilität = weitere Stop Loss
                volatility_multiplier = min(volatility / 0.02, 3.0)  # Max 3x Multiplier
                stop_loss_pct *= volatility_multiplier
            
            # Stop Loss Preis berechnen
            if position_type.lower() == 'long':
                stop_loss_price = entry_price * (1 - stop_loss_pct)
            else:  # short
                stop_loss_price = entry_price * (1 + stop_loss_pct)
            
            self.logger.info(f"Stop Loss für {symbol} {position_type}: {stop_loss_price:.6f} "
                           f"({stop_loss_pct:.2%})")
            
            return stop_loss_price
            
        except Exception as e:
            self.logger.error(f"Fehler bei Stop Loss Berechnung: {str(e)}")
            return entry_price * (0.98 if position_type.lower() == 'long' else 1.02)
    
    def update_trailing_stop(self, symbol: str, current_price: float, 
                           position_type: str, current_stop: float) -> float:
        """
        Aktualisiert Trailing Stop Loss
        
        Args:
            symbol: Trading Symbol
            current_price: Aktueller Preis
            position_type: 'long' oder 'short'
            current_stop: Aktueller Stop Loss
            
        Returns:
            Neuer Stop Loss Preis
        """
        try:
            if not self.trailing_stop_enabled:
                return current_stop
            
            if position_type.lower() == 'long':
                # Für Long: Stop Loss nur nach oben anpassen
                new_stop = current_price * (1 - self.trailing_stop_distance)
                if new_stop > current_stop:
                    self.logger.info(f"Trailing Stop für {symbol} Long: "
                                   f"{current_stop:.6f} -> {new_stop:.6f}")
                    return new_stop
            else:  # short
                # Für Short: Stop Loss nur nach unten anpassen
                new_stop = current_price * (1 + self.trailing_stop_distance)
                if new_stop < current_stop:
                    self.logger.info(f"Trailing Stop für {symbol} Short: "
                                   f"{current_stop:.6f} -> {new_stop:.6f}")
                    return new_stop
            
            return current_stop
            
        except Exception as e:
            self.logger.error(f"Fehler bei Trailing Stop Update: {str(e)}")
            return current_stop
    
    def assess_portfolio_risk(self, positions: List[Dict], 
                            portfolio_value: float) -> RiskMetrics:
        """
        Bewertet das gesamte Portfolio Risiko
        
        Args:
            positions: Liste aller offenen Positionen
            portfolio_value: Aktueller Portfolio Wert
            
        Returns:
            RiskMetrics Objekt
        """
        try:
            # Portfolio Value History aktualisieren
            self.portfolio_value_history.append({
                'timestamp': datetime.now(),
                'value': portfolio_value
            })
            
            # Nur letzte 100 Werte behalten
            if len(self.portfolio_value_history) > 100:
                self.portfolio_value_history = self.portfolio_value_history[-100:]
            
            # Current Exposure berechnen
            total_exposure = sum([pos.get('position_value', 0) for pos in positions])
            current_exposure = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Drawdown berechnen
            max_drawdown, current_drawdown = self._calculate_drawdown()
            
            # Sharpe Ratio berechnen
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Value at Risk (95%)
            var_95 = self._calculate_var()
            
            # Risk Score berechnen
            risk_score = self._calculate_risk_score(current_exposure, current_drawdown, var_95)
            
            # Max Position Size
            max_position_size = portfolio_value * self.max_position_risk
            
            risk_metrics = RiskMetrics(
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                max_position_size=max_position_size,
                current_exposure=current_exposure,
                risk_score=risk_score
            )
            
            # Risk Metrics History aktualisieren
            self.risk_metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': risk_metrics
            })
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Fehler bei Portfolio Risk Assessment: {str(e)}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def check_risk_limits(self, risk_metrics: RiskMetrics) -> Tuple[bool, List[str]]:
        """
        Überprüft Risk Limits und gibt Warnungen/Aktionen zurück
        
        Args:
            risk_metrics: RiskMetrics Objekt
            
        Returns:
            Tuple: (Trading erlaubt, Liste der Warnungen)
        """
        warnings = []
        trading_allowed = True
        
        try:
            # Drawdown Check
            if risk_metrics.current_drawdown > self.max_drawdown_limit:
                warnings.append(f"Max Drawdown überschritten: {risk_metrics.current_drawdown:.2%}")
                trading_allowed = False
            
            # Exposure Check
            if risk_metrics.current_exposure > 0.9:  # 90% Exposure Limit
                warnings.append(f"Hohe Portfolio Exposure: {risk_metrics.current_exposure:.2%}")
            
            # Daily Loss Check
            daily_loss = self._get_daily_pnl()
            if daily_loss < -self.max_daily_loss:
                warnings.append(f"Max täglicher Verlust überschritten: {daily_loss:.2%}")
                trading_allowed = False
            
            # Risk Score Check
            if risk_metrics.risk_score > 0.8:
                warnings.append(f"Hoher Risk Score: {risk_metrics.risk_score:.2f}")
                if risk_metrics.risk_score > 0.9:
                    trading_allowed = False
            
            # Emergency Shutdown Check
            if self.emergency_shutdown:
                warnings.append(f"Emergency Shutdown aktiv: {self.shutdown_reason}")
                trading_allowed = False
            
            if warnings:
                self.logger.warning(f"Risk Limits Check: {', '.join(warnings)}")
            
            return trading_allowed, warnings
            
        except Exception as e:
            self.logger.error(f"Fehler bei Risk Limits Check: {str(e)}")
            return False, [f"Fehler bei Risk Check: {str(e)}"]
    
    def should_close_position(self, position: Dict, current_price: float) -> Tuple[bool, str]:
        """
        Prüft ob eine Position geschlossen werden sollte
        
        Args:
            position: Position Dictionary
            current_price: Aktueller Preis
            
        Returns:
            Tuple: (Schließen, Grund)
        """
        try:
            symbol = position.get('symbol')
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss_price', 0)
            position_type = position.get('type', 'long')
            
            # Stop Loss Check
            if position_type.lower() == 'long' and current_price <= stop_loss:
                return True, f"Stop Loss erreicht: {current_price} <= {stop_loss}"
            elif position_type.lower() == 'short' and current_price >= stop_loss:
                return True, f"Stop Loss erreicht: {current_price} >= {stop_loss}"
            
            # Emergency Shutdown
            if self.emergency_shutdown:
                return True, f"Emergency Shutdown: {self.shutdown_reason}"
            
            # Position Age Check (z.B. max 7 Tage)
            if 'timestamp' in position:
                position_age = datetime.now() - position['timestamp']
                if position_age > timedelta(days=7):
                    return True, "Position zu alt (>7 Tage)"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Fehler bei Position Close Check: {str(e)}")
            return True, f"Fehler bei Check: {str(e)}"
    
    def trigger_emergency_shutdown(self, reason: str):
        """
        Löst Emergency Shutdown aus
        
        Args:
            reason: Grund für Shutdown
        """
        self.emergency_shutdown = True
        self.shutdown_reason = reason
        self.logger.critical(f"EMERGENCY SHUTDOWN ausgelöst: {reason}")
    
    def reset_emergency_shutdown(self):
        """Setzt Emergency Shutdown zurück"""
        self.emergency_shutdown = False
        self.shutdown_reason = ""
        self.logger.info("Emergency Shutdown zurückgesetzt")
    
    def get_position_risk_analysis(self, positions: List[Dict]) -> List[PositionRisk]:
        """
        Analysiert das Risiko aller Positionen
        
        Args:
            positions: Liste aller Positionen
            
        Returns:
            Liste von PositionRisk Objekten
        """
        position_risks = []
        
        try:
            for pos in positions:
                symbol = pos.get('symbol', '')
                position_size = pos.get('quantity', 0)
                entry_price = pos.get('entry_price', 0)
                current_price = pos.get('current_price', entry_price)
                stop_loss_price = pos.get('stop_loss_price', 0)
                
                # Unrealized PnL berechnen
                if pos.get('type', '').lower() == 'long':
                    unrealized_pnl = (current_price - entry_price) * position_size
                else:
                    unrealized_pnl = (entry_price - current_price) * position_size
                
                # Risk/Reward Ratio
                if stop_loss_price > 0:
                    risk = abs(entry_price - stop_loss_price)
                    # Simplified: Reward = 2x Risk
                    reward = risk * 2
                    risk_reward_ratio = reward / risk if risk > 0 else 0
                else:
                    risk_reward_ratio = 0
                
                position_value = position_size * current_price
                
                position_risk = PositionRisk(
                    symbol=symbol,
                    position_size=position_size,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    stop_loss_price=stop_loss_price,
                    risk_reward_ratio=risk_reward_ratio,
                    position_value=position_value
                )
                
                position_risks.append(position_risk)
            
            return position_risks
            
        except Exception as e:
            self.logger.error(f"Fehler bei Position Risk Analysis: {str(e)}")
            return []
    
    def _calculate_drawdown(self) -> Tuple[float, float]:
        """Berechnet Max und Current Drawdown"""
        if len(self.portfolio_value_history) < 2:
            return 0.0, 0.0
        
        values = [entry['value'] for entry in self.portfolio_value_history]
        peak = values[0]
        max_drawdown = 0.0
        current_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Current Drawdown
        current_value = values[-1]
        current_peak = max(values)
        current_drawdown = (current_peak - current_value) / current_peak
        
        return max_drawdown, current_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """Berechnet Sharpe Ratio"""
        if len(self.portfolio_value_history) < 10:
            return 0.0
        
        values = [entry['value'] for entry in self.portfolio_value_history]
        returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualisiert (vereinfacht)
        sharpe = (mean_return * 365) / (std_return * np.sqrt(365))
        return sharpe
    
    def _calculate_var(self) -> float:
        """Berechnet Value at Risk (95%)"""
        if len(self.portfolio_value_history) < 10:
            return 0.0
        
        values = [entry['value'] for entry in self.portfolio_value_history]
        returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
        
        if len(returns) < 2:
            return 0.0
        
        # 95% VaR
        var_95 = np.percentile(returns, 5)  # 5th percentile
        return abs(var_95)
    
    def _calculate_risk_score(self, exposure: float, drawdown: float, var: float) -> float:
        """Berechnet kombinierten Risk Score (0-1)"""
        # Gewichtete Kombination verschiedener Risk Faktoren
        exposure_score = min(exposure / 0.8, 1.0)  # Normalisiert auf 80% max exposure
        drawdown_score = min(drawdown / self.max_drawdown_limit, 1.0)
        var_score = min(var / 0.05, 1.0)  # Normalisiert auf 5% VaR
        
        # Gewichteter Score
        risk_score = (exposure_score * 0.3 + drawdown_score * 0.5 + var_score * 0.2)
        return min(risk_score, 1.0)
    
    def _get_daily_pnl(self) -> float:
        """Berechnet tägliches PnL"""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        # Vereinfacht: letzten 24 Stunden
        current_value = self.portfolio_value_history[-1]['value']
        
        # Suche nach Wert von vor ~24 Stunden
        day_ago = datetime.now() - timedelta(days=1)
        day_ago_value = current_value
        
        for entry in reversed(self.portfolio_value_history[:-1]):
            if entry['timestamp'] <= day_ago:
                day_ago_value = entry['value']
                break
        
        if day_ago_value == 0:
            return 0.0
        
        daily_return = (current_value - day_ago_value) / day_ago_value
        return daily_return
    
    def save_risk_state(self, filepath: str):
        """Speichert Risk Manager State"""
        try:
            state = {
                'portfolio_value_history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'value': entry['value']
                    } for entry in self.portfolio_value_history
                ],
                'emergency_shutdown': self.emergency_shutdown,
                'shutdown_reason': self.shutdown_reason,
                'risk_config': self.risk_config
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Risk Manager State gespeichert: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Risk Manager State: {str(e)}")
    
    def load_risk_state(self, filepath: str):
        """Lädt Risk Manager State"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Portfolio Value History laden
            self.portfolio_value_history = []
            for entry in state.get('portfolio_value_history', []):
                self.portfolio_value_history.append({
                    'timestamp': datetime.fromisoformat(entry['timestamp']),
                    'value': entry['value']
                })
            
            # Emergency Shutdown State
            self.emergency_shutdown = state.get('emergency_shutdown', False)
            self.shutdown_reason = state.get('shutdown_reason', '')
            
            self.logger.info(f"Risk Manager State geladen: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Risk Manager State: {str(e)}")

if __name__ == "__main__":
    # Test der Risk Manager Klasse
    print("Risk Manager Test")
    
    # Mock Config und Logger für Test
    class MockConfig:
        def get_config(self):
            return {
                'risk_management': {
                    'max_portfolio_risk': 0.02,
                    'max_position_risk': 0.01,
                    'max_daily_loss': 0.05,
                    'max_drawdown_limit': 0.15
                }
            }
    
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def critical(self, msg): print(f"CRITICAL: {msg}")
    
    # Test
    config = MockConfig()
    logger = MockLogger()
    
    risk_manager = RiskManager(config, logger)
    
    # Test Position Size Calculation
    position_size = risk_manager.calculate_position_size(
        symbol="BTCUSDT",
        entry_price=50000,
        stop_loss_price=49000,
        portfolio_value=10000,
        strategy_confidence=0.8
    )
    print(f"Berechnete Position Size: {position_size}")
    
    # Test Stop Loss Calculation
    stop_loss = risk_manager.calculate_stop_loss(
        symbol="BTCUSDT",
        entry_price=50000,
        position_type="long",
        volatility=0.03
    )
    print(f"Berechneter Stop Loss: {stop_loss}")
    
    print("Risk Manager Test abgeschlossen!")