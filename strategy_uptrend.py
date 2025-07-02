"""
Trading Bot Uptrend Strategy
Adaptive Uptrend-Strategie die darauf ausgelegt ist, Aufwärtstrends zu erkennen
und zu nutzen, während sie so lange wie möglich in profitablen Positionen bleibt.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import talib
from abc import ABC, abstractmethod
import math

@dataclass
class TrendSignal:
    """Struktur für Trend-Signale"""
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # 0.0 - 1.0
    confidence: float # 0.0 - 1.0
    reason: str
    indicators: Dict[str, float]
    timestamp: datetime
    
@dataclass
class PositionInfo:
    """Aktuelle Position-Informationen"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    max_profit: float
    max_drawdown: float

class TrendIndicator(ABC):
    """Basisklasse für Trend-Indikatoren"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_signal(self, data: pd.DataFrame) -> TrendSignal:
        pass

class MovingAverageTrend(TrendIndicator):
    """Moving Average Trend Indikator"""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50, ema_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ema_period = ema_period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Berechnet Moving Average Indikatoren"""
        close = data['close'].values
        
        # Simple Moving Averages
        sma_fast = talib.SMA(close, timeperiod=self.fast_period)[-1]
        sma_slow = talib.SMA(close, timeperiod=self.slow_period)[-1]
        
        # Exponential Moving Average
        ema = talib.EMA(close, timeperiod=self.ema_period)[-1]
        
        # MA Slope (Trend-Stärke)
        if len(close) >= self.fast_period + 5:
            ma_slope = (sma_fast - talib.SMA(close, timeperiod=self.fast_period)[-6]) / 5
        else:
            ma_slope = 0
        
        # Preisposition relativ zu MAs
        current_price = close[-1]
        price_above_fast = (current_price - sma_fast) / sma_fast
        price_above_slow = (current_price - sma_slow) / sma_slow
        
        return {
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'ema': ema,
            'ma_slope': ma_slope,
            'price_above_fast_pct': price_above_fast * 100,
            'price_above_slow_pct': price_above_slow * 100,
            'ma_distance': (sma_fast - sma_slow) / sma_slow * 100
        }
    
    def get_signal(self, data: pd.DataFrame) -> TrendSignal:
        """Generiert Trading-Signal basierend auf MA"""
        indicators = self.calculate(data)
        current_price = data['close'].iloc[-1]
        
        # Signal-Logik
        signal_type = 'HOLD'
        strength = 0.5
        confidence = 0.5
        reason = "Neutral MA signal"
        
        # Aufwärtstrend-Bedingungen
        if (indicators['sma_fast'] > indicators['sma_slow'] and
            current_price > indicators['sma_fast'] and
            indicators['ma_slope'] > 0):
            
            signal_type = 'BUY'
            strength = min(0.8, abs(indicators['ma_distance']) / 2 + 0.3)
            confidence = min(0.9, abs(indicators['ma_slope']) * 1000 + 0.5)
            reason = f"Strong uptrend: Price above MA, positive slope"
        
        # Schwache Verkaufssignale (nur bei klarem Trendbruch)
        elif (indicators['sma_fast'] < indicators['sma_slow'] and
              current_price < indicators['sma_fast'] and
              indicators['ma_slope'] < -0.001):
            
            signal_type = 'SELL'
            strength = min(0.6, abs(indicators['ma_distance']) / 3 + 0.2)
            confidence = min(0.7, abs(indicators['ma_slope']) * 1000 + 0.4)
            reason = f"Trend reversal: Price below MA, negative slope"
        
        return TrendSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            reason=reason,
            indicators=indicators,
            timestamp=datetime.now()
        )

class RSITrend(TrendIndicator):
    """RSI-basierter Trend-Indikator"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Berechnet RSI-Indikatoren"""
        close = data['close'].values
        
        rsi = talib.RSI(close, timeperiod=self.period)[-1]
        
        # RSI Trend (Slope)
        if len(close) >= self.period + 5:
            rsi_series = talib.RSI(close, timeperiod=self.period)
            rsi_slope = (rsi_series[-1] - rsi_series[-6]) / 5
        else:
            rsi_slope = 0
        
        # RSI Divergenz (vereinfacht)
        price_change = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0
        rsi_change = (rsi - talib.RSI(close, timeperiod=self.period)[-10]) if len(close) >= self.period + 10 else 0
        
        return {
            'rsi': rsi,
            'rsi_slope': rsi_slope,
            'rsi_position': (rsi - 50) / 50,  # -1 bis 1
            'price_rsi_divergence': price_change - rsi_change / 100
        }
    
    def get_signal(self, data: pd.DataFrame) -> TrendSignal:
        """Generiert RSI-basiertes Signal"""
        indicators = self.calculate(data)
        
        signal_type = 'HOLD'
        strength = 0.5
        confidence = 0.5
        reason = "Neutral RSI"
        
        rsi = indicators['rsi']
        
        # Kaufsignal bei überverkauft aber aufwärts RSI
        if rsi < self.oversold and indicators['rsi_slope'] > 0:
            signal_type = 'BUY'
            strength = min(0.8, (self.oversold - rsi) / self.oversold + 0.3)
            confidence = min(0.8, indicators['rsi_slope'] * 10 + 0.5)
            reason = f"RSI oversold but rising: {rsi:.1f}"
        
        # Moderate Kaufsignale bei mittlerem RSI mit positivem Trend
        elif 40 < rsi < 60 and indicators['rsi_slope'] > 1:
            signal_type = 'BUY'
            strength = min(0.6, indicators['rsi_slope'] / 5 + 0.2)
            confidence = 0.6
            reason = f"RSI in bullish territory with positive momentum"
        
        # Verkauf nur bei sehr überkauft UND negativem Trend
        elif rsi > self.overbought and indicators['rsi_slope'] < -1:
            signal_type = 'SELL'
            strength = min(0.6, (rsi - self.overbought) / (100 - self.overbought) + 0.2)
            confidence = min(0.7, abs(indicators['rsi_slope']) / 5 + 0.4)
            reason = f"RSI overbought and falling: {rsi:.1f}"
        
        return TrendSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            reason=reason,
            indicators=indicators,
            timestamp=datetime.now()
        )

class VolumeConfirmation(TrendIndicator):
    """Volumen-basierte Trend-Bestätigung"""
    
    def __init__(self, volume_ma_period: int = 20):
        self.volume_ma_period = volume_ma_period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Berechnet Volumen-Indikatoren"""
        if 'volume' not in data.columns:
            return {'volume_confirmed': 0, 'volume_ratio': 1, 'obv_trend': 0}
        
        volume = data['volume'].values
        close = data['close'].values
        
        # Volume Moving Average
        volume_ma = talib.SMA(volume, timeperiod=self.volume_ma_period)[-1]
        current_volume = volume[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
        
        # On-Balance Volume
        obv = talib.OBV(close, volume)
        obv_ma = talib.SMA(obv, timeperiod=10)
        obv_trend = (obv[-1] - obv_ma[-1]) / obv_ma[-1] if obv_ma[-1] != 0 else 0
        
        # Volume-Price Trend
        if len(close) >= 2:
            price_change = (close[-1] - close[-2]) / close[-2]
            volume_confirmed = 1 if (price_change > 0 and volume_ratio > 1.2) else 0
        else:
            volume_confirmed = 0
        
        return {
            'volume_ratio': volume_ratio,
            'volume_confirmed': volume_confirmed,
            'obv_trend': obv_trend,
            'avg_volume': volume_ma
        }
    
    def get_signal(self, data: pd.DataFrame) -> TrendSignal:
        """Volumen-Bestätigung für Trends"""
        indicators = self.calculate(data)
        
        # Volumen ist eher bestätigend als Signal-gebend
        strength = 0.3
        confidence = 0.6
        
        if indicators['volume_confirmed'] and indicators['obv_trend'] > 0:
            signal_type = 'BUY'
            strength = min(0.7, indicators['volume_ratio'] / 3 + 0.2)
            confidence = min(0.8, abs(indicators['obv_trend']) * 10 + 0.5)
            reason = f"Volume confirms uptrend"
        else:
            signal_type = 'HOLD'
            reason = "Volume neutral or weak"
        
        return TrendSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            reason=reason,
            indicators=indicators,
            timestamp=datetime.now()
        )

class UptrendStrategy:
    """Hauptklasse für die Uptrend-Strategie"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.name = "uptrend_strategy"
        
        # Strategy-spezifische Parameter
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% des Portfolios
        self.min_confidence = config.get('min_confidence', 0.6)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 5.0)  # 5% Trailing Stop
        self.take_profit_pct = config.get('take_profit_pct', 15.0)    # 15% Take Profit
        self.max_drawdown_pct = config.get('max_drawdown_pct', 8.0)   # 8% Max Drawdown
        
        # Position-Tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.trailing_stops: Dict[str, float] = {}
        
        # Indikatoren initialisieren
        self.ma_indicator = MovingAverageTrend(
            fast_period=config.get('ma_fast_period', 20),
            slow_period=config.get('ma_slow_period', 50),
            ema_period=config.get('ema_period', 9)
        )
        
        self.rsi_indicator = RSITrend(
            period=config.get('rsi_period', 14),
            overbought=config.get('rsi_overbought', 75),  # Höher für Uptrend-Bias
            oversold=config.get('rsi_oversold', 25)       # Niedriger für Uptrend-Bias
        )
        
        self.volume_indicator = VolumeConfirmation(
            volume_ma_period=config.get('volume_ma_period', 20)
        )
        
        # Performance-Tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_time': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        self.trade_history: List[Dict[str, Any]] = []
    
    def analyze_market(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert die Marktsituation für ein Symbol"""
        
        if len(data) < 50:  # Mindestens 50 Datenpunkte
            return {
                'market_condition': 'insufficient_data',
                'overall_signal': 'HOLD',
                'confidence': 0.0,
                'strength': 0.0,
                'reason': 'Not enough data for analysis'
            }
        
        # Alle Indikatoren berechnen
        ma_signal = self.ma_indicator.get_signal(data)
        rsi_signal = self.rsi_indicator.get_signal(data)
        volume_signal = self.volume_indicator.get_signal(data)
        
        # Zusätzliche Marktanalyse
        market_analysis = self._analyze_market_structure(data)
        
        # Signal-Aggregation mit Gewichtung
        signals = [
            (ma_signal, 0.4),      # MA hat höchste Gewichtung für Trend
            (rsi_signal, 0.3),     # RSI für Timing
            (volume_signal, 0.3)   # Volume für Bestätigung
        ]
        
        overall_signal, confidence, strength = self._aggregate_signals(signals)
        
        # Marktbedingung bestimmen
        market_condition = self._determine_market_condition(data, ma_signal, rsi_signal)
        
        analysis_result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'market_condition': market_condition,
            'overall_signal': overall_signal,
            'confidence': confidence,
            'strength': strength,
            'reason': self._generate_reason(ma_signal, rsi_signal, volume_signal),
            'signals': {
                'ma': ma_signal,
                'rsi': rsi_signal,
                'volume': volume_signal
            },
            'market_structure': market_analysis,
            'current_price': data['close'].iloc[-1]
        }
        
        if self.logger:
            self.logger.log_market_analysis(
                symbol=symbol,
                timeframe=self.config.get('timeframe', '1h'),
                market_condition=market_condition,
                confidence=confidence,
                indicators={
                    'ma_signal': ma_signal.signal_type,
                    'rsi': rsi_signal.indicators.get('rsi', 0),
                    'volume_ratio': volume_signal.indicators.get('volume_ratio', 1),
                    'overall_strength': strength
                }
            )
        
        return analysis_result
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert die Marktstruktur"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Volatilität (ATR)
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        volatility_pct = (atr / close[-1]) * 100
        
        # Trend-Stärke (ADX)
        adx = talib.ADX(high, low, close, timeperiod=14)[-1]
        
        # Support/Resistance Levels (vereinfacht)
        recent_highs = np.max(high[-20:]) if len(high) >= 20 else high[-1]
        recent_lows = np.min(low[-20:]) if len(low) >= 20 else low[-1]
        
        current_price = close[-1]
        distance_to_high = (recent_highs - current_price) / current_price * 100
        distance_to_low = (current_price - recent_lows) / current_price * 100
        
        return {
            'volatility_pct': volatility_pct,
            'trend_strength': adx,
            'distance_to_high_pct': distance_to_high,
            'distance_to_low_pct': distance_to_low,
            'recent_high': recent_highs,
            'recent_low': recent_lows
        }
    
    def _aggregate_signals(self, signals: List[Tuple[TrendSignal, float]]) -> Tuple[str, float, float]:
        """Aggregiert mehrere Signale zu einem Gesamtsignal"""
        
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        strength_sum = 0.0
        
        for signal, weight in signals:
            total_weight += weight
            confidence_sum += signal.confidence * weight
            strength_sum += signal.strength * weight
            
            if signal.signal_type == 'BUY':
                buy_score += signal.strength * signal.confidence * weight
            elif signal.signal_type == 'SELL':
                sell_score += signal.strength * signal.confidence * weight
        
        avg_confidence = confidence_sum / total_weight if total_weight > 0 else 0
        avg_strength = strength_sum / total_weight if total_weight > 0 else 0
        
        # Entscheidung mit Uptrend-Bias
        if buy_score > sell_score * 1.5:  # Bias towards buying
            return 'BUY', avg_confidence, avg_strength
        elif sell_score > buy_score and sell_score > 0.3:  # Nur bei starken Verkaufssignalen
            return 'SELL', avg_confidence, avg_strength
        else:
            return 'HOLD', avg_confidence, avg_strength
    
    def _determine_market_condition(self, data: pd.DataFrame, ma_signal: TrendSignal, 
                                  rsi_signal: TrendSignal) -> str:
        """Bestimmt die aktuelle Marktbedingung"""
        
        close = data['close'].values
        
        # Trend-Richtung basierend auf MA
        if ma_signal.indicators['ma_distance'] > 2:
            trend = 'strong_uptrend'
        elif ma_signal.indicators['ma_distance'] > 0.5:
            trend = 'uptrend'
        elif ma_signal.indicators['ma_distance'] > -0.5:
            trend = 'sideways'
        elif ma_signal.indicators['ma_distance'] > -2:
            trend = 'downtrend'
        else:
            trend = 'strong_downtrend'
        
        # RSI-Bedingung
        rsi = rsi_signal.indicators['rsi']
        if rsi > 70:
            rsi_condition = 'overbought'
        elif rsi > 50:
            rsi_condition = 'bullish'
        elif rsi > 30:
            rsi_condition = 'bearish'
        else:
            rsi_condition = 'oversold'
        
        # Kombinierte Bedingung
        return f"{trend}_{rsi_condition}"
    
    def _generate_reason(self, ma_signal: TrendSignal, rsi_signal: TrendSignal, 
                        volume_signal: TrendSignal) -> str:
        """Generiert eine Begründung für das Signal"""
        
        reasons = []
        
        if ma_signal.strength > 0.6:
            reasons.append(ma_signal.reason)
        
        if rsi_signal.strength > 0.6:
            reasons.append(rsi_signal.reason)
        
        if volume_signal.strength > 0.5:
            reasons.append(volume_signal.reason)
        
        if not reasons:
            reasons.append("Neutral market conditions")
        
        return " | ".join(reasons)
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Generiert Trading-Signale für ein Symbol"""
        
        analysis = self.analyze_market(symbol, data)
        current_price = data['close'].iloc[-1]
        
        # Position-Check
        has_position = symbol in self.positions
        position_info = self.positions.get(symbol) if has_position else None
        
        # Signal-Generierung
        signal_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'signal': analysis['overall_signal'],
            'confidence': analysis['confidence'],
            'strength': analysis['strength'],
            'market_condition': analysis['market_condition'],
            'reason': analysis['reason'],
            'has_position': has_position,
            'position_info': position_info.__dict__ if position_info else None,
            'action': None,
            'quantity': 0.0,
            'target_price': None,
            'stop_loss': None
        }
        
        # Trading-Logik
        if not has_position:
            # Neue Position eingehen
            if (analysis['overall_signal'] == 'BUY' and 
                analysis['confidence'] >= self.min_confidence and
                analysis['strength'] >= 0.4):
                
                signal_data['action'] = 'BUY'
                signal_data['quantity'] = self._calculate_position_size(
                    symbol, current_price, analysis['strength']
                )
                signal_data['stop_loss'] = self._calculate_stop_loss(current_price, 'BUY')
                signal_data['target_price'] = self._calculate_take_profit(current_price, 'BUY')
        
        else:
            # Position-Management
            position_action = self._manage_position(symbol, data, analysis)
            if position_action:
                signal_data.update(position_action)
        
        if self.logger and signal_data['action']:
            self.logger.log_strategy_decision(
                strategy=self.name,
                symbol=symbol,
                decision=signal_data['action'],
                confidence=analysis['confidence'],
                market_condition=analysis['market_condition'],
                current_price=current_price,
                position_size=signal_data['quantity']
            )
        
        return signal_data
    
    def _calculate_position_size(self, symbol: str, price: float, strength: float) -> float:
        """Berechnet die Positionsgröße basierend auf Risiko und Signal-Stärke"""
        
        # Basis-Positionsgröße (% des Portfolios)
        base_size = self.max_position_size * strength
        
        # Volatilitäts-Anpassung würde hier implementiert werden
        # Für jetzt einfache Berechnung
        
        return base_size
    
    def _calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Berechnet Stop-Loss Level"""
        
        if direction == 'BUY':
            return entry_price * (1 - self.max_drawdown_pct / 100)
        else:
            return entry_price * (1 + self.max_drawdown_pct / 100)
    
    def _calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """Berechnet Take-Profit Level"""
        
        if direction == 'BUY':
            return entry_price * (1 + self.take_profit_pct / 100)
        else:
            return entry_price * (1 - self.take_profit_pct / 100)
    
    def _manage_position(self, symbol: str, data: pd.DataFrame, 
                        analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Management bestehender Positionen"""
        
        position = self.positions[symbol]
        current_price = data['close'].iloc[-1]
        
        # Update Position Info
        position.current_price = current_price
        position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price * 100
        position.max_profit = max(position.max_profit, position.unrealized_pnl)
        position.max_drawdown = min(position.max_drawdown, position.unrealized_pnl)
        
        # Trailing Stop Update
        if symbol not in self.trailing_stops:
            self.trailing_stops[symbol] = position.entry_price * (1 - self.trailing_stop_pct / 100)
        
        # Trailing Stop Logic
        if position.unrealized_pnl > 0:  # Nur bei Gewinn
            new_trailing_stop = current_price * (1 - self.trailing_stop_pct / 100)
            self.trailing_stops[symbol] = max(self.trailing_stops[symbol], new_trailing_stop)
        
        # Exit-Bedingungen prüfen
        
        # 1. Trailing Stop getriggert
        if current_price <= self.trailing_stops[symbol]:
            return {
                'action': 'SELL',
                'quantity': position.size,
                'reason': f'Trailing stop triggered at {self.trailing_stops[symbol]:.4f}',
                'exit_type': 'trailing_stop'
            }
        
        # 2. Take Profit erreicht
        take_profit_level = position.entry_price * (1 + self.take_profit_pct / 100)
        if current_price >= take_profit_level:
            return {
                'action': 'SELL',
                'quantity': position.size,
                'reason': f'Take profit reached at {take_profit_level:.4f}',
                'exit_type': 'take_profit'
            }
        
        # 3. Max Drawdown erreicht
        if position.unrealized_pnl <= -self.max_drawdown_pct:
            return {
                'action': 'SELL',
                'quantity': position.size,
                'reason': f'Max drawdown reached: {position.unrealized_pnl:.2f}%',
                'exit_type': 'stop_loss'
            }
        
        # 4. Starkes Verkaufssignal
        if (analysis['overall_signal'] == 'SELL' and 
            analysis['confidence'] > 0.8 and 
            analysis['strength'] > 0.6):
            
            return {
                'action': 'SELL',
                'quantity': position.size,
                'reason': f'Strong sell signal: {analysis["reason"]}',
                'exit_type': 'signal_exit'
            }
        
        # 5. Zeitbasierter Exit (nach langer Haltedauer ohne Gewinn)
        holding_time = datetime.now() - position.entry_time
        if (holding_time.days > 30 and position.unrealized_pnl < 2):
            return {
                'action': 'SELL',
                'quantity': position.size * 0.5,  # Nur 50% verkaufen
                'reason': f'Long holding time with minimal profit',
                'exit_type': 'time_based'
            }
        
        return None
    
    def execute_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Führt ein Trading-Signal aus (Simulation)"""
        
        symbol = signal_data['symbol']
        action = signal_data['action']
        
        if not action:
            return False
        
        try:
            if action == 'BUY':
                # Neue Position eröffnen
                position = PositionInfo(
                    symbol=symbol,
                    size=signal_data['quantity'],
                    entry_price=signal_data['current_price'],
                    current_price=signal_data['current_price'],
                    unrealized_pnl=0.0,
                    entry_time=datetime.now(),
                    max_profit=0.0,
                    max_drawdown=0.0
                )
                
                self.positions[symbol] = position
                self.trailing_stops[symbol] = signal_data['stop_loss']
                
                if self.logger:
                    self.logger.log_trade(
                        action='BUY',
                        symbol=symbol,
                        amount=signal_data['quantity'],
                        price=signal_data['current_price'],
                        strategy=self.name,
                        exchange='simulation'
                    )
            
            elif action == 'SELL':
                # Position schließen oder reduzieren
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # Trade für Performance-Tracking speichern
                    trade_record = {
                        'symbol': symbol,
                        'entry_time': position.entry_time,
                        'exit_time': datetime.now(),
                        'entry_price': position.entry_price,
                        'exit_price': signal_data['current_price'],
                        'quantity': signal_data['quantity'],
                        'pnl': position.unrealized_pnl,
                        'pnl_abs': (signal_data['current_price'] - position.entry_price) * signal_data['quantity'],
                        'exit_reason': signal_data.get('reason', 'Manual exit'),
                        'exit_type': signal_data.get('exit_type', 'manual'),
                        'holding_time': (datetime.now() - position.entry_time).total_seconds() / 3600  # Stunden
                    }
                    
                    self.trade_history.append(trade_record)
                    
                    # Performance-Metriken aktualisieren
                    self._update_performance_metrics(trade_record)
                    
                    # Position entfernen oder reduzieren
                    if signal_data['quantity'] >= position.size:
                        # Komplette Position schließen
                        del self.positions[symbol]
                        if symbol in self.trailing_stops:
                            del self.trailing_stops[symbol]
                    else:
                        # Position reduzieren
                        position.size -= signal_data['quantity']
                    
                    if self.logger:
                        self.logger.log_trade(
                            action='SELL',
                            symbol=symbol,
                            amount=signal_data['quantity'],
                            price=signal_data['current_price'],
                            strategy=self.name,
                            exchange='simulation',
                            pnl=trade_record['pnl'],
                            reason=signal_data.get('reason', '')
                        )
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing signal for {symbol}: {str(e)}")
            return False
    
    def _update_performance_metrics(self, trade_record: Dict[str, Any]):
        """Aktualisiert Performance-Metriken basierend auf abgeschlossenem Trade"""
        
        self.performance_metrics['total_trades'] += 1
        
        if trade_record['pnl'] > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        self.performance_metrics['total_return'] += trade_record['pnl']
        self.performance_metrics['best_trade'] = max(
            self.performance_metrics['best_trade'], 
            trade_record['pnl']
        )
        self.performance_metrics['worst_trade'] = min(
            self.performance_metrics['worst_trade'], 
            trade_record['pnl']
        )
        
        # Durchschnittliche Haltedauer
        total_holding_time = sum(t['holding_time'] for t in self.trade_history)
        self.performance_metrics['avg_holding_time'] = total_holding_time / len(self.trade_history)
        
        # Max Drawdown aktualisieren
        if trade_record['pnl'] < 0:
            current_drawdown = abs(trade_record['pnl'])
            self.performance_metrics['max_drawdown'] = max(
                self.performance_metrics['max_drawdown'],
                current_drawdown
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung der Strategy-Performance zurück"""
        
        if self.performance_metrics['total_trades'] == 0:
            return {
                'strategy': self.name,
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'avg_return_per_trade': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'avg_holding_time_hours': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'active_positions': len(self.positions),
                'sharpe_ratio': 0.0
            }
        
        winning_trades = self.performance_metrics['winning_trades']
        losing_trades = self.performance_metrics['losing_trades']
        total_trades = self.performance_metrics['total_trades']
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_return = self.performance_metrics['total_return'] / total_trades if total_trades > 0 else 0
        
        # Profit Factor (Verhältnis Gewinne zu Verlusten)
        total_wins = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Vereinfachte Sharpe Ratio
        if len(self.trade_history) > 1:
            returns = [t['pnl'] for t in self.trade_history]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'strategy': self.name,
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'total_return': self.performance_metrics['total_return'],
            'avg_return_per_trade': avg_return,
            'profit_factor': profit_factor,
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'avg_holding_time_hours': self.performance_metrics['avg_holding_time'],
            'best_trade': self.performance_metrics['best_trade'],
            'worst_trade': self.performance_metrics['worst_trade'],
            'active_positions': len(self.positions),
            'sharpe_ratio': sharpe_ratio,
            'current_positions': {k: v.__dict__ for k, v in self.positions.items()}
        }
    
    def update_position_info(self, symbol: str, current_price: float):
        """Aktualisiert Position-Informationen mit aktuellem Preis"""
        
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price * 100
            position.max_profit = max(position.max_profit, position.unrealized_pnl)
            position.max_drawdown = min(position.max_drawdown, position.unrealized_pnl)
    
    def reset_strategy(self):
        """Setzt die Strategie zurück (für Backtesting oder Neustart)"""
        
        self.positions.clear()
        self.trailing_stops.clear()
        self.trade_history.clear()
        
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_time': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
    
    def get_current_exposure(self) -> Dict[str, float]:
        """Gibt die aktuelle Markt-Exposition zurück"""
        
        exposure = {}
        for symbol, position in self.positions.items():
            exposure[symbol] = position.size * position.current_price
        
        return exposure
    
    def should_rebalance(self) -> bool:
        """Prüft, ob ein Rebalancing erforderlich ist"""
        
        total_exposure = sum(self.get_current_exposure().values())
        
        # Wenn Gesamtexposition zu groß wird
        if total_exposure > self.max_position_size * 3:  # 3x der Basis-Positionsgröße
            return True
        
        # Wenn einzelne Positionen zu groß werden
        for symbol, position in self.positions.items():
            position_value = position.size * position.current_price
            if position_value > self.max_position_size * 1.5:
                return True
        
        return False
    
    def get_rebalance_instructions(self) -> List[Dict[str, Any]]:
        """Gibt Anweisungen für Rebalancing zurück"""
        
        instructions = []
        
        for symbol, position in self.positions.items():
            position_value = position.size * position.current_price
            target_value = self.max_position_size
            
            if position_value > target_value * 1.2:  # 20% über Zielwert
                reduce_amount = (position_value - target_value) / position.current_price
                
                instructions.append({
                    'action': 'REDUCE',
                    'symbol': symbol,
                    'amount': reduce_amount,
                    'reason': f'Position too large: {position_value:.4f} vs target {target_value:.4f}'
                })
        
        return instructions
    
    def emergency_exit_all(self, reason: str = "Emergency exit") -> List[Dict[str, Any]]:
        """Notfall-Ausstieg aus allen Positionen"""
        
        exit_signals = []
        
        for symbol, position in self.positions.items():
            exit_signals.append({
                'symbol': symbol,
                'action': 'SELL',
                'quantity': position.size,
                'current_price': position.current_price,
                'reason': reason,
                'exit_type': 'emergency'
            })
        
        if self.logger:
            self.logger.warning(f"Emergency exit triggered for all positions: {reason}")
        
        return exit_signals
    
    def validate_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Validiert ein Trading-Signal vor der Ausführung"""
        
        symbol = signal_data['symbol']
        action = signal_data.get('action')
        quantity = signal_data.get('quantity', 0)
        
        if not action:
            return False
        
        # Basis-Validierungen
        if quantity <= 0:
            if self.logger:
                self.logger.warning(f"Invalid quantity for {symbol}: {quantity}")
            return False
        
        if action == 'BUY':
            # Prüfe, ob bereits Position vorhanden
            if symbol in self.positions:
                if self.logger:
                    self.logger.warning(f"Already have position in {symbol}")
                return False
            
            # Prüfe Positionsgröße
            if quantity > self.max_position_size:
                if self.logger:
                    self.logger.warning(f"Position size too large for {symbol}: {quantity}")
                return False
        
        elif action == 'SELL':
            # Prüfe, ob Position vorhanden
            if symbol not in self.positions:
                if self.logger:
                    self.logger.warning(f"No position to sell in {symbol}")
                return False
            
            # Prüfe, ob genug zu verkaufen vorhanden
            current_position = self.positions[symbol].size
            if quantity > current_position:
                if self.logger:
                    self.logger.warning(f"Not enough position to sell in {symbol}: {quantity} > {current_position}")
                return False
        
        return True
    
    def get_status_report(self) -> Dict[str, Any]:
        """Gibt einen detaillierten Status-Report zurück"""
        
        return {
            'strategy_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'active_positions': len(self.positions),
            'positions_detail': {k: v.__dict__ for k, v in self.positions.items()},
            'trailing_stops': self.trailing_stops.copy(),
            'performance': self.get_performance_summary(),
            'config': {
                'max_position_size': self.max_position_size,
                'min_confidence': self.min_confidence,
                'trailing_stop_pct': self.trailing_stop_pct,
                'take_profit_pct': self.take_profit_pct,
                'max_drawdown_pct': self.max_drawdown_pct
            },
            'recent_trades': self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
        }

# Hilfsfunktionen für externe Nutzung

def create_uptrend_strategy(config: Dict[str, Any], logger=None) -> UptrendStrategy:
    """Factory-Funktion zum Erstellen einer Uptrend-Strategie"""
    return UptrendStrategy(config, logger)

def validate_uptrend_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validiert die Konfiguration für die Uptrend-Strategie"""
    
    errors = []
    required_keys = ['max_position_size', 'min_confidence', 'trailing_stop_pct']
    
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: {key}")
    
    # Werte-Validierung
    if 'max_position_size' in config:
        if not 0 < config['max_position_size'] <= 1:
            errors.append("max_position_size must be between 0 and 1")
    
    if 'min_confidence' in config:
        if not 0 < config['min_confidence'] <= 1:
            errors.append("min_confidence must be between 0 and 1")
    
    if 'trailing_stop_pct' in config:
        if not 0 < config['trailing_stop_pct'] <= 50:
            errors.append("trailing_stop_pct must be between 0 and 50")
    
    return len(errors) == 0, errors

def get_default_uptrend_config() -> Dict[str, Any]:
    """Gibt die Standard-Konfiguration für die Uptrend-Strategie zurück"""
    
    return {
        'max_position_size': 0.1,        # 10% des Portfolios
        'min_confidence': 0.6,           # Mindest-Konfidenz für Signale
        'trailing_stop_pct': 5.0,        # 5% Trailing Stop
        'take_profit_pct': 15.0,         # 15% Take Profit
        'max_drawdown_pct': 8.0,         # 8% Max Drawdown
        'ma_fast_period': 20,            # Schneller MA-Zeitraum
        'ma_slow_period': 50,            # Langsamer MA-Zeitraum
        'ema_period': 9,                 # EMA-Zeitraum
        'rsi_period': 14,                # RSI-Zeitraum
        'rsi_overbought': 75,            # RSI Overbought Level
        'rsi_oversold': 25,              # RSI Oversold Level
        'volume_ma_period': 20,          # Volume MA-Zeitraum
        'timeframe': '1h'                # Standard-Zeitrahmen
    }

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Test-Konfiguration
    config = get_default_uptrend_config()
    
    # Strategie erstellen
    strategy = create_uptrend_strategy(config)
    
    # Beispiel-Daten (würden normalerweise vom data_manager kommen)
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'open': np.random.uniform(50000, 52000, 100),
        'high': np.random.uniform(50500, 52500, 100),
        'low': np.random.uniform(49500, 51500, 100),
        'close': np.random.uniform(50000, 52000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    # Market-Analyse durchführen
    analysis = strategy.analyze_market('BTCUSDT', sample_data)
    print("Market Analysis:")
    print(f"Signal: {analysis['overall_signal']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Market Condition: {analysis['market_condition']}")
    
    # Signal generieren
    signal = strategy.generate_signals('BTCUSDT', sample_data)
    print(f"\nTrading Signal:")
    print(f"Action: {signal['action']}")
    print(f"Reason: {signal['reason']}")
    
    # Performance-Summary
    performance = strategy.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Active Positions: {performance['active_positions']}")