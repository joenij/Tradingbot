"""
Downtrend Trading Strategy
Verkauft vor größeren Abwärtsbewegungen und kauft günstig wieder ein
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import talib
import warnings
warnings.filterwarnings('ignore')

from logger import get_logger

class DowntrendPhase(Enum):
    EARLY_DECLINE = "early_decline"      # Früher Rückgang erkannt
    ACCELERATING = "accelerating"        # Beschleunigender Rückgang
    OVERSOLD = "oversold"               # Überverkauft
    RECOVERY_SIGNAL = "recovery_signal"  # Erste Erholungszeichen
    RECOVERY = "recovery"               # Aktive Erholung

@dataclass
class DowntrendSignal:
    signal_type: str
    strength: float
    confidence: float
    price_target: float
    stop_loss: float
    timeframe: str
    indicators: Dict[str, float]
    risk_level: str
    expected_duration: int  # In Stunden

class DowntrendStrategy:
    def __init__(self, config_manager, logger=None):
        self.config = config_manager
        self.logger = logger or get_logger()
        
        # Strategie-Parameter aus Config
        self.params = self.config.get('strategies', {}).get('downtrend', {})
        
        # Standard-Parameter falls Config fehlt
        self.default_params = {
            'rsi_threshold': 30,           # RSI Überverkauft-Schwelle
            'rsi_oversold': 20,           # Extreme Überverkauft-Schwelle
            'macd_threshold': -0.001,      # MACD Schwelle für Downtrend
            'bb_threshold': 0.02,          # Bollinger Band Schwelle
            'volume_spike_threshold': 2.0,  # Volumen-Spike Faktor
            'decline_threshold': 0.03,     # Minimaler Rückgang für Signal (3%)
            'max_position_size': 0.3,      # Maximale Positionsgröße
            'stop_loss_pct': 0.05,         # Stop-Loss Prozent
            'take_profit_pct': 0.08,       # Take-Profit Prozent
            'fear_greed_threshold': 25,    # Fear & Greed Index Schwelle
            'trend_lookback': 20,          # Tage für Trendanalyse
            'volatility_threshold': 0.03,  # Volatilitäts-Schwelle
            'reentry_threshold': 0.15,     # Wiedereinstiegs-Schwelle
            'min_hold_time': 4,            # Minimale Haltezeit in Stunden
            'max_hold_time': 168,          # Maximale Haltezeit in Stunden (7 Tage)
            'risk_per_trade': 0.02,        # Risiko pro Trade (2%)
            'recovery_confirmation': 3,     # Bestätigungen für Erholung
        }
        
        # Parameter zusammenführen
        for key, value in self.default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        # Interne Zustandsvariablen
        self.current_phase = DowntrendPhase.EARLY_DECLINE
        self.positions = {}
        self.trade_history = []
        self.market_memory = []
        self.fear_greed_history = []
        self.volatility_history = []
        
        # Performance-Tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'avg_trade_duration': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'recovery_success_rate': 0.0
        }
        
        self.logger.info("Downtrend Strategy initialisiert", 'strategy', 
                        strategy_params=self.params)
    
    def analyze_market_condition(self, df: pd.DataFrame, symbol: str, 
                               timeframe: str = '1h') -> Dict[str, Any]:
        """Analysiert die aktuelle Marktsituation für Downtrend-Erkennung"""
        
        try:
            if len(df) < 50:
                return {'condition': 'insufficient_data', 'confidence': 0.0}
            
            # Technische Indikatoren berechnen
            indicators = self._calculate_indicators(df)
            
            # Marktphasen-Analyse
            market_phase = self._identify_market_phase(df, indicators)
            
            # Volatilitäts-Analyse
            volatility_analysis = self._analyze_volatility(df)
            
            # Sentiment-Analyse (vereinfacht)
            sentiment = self._analyze_sentiment(df, indicators)
            
            # Volumen-Analyse
            volume_analysis = self._analyze_volume(df)
            
            # Trend-Stärke bewerten
            trend_strength = self._calculate_trend_strength(df, indicators)
            
            # Downtrend-Wahrscheinlichkeit berechnen
            downtrend_probability = self._calculate_downtrend_probability(
                indicators, volatility_analysis, sentiment, volume_analysis, trend_strength
            )
            
            # Markt-Condition bestimmen
            condition = self._determine_market_condition(
                downtrend_probability, indicators, market_phase
            )
            
            analysis = {
                'condition': condition,
                'confidence': downtrend_probability,
                'phase': market_phase.value,
                'indicators': indicators,
                'volatility': volatility_analysis,
                'sentiment': sentiment,
                'volume': volume_analysis,
                'trend_strength': trend_strength,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            # Speichere Analyse für Lernzwecke
            self.market_memory.append(analysis)
            if len(self.market_memory) > 1000:
                self.market_memory = self.market_memory[-1000:]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fehler in Marktanalyse: {e}", 'strategy')
            return {'condition': 'error', 'confidence': 0.0}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Berechnet technische Indikatoren für Downtrend-Analyse"""
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1]
            indicators['rsi_fast'] = talib.RSI(close, timeperiod=7)[-1]
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macdsignal[-1]
            indicators['macd_histogram'] = macdhist[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            indicators['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # Moving Averages
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1]
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1]
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)[-1]
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
            indicators['stoch_k'] = slowk[-1]
            indicators['stoch_d'] = slowd[-1]
            
            # ATR (Average True Range)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # CCI (Commodity Channel Index)
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)[-1]
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1]
            
            # ADX (Average Directional Index)
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)[-1]
            
            # Volume-basierte Indikatoren
            indicators['volume_sma'] = np.mean(volume[-20:])
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma']
            
            # Momentum
            indicators['momentum'] = talib.MOM(close, timeperiod=10)[-1]
            
            # Price Position
            indicators['price_position'] = (close[-1] - np.min(close[-20:])) / (np.max(close[-20:]) - np.min(close[-20:]))
            
            # Trend-Indikatoren
            indicators['price_vs_sma20'] = (close[-1] - indicators['sma_20']) / indicators['sma_20']
            indicators['price_vs_sma50'] = (close[-1] - indicators['sma_50']) / indicators['sma_50']
            indicators['sma20_vs_sma50'] = (indicators['sma_20'] - indicators['sma_50']) / indicators['sma_50']
            
        except Exception as e:
            self.logger.error(f"Fehler bei Indikator-Berechnung: {e}", 'strategy')
            # Fallback-Werte
            indicators = {key: 0.0 for key in ['rsi', 'macd', 'bb_position', 'stoch_k', 'adx']}
        
        return indicators
    
    def _identify_market_phase(self, df: pd.DataFrame, indicators: Dict[str, float]) -> DowntrendPhase:
        """Identifiziert die aktuelle Marktphase"""
        
        try:
            close = df['close'].values
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            bb_position = indicators.get('bb_position', 0.5)
            price_vs_sma20 = indicators.get('price_vs_sma20', 0)
            
            # Preisänderung der letzten Tage
            price_change_1d = (close[-1] - close[-24]) / close[-24] if len(close) >= 24 else 0
            price_change_3d = (close[-1] - close[-72]) / close[-72] if len(close) >= 72 else 0
            price_change_7d = (close[-1] - close[-168]) / close[-168] if len(close) >= 168 else 0
            
            # Volatilität
            volatility = np.std(close[-24:]) / np.mean(close[-24:]) if len(close) >= 24 else 0
            
            # Phasen-Bestimmung
            if rsi < 20 and bb_position < 0.1 and price_change_3d < -0.15:
                return DowntrendPhase.OVERSOLD
            elif rsi > 30 and price_change_1d > 0.02 and macd > indicators.get('macd_signal', 0):
                return DowntrendPhase.RECOVERY_SIGNAL
            elif rsi > 40 and price_change_3d > 0.05:
                return DowntrendPhase.RECOVERY
            elif price_change_1d < -0.02 and price_change_3d < -0.08 and volatility > 0.03:
                return DowntrendPhase.ACCELERATING
            else:
                return DowntrendPhase.EARLY_DECLINE
                
        except Exception as e:
            self.logger.error(f"Fehler bei Phasen-Identifikation: {e}", 'strategy')
            return DowntrendPhase.EARLY_DECLINE
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analysiert die Marktvolatilität"""
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Verschiedene Volatilitäts-Maße
            volatility_1h = np.std(close[-24:]) / np.mean(close[-24:]) if len(close) >= 24 else 0
            volatility_4h = np.std(close[-96:]) / np.mean(close[-96:]) if len(close) >= 96 else 0
            volatility_1d = np.std(close[-168:]) / np.mean(close[-168:]) if len(close) >= 168 else 0
            
            # True Range basierte Volatilität
            tr = np.maximum(high[1:] - low[1:], 
                           np.maximum(np.abs(high[1:] - close[:-1]), 
                                    np.abs(low[1:] - close[:-1])))
            atr_volatility = np.mean(tr[-14:]) / np.mean(close[-14:]) if len(tr) >= 14 else 0
            
            # Volatilitäts-Trends
            vol_trend = np.polyfit(range(len(close[-48:])), close[-48:], 1)[0] if len(close) >= 48 else 0
            
            return {
                'volatility_1h': volatility_1h,
                'volatility_4h': volatility_4h,
                'volatility_1d': volatility_1d,
                'atr_volatility': atr_volatility,
                'volatility_trend': vol_trend,
                'is_high_volatility': volatility_1h > self.params['volatility_threshold']
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Volatilitäts-Analyse: {e}", 'strategy')
            return {'volatility_1h': 0.0, 'is_high_volatility': False}
    
    def _analyze_sentiment(self, df: pd.DataFrame, indicators: Dict[str, float]) -> Dict[str, float]:
        """Analysiert das Marktsentiment (vereinfacht)"""
        
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            # Sentiment-Proxies
            rsi = indicators.get('rsi', 50)
            fear_greed_proxy = 100 - rsi  # Vereinfachte Fear & Greed Approximation
            
            # Volumen-Sentiment
            volume_trend = np.polyfit(range(len(volume[-24:])), volume[-24:], 1)[0] if len(volume) >= 24 else 0
            
            # Price-Action Sentiment
            green_candles = sum(1 for i in range(-10, 0) if close[i] > close[i-1])
            red_candles = 10 - green_candles
            
            # Panic-Indikator (starke Abwärtsbewegungen mit hohem Volumen)
            panic_score = 0
            for i in range(-5, 0):
                if len(close) > abs(i):
                    price_drop = (close[i-1] - close[i]) / close[i-1]
                    if price_drop > 0.03 and volume[i] > np.mean(volume[-20:]) * 1.5:
                        panic_score += 1
            
            return {
                'fear_greed_proxy': fear_greed_proxy,
                'volume_sentiment': volume_trend,
                'green_red_ratio': green_candles / 10,
                'panic_score': panic_score,
                'overall_sentiment': (fear_greed_proxy + (green_candles * 10)) / 2
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Sentiment-Analyse: {e}", 'strategy')
            return {'fear_greed_proxy': 50, 'overall_sentiment': 50}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analysiert Volumen-Muster"""
        
        try:
            volume = df['volume'].values
            close = df['close'].values
            
            # Volumen-Statistiken
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume
            
            # Volumen-Spikes in Abwärtsbewegungen
            volume_spikes_down = 0
            for i in range(-5, 0):
                if len(close) > abs(i) and len(volume) > abs(i):
                    price_change = (close[i] - close[i-1]) / close[i-1]
                    vol_ratio = volume[i] / avg_volume
                    if price_change < -0.02 and vol_ratio > 1.5:
                        volume_spikes_down += 1
            
            # On-Balance Volume Trend
            obv = np.cumsum(np.where(close[1:] > close[:-1], volume[1:], -volume[1:]))
            obv_trend = np.polyfit(range(len(obv[-20:])), obv[-20:], 1)[0] if len(obv) >= 20 else 0
            
            return {
                'volume_ratio': volume_ratio,
                'avg_volume': avg_volume,
                'volume_spikes_down': volume_spikes_down,
                'obv_trend': obv_trend,
                'is_volume_spike': volume_ratio > self.params['volume_spike_threshold']
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Volumen-Analyse: {e}", 'strategy')
            return {'volume_ratio': 1.0, 'is_volume_spike': False}
    
    def _calculate_trend_strength(self, df: pd.DataFrame, indicators: Dict[str, float]) -> Dict[str, float]:
        """Berechnet die Trend-Stärke"""
        
        try:
            close = df['close'].values
            
            # Linearer Trend
            if len(close) >= 20:
                trend_slope = np.polyfit(range(20), close[-20:], 1)[0]
                trend_strength = abs(trend_slope) / close[-1]
            else:
                trend_slope = 0
                trend_strength = 0
            
            # ADX-basierte Trend-Stärke
            adx = indicators.get('adx', 0)
            
            # Moving Average Konvergenz/Divergenz
            sma_20 = indicators.get('sma_20', close[-1])
            sma_50 = indicators.get('sma_50', close[-1])
            ma_divergence = abs(sma_20 - sma_50) / sma_50
            
            # Trend-Konsistenz
            trend_consistency = 0
            if len(close) >= 10:
                for i in range(-9, 0):
                    if close[i] < close[i-1]:
                        trend_consistency += 1
                trend_consistency = trend_consistency / 9
            
            return {
                'trend_slope': trend_slope,
                'trend_strength': trend_strength,
                'adx_strength': adx,
                'ma_divergence': ma_divergence,
                'trend_consistency': trend_consistency,
                'is_strong_downtrend': trend_strength > 0.001 and trend_slope < 0 and adx > 25
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei Trend-Stärke-Berechnung: {e}", 'strategy')
            return {'trend_strength': 0.0, 'is_strong_downtrend': False}
    
    def _calculate_downtrend_probability(self, indicators: Dict[str, float], 
                                       volatility: Dict[str, float],
                                       sentiment: Dict[str, float],
                                       volume: Dict[str, float],
                                       trend: Dict[str, float]) -> float:
        """Berechnet die Wahrscheinlichkeit eines Downtrends"""
        
        try:
            probability = 0.0
            
            # Technische Indikatoren (40% Gewichtung)
            tech_score = 0.0
            
            # RSI
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                tech_score += 0.3
            elif rsi < 40:
                tech_score += 0.2
            elif rsi > 70:
                tech_score -= 0.1
            
            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd < macd_signal and macd < 0:
                tech_score += 0.2
            
            # Bollinger Bands
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.2:
                tech_score += 0.2
            elif bb_position < 0.4:
                tech_score += 0.1
            
            # Preis vs. Moving Averages
            price_vs_sma20 = indicators.get('price_vs_sma20', 0)
            price_vs_sma50 = indicators.get('price_vs_sma50', 0)
            if price_vs_sma20 < -0.05 and price_vs_sma50 < -0.05:
                tech_score += 0.3
            elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
                tech_score += 0.1
            
            probability += tech_score * 0.4
            
            # Volatilität (20% Gewichtung)
            vol_score = 0.0
            if volatility.get('is_high_volatility', False):
                vol_score += 0.5
            if volatility.get('volatility_trend', 0) > 0:
                vol_score += 0.3
            
            probability += vol_score * 0.2
            
            # Sentiment (20% Gewichtung)
            sent_score = 0.0
            fear_greed = sentiment.get('fear_greed_proxy', 50)
            if fear_greed > 75:  # Hohe Angst
                sent_score += 0.6
            elif fear_greed > 60:
                sent_score += 0.3
            
            panic_score = sentiment.get('panic_score', 0)
            if panic_score >= 3:
                sent_score += 0.4
            elif panic_score >= 1:
                sent_score += 0.2
            
            probability += sent_score * 0.2
            
            # Volumen (10% Gewichtung)
            vol_analysis_score = 0.0
            if volume.get('volume_spikes_down', 0) >= 2:
                vol_analysis_score += 0.5
            if volume.get('obv_trend', 0) < 0:
                vol_analysis_score += 0.3
            
            probability += vol_analysis_score * 0.1
            
            # Trend-Stärke (10% Gewichtung)
            if trend.get('is_strong_downtrend', False):
                probability += 0.6 * 0.1
            elif trend.get('trend_slope', 0) < 0:
                probability += 0.3 * 0.1
            
            # Normalisierung auf 0-1
            probability = max(0.0, min(1.0, probability))
            
            return probability
            
        except Exception as e:
            self.logger.error(f"Fehler bei Downtrend-Wahrscheinlichkeits-Berechnung: {e}", 'strategy')
            return 0.5
    
    def _determine_market_condition(self, probability: float, indicators: Dict[str, float], 
                                  phase: DowntrendPhase) -> str:
        """Bestimmt die Markt-Condition basierend auf Analyse"""
        
        if probability > 0.7:
            return 'strong_downtrend'
        elif probability > 0.5:
            return 'weak_downtrend'
        elif probability > 0.3:
            return 'neutral_bearish'
        elif phase == DowntrendPhase.OVERSOLD:
            return 'oversold'
        elif phase == DowntrendPhase.RECOVERY_SIGNAL:
            return 'recovery_signal'
        elif phase == DowntrendPhase.RECOVERY:
            return 'recovery'
        else:
            return 'neutral'
    
    def generate_signals(self, df: pd.DataFrame, symbol: str, 
                        current_price: float, timeframe: str = '1h') -> List[DowntrendSignal]:
        """Generiert Handelssignale basierend auf Downtrend-Analyse"""
        
        signals = []
        
        try:
            # Marktanalyse durchführen
            analysis = self.analyze_market_condition(df, symbol, timeframe)
            
            if analysis['condition'] == 'error':
                return signals
            
            condition = analysis['condition']
            confidence = analysis['confidence']
            indicators = analysis['indicators']
            phase = analysis['phase']
            
            # Signal-Generierung basierend auf Condition
            if condition == 'strong_downtrend':
                signals.extend(self._generate_sell_signals(
                    current_price, indicators, confidence, 'strong'
                ))
                
            elif condition == 'weak_downtrend':
                signals.extend(self._generate_sell_signals(
                    current_price, indicators, confidence, 'weak'
                ))
                
            elif condition == 'oversold':
                signals.extend(self._generate_buy_signals(
                    current_price, indicators, confidence, 'oversold'
                ))
                
            elif condition == 'recovery_signal':
                signals.extend(self._generate_buy_signals(
                    current_price, indicators, confidence, 'recovery'
                ))
                
            elif condition == 'recovery':
                signals.extend(self._generate_buy_signals(
                    current_price, indicators, confidence, 'strong_recovery'
                ))
            
            # Signale bewerten und filtern
            signals = self._filter_and_rank_signals(signals, analysis)
            
            # Logging
            if signals:
                self.logger.log_strategy_decision(
                    strategy='downtrend_strategy',
                    symbol=symbol,
                    decision=f"{len(signals)} signals generated",
                    confidence=confidence,
                    market_condition=condition,
                    phase=phase,
                    signal_types=[s.signal_type for s in signals]
                )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Fehler bei Signal-Generierung: {e}", 'strategy')
            return []
    
    def _generate_sell_signals(self, current_price: float, indicators: Dict[str, float], 
                             confidence: float, strength: str) -> List[DowntrendSignal]:
        """Generiert Verkaufssignale"""
        
        signals = []
        
        try:
            # Basis-Parameter je nach Stärke
            if strength == 'strong':
                base_confidence = 0.8
                price_target_pct = 0.15  # 15% Preisrückgang erwartet
                stop_loss_pct = 0.03     # 3% Stop-Loss
                risk_level = 'medium'
            else:  # weak
                base_confidence = 0.6
                price_target_pct = 0.08  # 8% Preisrückgang erwartet
                stop_loss_pct = 0.04     # 4% Stop-Loss
                risk_level = 'low'
            
            # Haupt-Verkaufssignal
            signals.append(DowntrendSignal(
                signal_type='SELL',
                strength=confidence,
                confidence=min(confidence * base_confidence, 0.95),
                price_target=current_price * (1 - price_target_pct),
                stop_loss=current_price * (1 + stop_loss_pct),
                timeframe='1h',
                indicators=indicators,
                risk_level=risk_level,
                expected_duration=24 if strength == 'strong' else 12
            ))
            
            # Zusätzliche Signale basierend auf spezifischen Bedingungen
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            bb_position = indicators.get('bb_position', 0.5)
            
            # Kurzfristiges Verkaufssignal bei Überkauf
            if rsi > 65 and bb_position > 0.8:
                signals.append(DowntrendSignal(
                    signal_type='SHORT_SELL',
                    strength=0.7,
                    confidence=0.6,
                    price_target=current_price * 0.98,
                    stop_loss=current_price * 1.02,
		    timeframe='15m',
                    indicators=indicators,
                    risk_level='low',
                    expected_duration=2
                ))
            
            # DCA-Verkaufssignal bei starkem Downtrend
            if strength == 'strong' and macd < -0.002:
                signals.append(DowntrendSignal(
                    signal_type='DCA_SELL',
                    strength=0.8,
                    confidence=0.7,
                    price_target=current_price * 0.90,
                    stop_loss=current_price * 1.05,
                    timeframe='4h',
                    indicators=indicators,
                    risk_level='medium',
                    expected_duration=48
                ))
                
        except Exception as e:
            self.logger.error(f"Fehler bei Verkaufssignal-Generierung: {e}", 'strategy')
        
        return signals
    
    def _generate_buy_signals(self, current_price: float, indicators: Dict[str, float], 
                            confidence: float, signal_type: str) -> List[DowntrendSignal]:
        """Generiert Kaufsignale für Erholung/Oversold"""
        
        signals = []
        
        try:
            # Parameter basierend auf Signal-Typ
            if signal_type == 'oversold':
                base_confidence = 0.7
                price_target_pct = 0.12  # 12% Gewinn erwartet
                stop_loss_pct = 0.06     # 6% Stop-Loss
                risk_level = 'medium'
                duration = 12
                
            elif signal_type == 'recovery':
                base_confidence = 0.65
                price_target_pct = 0.08  # 8% Gewinn erwartet
                stop_loss_pct = 0.05     # 5% Stop-Loss
                risk_level = 'medium'
                duration = 8
                
            else:  # strong_recovery
                base_confidence = 0.8
                price_target_pct = 0.15  # 15% Gewinn erwartet
                stop_loss_pct = 0.04     # 4% Stop-Loss
                risk_level = 'low'
                duration = 24
            
            # Haupt-Kaufsignal
            signals.append(DowntrendSignal(
                signal_type='BUY',
                strength=confidence,
                confidence=min(confidence * base_confidence, 0.95),
                price_target=current_price * (1 + price_target_pct),
                stop_loss=current_price * (1 - stop_loss_pct),
                timeframe='1h',
                indicators=indicators,
                risk_level=risk_level,
                expected_duration=duration
            ))
            
            # Spezielle Signale basierend auf Indikatoren
            rsi = indicators.get('rsi', 50)
            bb_position = indicators.get('bb_position', 0.5)
            williams_r = indicators.get('williams_r', -50)
            
            # Extreme Oversold Signal
            if rsi < 20 and bb_position < 0.1 and williams_r < -80:
                signals.append(DowntrendSignal(
                    signal_type='EXTREME_BUY',
                    strength=0.9,
                    confidence=0.8,
                    price_target=current_price * 1.20,
                    stop_loss=current_price * 0.92,
                    timeframe='4h',
                    indicators=indicators,
                    risk_level='high',
                    expected_duration=48
                ))
            
            # Divergenz-basiertes Signal (vereinfacht)
            if signal_type == 'recovery' and rsi > 35:
                signals.append(DowntrendSignal(
                    signal_type='DIVERGENCE_BUY',
                    strength=0.7,
                    confidence=0.6,
                    price_target=current_price * 1.10,
                    stop_loss=current_price * 0.95,
                    timeframe='2h',
                    indicators=indicators,
                    risk_level='medium',
                    expected_duration=16
                ))
                
        except Exception as e:
            self.logger.error(f"Fehler bei Kaufsignal-Generierung: {e}", 'strategy')
        
        return signals
    
    def _filter_and_rank_signals(self, signals: List[DowntrendSignal], 
                                analysis: Dict[str, Any]) -> List[DowntrendSignal]:
        """Filtert und rankt Signale nach Qualität"""
        
        if not signals:
            return signals
        
        try:
            # Signale bewerten
            scored_signals = []
            
            for signal in signals:
                score = self._calculate_signal_score(signal, analysis)
                scored_signals.append((signal, score))
            
            # Nach Score sortieren (höchster zuerst)
            scored_signals.sort(key=lambda x: x[1], reverse=True)
            
            # Nur die besten Signale behalten (max 3)
            filtered_signals = []
            signal_types_used = set()
            
            for signal, score in scored_signals[:5]:  # Top 5 betrachten
                # Duplikate vermeiden
                if signal.signal_type not in signal_types_used:
                    filtered_signals.append(signal)
                    signal_types_used.add(signal.signal_type)
                
                if len(filtered_signals) >= 3:
                    break
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Fehler beim Filtern der Signale: {e}", 'strategy')
            return signals[:3]  # Fallback: erste 3 Signale
    
    def _calculate_signal_score(self, signal: DowntrendSignal, 
                              analysis: Dict[str, Any]) -> float:
        """Berechnet Score für Signal-Qualität"""
        
        try:
            score = 0.0
            
            # Basis-Score aus Confidence und Strength
            score += signal.confidence * 0.4
            score += signal.strength * 0.3
            
            # Risk-Reward Ratio
            if signal.signal_type in ['BUY', 'EXTREME_BUY', 'DIVERGENCE_BUY']:
                risk = signal.stop_loss / signal.price_target  # Umgekehrt für Kaufsignale
                reward = signal.price_target / signal.stop_loss
            else:
                risk = signal.stop_loss / signal.price_target
                reward = signal.price_target / signal.stop_loss
            
            risk_reward = reward / max(risk, 0.01)
            score += min(risk_reward / 3.0, 0.2)  # Max 0.2 Punkte für R/R
            
            # Markt-Kontext Bonus
            condition = analysis.get('condition', 'neutral')
            if signal.signal_type == 'SELL' and condition in ['strong_downtrend', 'weak_downtrend']:
                score += 0.1
            elif signal.signal_type in ['BUY', 'EXTREME_BUY'] and condition in ['oversold', 'recovery_signal']:
                score += 0.1
            
            # Volatilitäts-Adjustment
            volatility = analysis.get('volatility', {})
            if volatility.get('is_high_volatility', False):
                if signal.risk_level == 'high':
                    score -= 0.05  # Penalty für hohes Risiko bei hoher Volatilität
                elif signal.risk_level == 'low':
                    score += 0.05  # Bonus für niedriges Risiko
            
            # Zeitrahmen-Konsistenz
            phase = analysis.get('phase', '')
            if signal.expected_duration <= 4 and phase in ['accelerating', 'early_decline']:
                score += 0.05  # Kurzfristige Signale in volatilen Phasen
            elif signal.expected_duration >= 24 and phase in ['oversold', 'recovery']:
                score += 0.05  # Längerfristige Signale in Erholungsphasen
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Fehler bei Signal-Score-Berechnung: {e}", 'strategy')
            return 0.5
    
    def calculate_position_size(self, signal: DowntrendSignal, account_balance: float, 
                              current_price: float) -> float:
        """Berechnet optimale Positionsgröße"""
        
        try:
            # Risiko pro Trade
            risk_per_trade = self.params['risk_per_trade']
            max_position_size = self.params['max_position_size']
            
            # Stop-Loss Distanz berechnen
            if signal.signal_type in ['BUY', 'EXTREME_BUY', 'DIVERGENCE_BUY']:
                stop_distance = abs(current_price - signal.stop_loss) / current_price
            else:
                stop_distance = abs(signal.stop_loss - current_price) / current_price
            
            # Positionsgröße basierend auf Risiko
            risk_amount = account_balance * risk_per_trade
            position_value = risk_amount / max(stop_distance, 0.01)
            position_size = position_value / account_balance
            
            # Begrenzungen anwenden
            position_size = min(position_size, max_position_size)
            
            # Adjustment basierend auf Signal-Qualität
            confidence_multiplier = 0.5 + (signal.confidence * 0.5)  # 0.5 - 1.0
            position_size *= confidence_multiplier
            
            # Risk-Level Adjustment
            risk_multipliers = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.8
            }
            position_size *= risk_multipliers.get(signal.risk_level, 1.0)
            
            # Minimum Position Size
            position_size = max(position_size, 0.01)  # Mindestens 1%
            
            return round(position_size, 4)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Positionsgrößen-Berechnung: {e}", 'strategy')
            return 0.05  # 5% Fallback
    
    def update_strategy_performance(self, trade_result: Dict[str, Any]):
        """Aktualisiert Performance-Metriken der Strategie"""
        
        try:
            self.performance_metrics['total_trades'] += 1
            
            profit = trade_result.get('profit', 0.0)
            self.performance_metrics['total_profit'] += profit
            
            if profit > 0:
                self.performance_metrics['winning_trades'] += 1
                if profit > self.performance_metrics['best_trade']:
                    self.performance_metrics['best_trade'] = profit
            else:
                self.performance_metrics['losing_trades'] += 1
                if profit < self.performance_metrics['worst_trade']:
                    self.performance_metrics['worst_trade'] = profit
            
            # Win Rate berechnen
            total_trades = self.performance_metrics['total_trades']
            if total_trades > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['winning_trades'] / total_trades
                )
                self.performance_metrics['avg_profit_per_trade'] = (
                    self.performance_metrics['total_profit'] / total_trades
                )
            
            # Trade zu Historie hinzufügen
            trade_result['timestamp'] = datetime.now().isoformat()
            self.trade_history.append(trade_result)
            
            # Historie begrenzen
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            # Recovery Success Rate (für Buy-Signale)
            if trade_result.get('signal_type') in ['BUY', 'EXTREME_BUY', 'DIVERGENCE_BUY']:
                recovery_trades = [t for t in self.trade_history 
                                 if t.get('signal_type') in ['BUY', 'EXTREME_BUY', 'DIVERGENCE_BUY']]
                if recovery_trades:
                    successful_recovery = sum(1 for t in recovery_trades if t.get('profit', 0) > 0)
                    self.performance_metrics['recovery_success_rate'] = (
                        successful_recovery / len(recovery_trades)
                    )
                    
            self.logger.info("Strategy Performance aktualisiert", 'strategy',
                           performance=self.performance_metrics)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Performance-Update: {e}", 'strategy')
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Status der Strategie zurück"""
        
        return {
            'name': 'Downtrend Strategy',
            'current_phase': self.current_phase.value,
            'active_positions': len(self.positions),
            'performance': self.performance_metrics.copy(),
            'parameters': self.params.copy(),
            'memory_size': len(self.market_memory),
            'trade_history_size': len(self.trade_history),
            'last_analysis': self.market_memory[-1] if self.market_memory else None
        }
    
    def optimize_parameters(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimiert Strategie-Parameter basierend auf historischen Daten"""
        
        try:
            self.logger.info("Starte Parameter-Optimierung für Downtrend Strategy", 'strategy')
            
            # Parameter-Bereiche für Optimierung
            param_ranges = {
                'rsi_threshold': [25, 30, 35],
                'rsi_oversold': [15, 20, 25],
                'macd_threshold': [-0.002, -0.001, -0.0005],
                'decline_threshold': [0.02, 0.03, 0.04],
                'stop_loss_pct': [0.03, 0.05, 0.07],
                'take_profit_pct': [0.06, 0.08, 0.10],
                'volatility_threshold': [0.02, 0.03, 0.04],
                'volume_spike_threshold': [1.5, 2.0, 2.5]
            }
            
            best_params = None
            best_score = -float('inf')
            
            # Einfache Grid-Search (begrenzt für Performance)
            test_combinations = 0
            max_combinations = 50  # Begrenzen für Laufzeit
            
            import itertools
            param_keys = list(param_ranges.keys())
            
            for combination in itertools.product(*[param_ranges[key] for key in param_keys]):
                if test_combinations >= max_combinations:
                    break
                
                # Test-Parameter setzen
                test_params = self.params.copy()
                for i, key in enumerate(param_keys):
                    test_params[key] = combination[i]
                
                # Einfacher Backtest
                score = self._simple_backtest(historical_data, test_params)
                
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                
                test_combinations += 1
            
            # Beste Parameter übernehmen falls gefunden
            if best_params and best_score > 0:
                old_params = self.params.copy()
                self.params.update(best_params)
                
                self.logger.info("Parameter-Optimierung abgeschlossen", 'strategy',
                               old_params=old_params,
                               new_params=self.params,
                               improvement_score=best_score)
                
                return {
                    'optimized': True,
                    'improvement_score': best_score,
                    'old_params': old_params,
                    'new_params': self.params.copy()
                }
            else:
                self.logger.info("Keine Verbesserung durch Parameter-Optimierung gefunden", 'strategy')
                return {'optimized': False}
                
        except Exception as e:
            self.logger.error(f"Fehler bei Parameter-Optimierung: {e}", 'strategy')
            return {'optimized': False, 'error': str(e)}
    
    def _simple_backtest(self, data: pd.DataFrame, test_params: Dict[str, Any]) -> float:
        """Einfacher Backtest für Parameter-Optimierung"""
        
        try:
            if len(data) < 100:
                return 0.0
            
            # Temporäre Parameter setzen
            original_params = self.params.copy()
            self.params = test_params
            
            total_return = 0.0
            trades = 0
            wins = 0
            
            # Durch Daten iterieren (vereinfacht)
            for i in range(50, len(data) - 10, 24):  # Jeden Tag testen
                try:
                    df_slice = data.iloc[i-50:i+1]
                    current_price = data.iloc[i]['close']
                    
                    # Signale generieren
                    signals = self.generate_signals(df_slice, 'TEST', current_price)
                    
                    if signals:
                        signal = signals[0]  # Bestes Signal nehmen
                        
                        # Einfache Gewinn/Verlust-Simulation
                        if signal.signal_type in ['SELL', 'SHORT_SELL']:
                            # Verkaufssignal - prüfe ob Preis in nächsten Stunden fällt
                            future_prices = data.iloc[i+1:i+min(signal.expected_duration, 10)]['close']
                            if len(future_prices) > 0:
                                min_price = future_prices.min()
                                if min_price < signal.price_target:
                                    profit = (current_price - min_price) / current_price
                                    total_return += profit
                                    wins += 1
                                else:
                                    total_return -= 0.02  # Kleiner Verlust
                                trades += 1
                                
                        elif signal.signal_type in ['BUY', 'EXTREME_BUY']:
                            # Kaufsignal - prüfe ob Preis in nächsten Stunden steigt
                            future_prices = data.iloc[i+1:i+min(signal.expected_duration, 10)]['close']
                            if len(future_prices) > 0:
                                max_price = future_prices.max()
                                if max_price > signal.price_target:
                                    profit = (max_price - current_price) / current_price
                                    total_return += profit
                                    wins += 1
                                else:
                                    total_return -= 0.02  # Kleiner Verlust
                                trades += 1
                                
                except Exception:
                    continue
            
            # Parameter zurücksetzen
            self.params = original_params
            
            # Score berechnen
            if trades > 0:
                win_rate = wins / trades
                avg_return = total_return / trades
                score = (win_rate * 0.6) + (avg_return * 0.4)
                return score
            else:
                return 0.0
                
        except Exception as e:
            self.params = original_params  # Parameter zurücksetzen bei Fehler
            return 0.0
    
    def adapt_to_market_regime(self, market_regime: str, regime_confidence: float):
        """Passt Strategie an Marktregime an"""
        
        try:
            self.logger.info(f"Adapting Downtrend Strategy to market regime: {market_regime}", 'strategy')
            
            if market_regime == 'bearish' and regime_confidence > 0.7:
                # Aggressivere Parameter für Bärenmarkt
                self.params['rsi_threshold'] = 35
                self.params['decline_threshold'] = 0.025
                self.params['max_position_size'] = 0.4
                self.params['volume_spike_threshold'] = 1.8
                
            elif market_regime == 'volatile' and regime_confidence > 0.6:
                # Vorsichtigere Parameter für volatilen Markt
                self.params['stop_loss_pct'] = 0.06
                self.params['volatility_threshold'] = 0.025
                self.params['max_position_size'] = 0.25
                self.params['min_hold_time'] = 2
                
            elif market_regime == 'recovery' and regime_confidence > 0.6:
                # Fokus auf Erholungs-Signale
                self.params['rsi_oversold'] = 25
                self.params['take_profit_pct'] = 0.12
                self.params['recovery_confirmation'] = 2
                
            elif market_regime == 'sideways':
                # Neutrale Parameter für Seitwärtsmarkt
                self.params['decline_threshold'] = 0.035
                self.params['max_position_size'] = 0.2
                self.params['min_hold_time'] = 6
            
            self.logger.info("Downtrend Strategy adapted to market regime", 'strategy',
                           regime=market_regime,
                           confidence=regime_confidence,
                           new_params=self.params)
                           
        except Exception as e:
            self.logger.error(f"Fehler bei Marktregime-Anpassung: {e}", 'strategy')
    
    def cleanup_old_data(self):
        """Bereinigt alte Daten um Speicher zu sparen"""
        
        try:
            # Market Memory begrenzen
            if len(self.market_memory) > 500:
                self.market_memory = self.market_memory[-500:]
            
            # Trade History begrenzen
            if len(self.trade_history) > 500:
                self.trade_history = self.trade_history[-500:]
            
            # Fear & Greed History begrenzen
            if len(self.fear_greed_history) > 200:
                self.fear_greed_history = self.fear_greed_history[-200:]
            
            # Volatility History begrenzen
            if len(self.volatility_history) > 200:
                self.volatility_history = self.volatility_history[-200:]
            
            self.logger.debug("Alte Daten bereinigt", 'strategy')
            
        except Exception as e:
            self.logger.error(f"Fehler bei Datenbereinigung: {e}", 'strategy')
    
    def get_market_insights(self) -> Dict[str, Any]:
        """Gibt Markt-Insights basierend auf gesammelten Daten zurück"""
        
        try:
            insights = {
                'total_analyses': len(self.market_memory),
                'common_conditions': {},
                'avg_confidence': 0.0,
                'phase_distribution': {},
                'recent_trends': {}
            }
            
            if not self.market_memory:
                return insights
            
            # Häufigste Bedingungen
            conditions = [analysis.get('condition', 'unknown') for analysis in self.market_memory]
            for condition in set(conditions):
                insights['common_conditions'][condition] = conditions.count(condition)
            
            # Durchschnittliche Confidence
            confidences = [analysis.get('confidence', 0) for analysis in self.market_memory]
            insights['avg_confidence'] = np.mean(confidences) if confidences else 0.0
            
            # Phasen-Verteilung
            phases = [analysis.get('phase', 'unknown') for analysis in self.market_memory]
            for phase in set(phases):
                insights['phase_distribution'][phase] = phases.count(phase)
            
            # Aktuelle Trends (letzte 20 Analysen)
            recent_analyses = self.market_memory[-20:] if len(self.market_memory) >= 20 else self.market_memory
            recent_conditions = [analysis.get('condition', 'unknown') for analysis in recent_analyses]
            recent_confidences = [analysis.get('confidence', 0) for analysis in recent_analyses]
            
            insights['recent_trends'] = {
                'dominant_condition': max(set(recent_conditions), key=recent_conditions.count) if recent_conditions else 'unknown',
                'avg_recent_confidence': np.mean(recent_confidences) if recent_confidences else 0.0,
                'trend_consistency': len(set(recent_conditions[-5:])) if len(recent_conditions) >= 5 else 0
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Fehler bei Markt-Insights-Generierung: {e}", 'strategy')
            return {'error': str(e)}