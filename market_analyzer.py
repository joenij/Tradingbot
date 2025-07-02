"""
Trading Bot Market Analyzer
Intelligente Marktanalyse mit technischen Indikatoren und ML-basierter Trendidentifikation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib nicht verfügbar. Fallback auf eigene Implementierungen.")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import json

class MarketCondition(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

@dataclass
class MarketSignal:
    condition: MarketCondition
    confidence: float
    strength: float
    indicators: Dict[str, float]
    timestamp: datetime
    recommendation: str
    support_levels: List[float]
    resistance_levels: List[float]

@dataclass
class TechnicalIndicators:
    # Trend-Indikatoren
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Momentum-Indikatoren
    rsi: float = 0.0
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    williams_r: float = 0.0
    roc: float = 0.0
    
    # Volatilität-Indikatoren
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    atr: float = 0.0
    
    # Volumen-Indikatoren
    volume_sma: float = 0.0
    volume_ratio: float = 0.0
    obv: float = 0.0
    
    # Support/Resistance
    pivot_point: float = 0.0
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    support_1: float = 0.0
    support_2: float = 0.0

class MarketAnalyzer:
    def __init__(self, config_manager=None, logger=None, data_manager=None):
        self.config = config_manager
        self.logger = logger
        self.data_manager = data_manager
        
        # ML-Modelle für Marktbedingungen
        self.trend_classifier = None
        self.volatility_predictor = None
        self.price_direction_model = None
        
        # Scaler für Feature-Normalisierung
        self.feature_scaler = StandardScaler()
        
        # Model-Pfade
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Cache für berechnete Indikatoren
        self.indicator_cache = {}
        self.analysis_history = []
        
        # Konfiguration
        self.lookback_periods = {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        
        # Schwellenwerte für Marktbedingungen
        self.thresholds = {
            'trend_strength': 0.6,
            'volatility_high': 0.02,
            'volatility_low': 0.005,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'confidence_min': 0.6
        }
        
        self._load_models()
        
        if self.logger:
            self.logger.info("MarketAnalyzer initialisiert", 'strategy')
    
    def analyze_market(self, symbol: str, timeframe: str = '1h', 
                      periods: int = 500) -> MarketSignal:
        """Hauptmethode für Marktanalyse"""
        try:
            # Daten abrufen
            if not self.data_manager:
                raise ValueError("DataManager nicht verfügbar")
            
            df = self.data_manager.get_klines(symbol, timeframe, limit=periods)
            if df is None or len(df) < 50:
                if self.logger:
                    self.logger.warning(f"Nicht genügend Daten für {symbol}", 'strategy')
                return self._create_unknown_signal(symbol)
            
            # Technische Indikatoren berechnen
            indicators = self._calculate_indicators(df)
            
            # ML-basierte Marktbewertung
            market_condition, confidence = self._classify_market_condition(df, indicators)
            
            # Trend-Stärke bestimmen
            trend_strength = self._calculate_trend_strength(df, indicators)
            
            # Support/Resistance Levels
            support_levels, resistance_levels = self._find_support_resistance(df)
            
            # Handelssignal generieren
            recommendation = self._generate_recommendation(
                market_condition, confidence, trend_strength, indicators
            )
            
            # Signal erstellen
            signal = MarketSignal(
                condition=market_condition,
                confidence=confidence,
                strength=trend_strength,
                indicators=self._indicators_to_dict(indicators),
                timestamp=datetime.now(),
                recommendation=recommendation,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )
            
            # In Historie speichern
            self.analysis_history.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'timestamp': datetime.now()
            })
            
            # Historie begrenzen
            if len(self.analysis_history) > 1000:
                self.analysis_history = self.analysis_history[-500:]
            
            if self.logger:
                self.logger.log_market_analysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    market_condition=market_condition.value,
                    confidence=confidence,
                    indicators=signal.indicators
                )
            
            return signal
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler bei Marktanalyse für {symbol}")
            return self._create_unknown_signal(symbol)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Berechnet alle technischen Indikatoren"""
        indicators = TechnicalIndicators()
        
        # Preise extrahieren
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
        
        try:
            # Moving Averages
            if TALIB_AVAILABLE:
                indicators.sma_20 = talib.SMA(close, timeperiod=20)[-1]
                indicators.sma_50 = talib.SMA(close, timeperiod=50)[-1]
                indicators.sma_200 = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else close[-1]
                indicators.ema_12 = talib.EMA(close, timeperiod=12)[-1]
                indicators.ema_26 = talib.EMA(close, timeperiod=26)[-1]
            else:
                indicators.sma_20 = self._sma(close, 20)
                indicators.sma_50 = self._sma(close, 50)
                indicators.sma_200 = self._sma(close, 200) if len(close) >= 200 else close[-1]
                indicators.ema_12 = self._ema(close, 12)
                indicators.ema_26 = self._ema(close, 26)
            
            # MACD
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators.macd = macd[-1]
                indicators.macd_signal = macd_signal[-1]
                indicators.macd_histogram = macd_hist[-1]
            else:
                macd_line = indicators.ema_12 - indicators.ema_26
                indicators.macd = macd_line
                indicators.macd_signal = macd_line  # Vereinfachung
                indicators.macd_histogram = 0
            
            # RSI
            if TALIB_AVAILABLE:
                indicators.rsi = talib.RSI(close, timeperiod=14)[-1]
            else:
                indicators.rsi = self._rsi(close, 14)
            
            # Stochastic
            if TALIB_AVAILABLE:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                indicators.stoch_k = slowk[-1]
                indicators.stoch_d = slowd[-1]
            else:
                indicators.stoch_k, indicators.stoch_d = self._stochastic(high, low, close, 14)
            
            # Williams %R
            if TALIB_AVAILABLE:
                indicators.williams_r = talib.WILLR(high, low, close, timeperiod=14)[-1]
            else:
                indicators.williams_r = self._williams_r(high, low, close, 14)
            
            # Rate of Change
            if len(close) >= 12:
                indicators.roc = ((close[-1] - close[-12]) / close[-12]) * 100
            
            # Bollinger Bands
            if TALIB_AVAILABLE:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators.bb_upper = bb_upper[-1]
                indicators.bb_middle = bb_middle[-1]
                indicators.bb_lower = bb_lower[-1]
                indicators.bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            else:
                bb_upper, bb_middle, bb_lower = self._bollinger_bands(close, 20, 2)
                indicators.bb_upper = bb_upper
                indicators.bb_middle = bb_middle
                indicators.bb_lower = bb_lower
                indicators.bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            
            # ATR
            if TALIB_AVAILABLE:
                indicators.atr = talib.ATR(high, low, close, timeperiod=14)[-1]
            else:
                indicators.atr = self._atr(high, low, close, 14)
            
            # Volume-Indikatoren
            if len(volume) >= 20:
                indicators.volume_sma = np.mean(volume[-20:])
                indicators.volume_ratio = volume[-1] / indicators.volume_sma if indicators.volume_sma > 0 else 1
                
                if TALIB_AVAILABLE:
                    indicators.obv = talib.OBV(close, volume)[-1]
                else:
                    indicators.obv = self._obv(close, volume)
            
            # Pivot Points
            if len(df) >= 1:
                prev_high = high[-2] if len(high) >= 2 else high[-1]
                prev_low = low[-2] if len(low) >= 2 else low[-1]
                prev_close = close[-2] if len(close) >= 2 else close[-1]
                
                indicators.pivot_point = (prev_high + prev_low + prev_close) / 3
                indicators.resistance_1 = 2 * indicators.pivot_point - prev_low
                indicators.resistance_2 = indicators.pivot_point + (prev_high - prev_low)
                indicators.support_1 = 2 * indicators.pivot_point - prev_high
                indicators.support_2 = indicators.pivot_point - (prev_high - prev_low)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Fehler bei Indikator-Berechnung: {e}", 'strategy')
        
        return indicators
    
    def _classify_market_condition(self, df: pd.DataFrame, 
                                 indicators: TechnicalIndicators) -> Tuple[MarketCondition, float]:
        """Klassifiziert Marktbedingungen mit ML"""
        
        try:
            # Features für ML-Modell extrahieren
            features = self._extract_features(df, indicators)
            
            if self.trend_classifier is not None and len(features) > 0:
                # ML-Vorhersage
                features_scaled = self.feature_scaler.transform([features])
                prediction = self.trend_classifier.predict(features_scaled)[0]
                confidence = max(self.trend_classifier.predict_proba(features_scaled)[0])
                
                # Prediction zu MarketCondition mappen
                condition_map = {0: MarketCondition.DOWNTREND, 1: MarketCondition.SIDEWAYS, 2: MarketCondition.UPTREND}
                market_condition = condition_map.get(prediction, MarketCondition.UNKNOWN)
            else:
                # Fallback: Regelbasierte Klassifikation
                market_condition, confidence = self._rule_based_classification(df, indicators)
            
            return market_condition, confidence
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Fehler bei ML-Klassifikation: {e}", 'strategy')
            return self._rule_based_classification(df, indicators)
    
    def _rule_based_classification(self, df: pd.DataFrame, 
                                 indicators: TechnicalIndicators) -> Tuple[MarketCondition, float]:
        """Regelbasierte Marktklassifikation als Fallback"""
        
        close = df['close'].values
        current_price = close[-1]
        
        # Trend-Analyse basierend auf Moving Averages
        uptrend_signals = 0
        downtrend_signals = 0
        sideways_signals = 0
        
        # MA-Arrangement prüfen
        if current_price > indicators.sma_20 > indicators.sma_50:
            uptrend_signals += 2
        elif current_price < indicators.sma_20 < indicators.sma_50:
            downtrend_signals += 2
        else:
            sideways_signals += 1
        
        # MACD-Signal
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            uptrend_signals += 1
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            downtrend_signals += 1
        else:
            sideways_signals += 1
        
        # RSI-Bereich
        if indicators.rsi > 50 and indicators.rsi < 70:
            uptrend_signals += 1
        elif indicators.rsi < 50 and indicators.rsi > 30:
            downtrend_signals += 1
        else:
            sideways_signals += 1
        
        # Preismuster der letzten Kerzen
        if len(close) >= 5:
            recent_trend = (close[-1] - close[-5]) / close[-5]
            if recent_trend > 0.02:  # 2% Anstieg
                uptrend_signals += 1
            elif recent_trend < -0.02:  # 2% Rückgang
                downtrend_signals += 1
            else:
                sideways_signals += 1
        
        # Volatilität prüfen
        volatility = indicators.atr / current_price if current_price > 0 else 0
        if volatility > self.thresholds['volatility_high']:
            if uptrend_signals > downtrend_signals:
                market_condition = MarketCondition.VOLATILE
            else:
                market_condition = MarketCondition.VOLATILE
        elif volatility < self.thresholds['volatility_low']:
            market_condition = MarketCondition.SIDEWAYS
        else:
            # Normale Volatilität - Trend bestimmen
            if uptrend_signals > downtrend_signals and uptrend_signals > sideways_signals:
                market_condition = MarketCondition.UPTREND
            elif downtrend_signals > uptrend_signals and downtrend_signals > sideways_signals:
                market_condition = MarketCondition.DOWNTREND
            else:
                market_condition = MarketCondition.SIDEWAYS
        
        # Confidence basierend auf Signal-Eindeutigkeit
        total_signals = uptrend_signals + downtrend_signals + sideways_signals
        if total_signals > 0:
            max_signals = max(uptrend_signals, downtrend_signals, sideways_signals)
            confidence = max_signals / total_signals
        else:
            confidence = 0.5
        
        return market_condition, confidence
    
    def _calculate_trend_strength(self, df: pd.DataFrame, 
                                indicators: TechnicalIndicators) -> float:
        """Berechnet die Stärke des aktuellen Trends"""
        
        try:
            close = df['close'].values
            strength_factors = []
            
            # MA-Slope (Steigung)
            if len(close) >= 20:
                ma_slope = (indicators.sma_20 - np.mean(close[-40:-20])) / np.mean(close[-40:-20])
                strength_factors.append(abs(ma_slope) * 10)  # Normierung
            
            # MACD-Histogram Momentum
            if indicators.macd_histogram != 0:
                strength_factors.append(abs(indicators.macd_histogram) / indicators.atr if indicators.atr > 0 else 0)
            
            # RSI-Abstand von Mittellinie
            rsi_strength = abs(indicators.rsi - 50) / 50
            strength_factors.append(rsi_strength)
            
            # Preis-Volatilität
            if indicators.atr > 0 and close[-1] > 0:
                vol_strength = indicators.atr / close[-1]
                strength_factors.append(min(vol_strength * 10, 1.0))  # Cap bei 1.0
            
            # Volume-Bestätigung
            if indicators.volume_ratio > 0:
                vol_confirmation = min(indicators.volume_ratio / 2, 1.0)  # Cap bei 1.0
                strength_factors.append(vol_confirmation)
            
            # Durchschnittliche Stärke
            if strength_factors:
                trend_strength = np.mean(strength_factors)
                return min(max(trend_strength, 0.0), 1.0)  # Zwischen 0 und 1
            else:
                return 0.5
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Fehler bei Trend-Stärke-Berechnung: {e}", 'strategy')
            return 0.5
    
    def _find_support_resistance(self, df: pd.DataFrame, 
                               lookback: int = 50) -> Tuple[List[float], List[float]]:
        """Findet Support- und Resistance-Levels"""
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Lokale Hochs und Tiefs finden
            highs = []
            lows = []
            
            window = min(5, len(high) // 10)  # Adaptives Fenster
            
            for i in range(window, len(high) - window):
                # Lokale Hochs
                if all(high[i] >= high[i-j] for j in range(1, window+1)) and \
                   all(high[i] >= high[i+j] for j in range(1, window+1)):
                    highs.append(high[i])
                
                # Lokale Tiefs
                if all(low[i] <= low[i-j] for j in range(1, window+1)) and \
                   all(low[i] <= low[i+j] for j in range(1, window+1)):
                    lows.append(low[i])
            
            # Nach Häufigkeit clustern (Support/Resistance durch mehrfache Berührung)
            current_price = close[-1]
            tolerance = 0.02  # 2% Toleranz
            
            resistance_levels = []
            support_levels = []
            
            # Resistance-Levels (über aktuellem Preis)
            for high_level in highs:
                if high_level > current_price * (1 + tolerance * 0.5):
                    # Prüfe wie oft dieses Level getestet wurde
                    touches = sum(1 for h in high for h in high if abs(h - high_level) / high_level < tolerance)
                    if touches >= 2:  # Mindestens 2 Berührungen
                        resistance_levels.append(high_level)
            
            # Support-Levels (unter aktuellem Preis)
            for low_level in lows:
                if low_level < current_price * (1 - tolerance * 0.5):
                    touches = sum(1 for l in low if abs(l - low_level) / low_level < tolerance)
                    if touches >= 2:
                        support_levels.append(low_level)
            
            # Nach Nähe zum aktuellen Preis sortieren und begrenzen
            resistance_levels = sorted(resistance_levels)[:3]
            support_levels = sorted(support_levels, reverse=True)[:3]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Fehler bei Support/Resistance-Berechnung: {e}", 'strategy')
            return [], []
    
    def _generate_recommendation(self, market_condition: MarketCondition, 
                               confidence: float, trend_strength: float,
                               indicators: TechnicalIndicators) -> str:
        """Generiert Handelsempfehlung basierend auf Analyse"""
        
        if confidence < self.thresholds['confidence_min']:
            return "WAIT - Niedrige Confidence, Marktrichtung unklar"
        
        if market_condition == MarketCondition.UPTREND:
            if trend_strength > 0.7:
                return "STRONG_BUY - Starker Aufwärtstrend"
            elif indicators.rsi < 70:
                return "BUY - Aufwärtstrend mit Einstiegschance"
            else:
                return "HOLD - Aufwärtstrend aber überkauft"
        
        elif market_condition == MarketCondition.DOWNTREND:
            if trend_strength > 0.7:
                return "STRONG_SELL - Starker Abwärtstrend"
            elif indicators.rsi > 30:
                return "SELL - Abwärtstrend, Verkaufsdruck"
            else:
                return "WAIT - Abwärtstrend aber überverkauft"
        
        elif market_condition == MarketCondition.SIDEWAYS:
            if indicators.rsi < 35:
                return "BUY - Seitwärtstrend, überverkauft"
            elif indicators.rsi > 65:
                return "SELL - Seitwärtstrend, überkauft"
            else:
                return "GRID - Ideal für Grid-Trading"
        
        elif market_condition == MarketCondition.VOLATILE:
            return "CAUTION - Hohe Volatilität, vorsichtig handeln"
        
        else:
            return "WAIT - Marktrichtung unklar"
    
    def _extract_features(self, df: pd.DataFrame, 
                         indicators: TechnicalIndicators) -> List[float]:
        """Extrahiert Features für ML-Modelle"""
        
        try:
            close = df['close'].values
            features = []
            
            # Preis-Features
            if len(close) >= 20:
                features.extend([
                    (close[-1] - close[-5]) / close[-5],  # 5-Perioden Return
                    (close[-1] - close[-10]) / close[-10],  # 10-Perioden Return
                    (close[-1] - close[-20]) / close[-20],  # 20-Perioden Return
                ])
            else:
                features.extend([0, 0, 0])
            
            # Indikator-Features
            features.extend([
                indicators.rsi / 100,  # Normiert auf 0-1
                indicators.stoch_k / 100,
                indicators.williams_r / -100,  # Normiert auf 0-1
                indicators.macd_histogram,
                indicators.bb_width,
                indicators.volume_ratio if indicators.volume_ratio > 0 else 1,
            ])
            
            # MA-Verhältnisse
            if indicators.sma_20 > 0 and indicators.sma_50 > 0:
                features.append(indicators.sma_20 / indicators.sma_50 - 1)
            else:
                features.append(0)
            
            # Volatilität
            if close[-1] > 0:
                features.append(indicators.atr / close[-1])
            else:
                features.append(0)
            
            return features
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Fehler bei Feature-Extraktion: {e}", 'strategy')
            return []
    
    def train_models(self, symbol: str, timeframe: str = '1h', 
                    train_periods: int = 10000) -> Dict[str, float]:
        """Trainiert ML-Modelle für Marktklassifikation"""
        
        try:
            if self.logger:
                self.logger.info(f"Starte ML-Training für {symbol}", 'ml')
            
            # Trainingsdaten abrufen
            df = self.data_manager.get_historical_data(symbol, timeframe, train_periods)
            if df is None or len(df) < 500:
                if self.logger:
                    self.logger.error("Nicht genügend Trainingsdaten verfügbar", 'ml')
                return {'error': 'Insufficient data'}
            
            # Features und Labels vorbereiten
            X, y = self._prepare_training_data(df)
            
            if len(X) == 0 or len(y) == 0:
                if self.logger:
                    self.logger.error("Keine gültigen Trainingsdaten", 'ml')
                return {'error': 'No valid training data'}
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Feature-Skalierung
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Modell trainieren
            self.trend_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.trend_classifier.fit(X_train_scaled, y_train)
            
            # Evaluierung
            y_pred = self.trend_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Modelle speichern
            self._save_models()
            
            results = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            if self.logger:
                self.logger.log_ml_training(
                    model_name='TrendClassifier',
                    accuracy=accuracy,
                    loss=1-accuracy,
                    training_samples=len(X_train),
                    validation_accuracy=accuracy
                )
            
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler beim ML-Training")
            return {'error': str(e)}
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[List[List[float]], List[int]]:
        """Bereitet Trainingsdaten vor"""
        
        X = []
        y = []
        
        # Sliding window über historische Daten
        window_size = 200
        
        for i in range(window_size, len(df) - 10):  # 10 Perioden für Future-Labels
            try:
                # Daten-Slice für Indikatoren
                slice_df = df.iloc[i-window_size:i].copy()
                
                # Indikatoren berechnen
                indicators = self._calculate_indicators(slice_df)
                
                # Features extrahieren
                features = self._extract_features(slice_df, indicators)
                
                if len(features) == 0:
                    continue
                
                # Label basierend auf zukünftiger Preisentwicklung
                current_price = df.iloc[i]['close']
                future_price = df.iloc[i+10]['close']  # 10 Perioden voraus
                
                price_change = (future_price - current_price) / current_price
                
                # Label-Kategorien
                if price_change > 0.02:  # > 2% Anstieg
                    label = 2  # Uptrend
                elif price_change < -0.02:  # < -2% Rückgang
                    label = 0  # Downtrend
                else:  # -2% bis 2%
                    label = 1  # Sideways
                
                X.append(features)
                y.append(label)
                
            except Exception as e:
                continue
        
        return X, y
    
    def _save_models(self):
        """Speichert trainierte Modelle"""
        try:
            if self.trend_classifier is not None:
                joblib.dump(self.trend_classifier, self.model_dir / 'trend_classifier.pkl')
                joblib.dump(self.feature_scaler, self.model_dir / 'feature_scaler.pkl')
                
                # Modell-Metadaten speichern
                metadata = {
                    'model_type': 'RandomForestClassifier',
                    'features_count': len(self.feature_scaler.feature_names_in_) if hasattr(self.feature_scaler, 'feature_names_in_') else 0,
                    'trained_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
                
                with open(self.model_dir / 'model_metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                if self.logger:
                    self.logger.info("ML-Modelle gespeichert", 'ml')
                    
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler beim Speichern der Modelle")
    
    def _load_models(self):
        """Lädt gespeicherte Modelle"""
        try:
            trend_model_path = self.model_dir / 'trend_classifier.pkl'
            scaler_path = self.model_dir / 'feature_scaler.pkl'
            
            if trend_model_path.exists() and scaler_path.exists():
                self.trend_classifier = joblib.load(trend_model_path)
                self.feature_scaler = joblib.load(scaler_path)
                
                if self.logger:
                    self.logger.info("ML-Modelle geladen", 'ml')
            else:
                if self.logger:
                    self.logger.info("Keine gespeicherten Modelle gefunden", 'ml')
                    
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Fehler beim Laden der Modelle: {e}", 'ml')
    
    def _create_unknown_signal(self, symbol: str) -> MarketSignal:
        """Erstellt Fallback-Signal bei Fehlern"""
        return MarketSignal(
            condition=MarketCondition.UNKNOWN,
            confidence=0.0,
            strength=0.0,
            indicators={},
            timestamp=datetime.now(),
            recommendation="WAIT - Unzureichende Datengrundlage",
            support_levels=[],
            resistance_levels=[]
        )
    
    def _indicators_to_dict(self, indicators: TechnicalIndicators) -> Dict[str, float]:
        """Konvertiert TechnicalIndicators zu Dictionary"""
        return {
            'sma_20': indicators.sma_20,
            'sma_50': indicators.sma_50,
            'sma_200': indicators.sma_200,
            'ema_12': indicators.ema_12,
            'ema_26': indicators.ema_26,
            'macd': indicators.macd,
            'macd_signal': indicators.macd_signal,
            'macd_histogram': indicators.macd_histogram,
            'rsi': indicators.rsi,
            'stoch_k': indicators.stoch_k,
            'stoch_d': indicators.stoch_d,
            'williams_r': indicators.williams_r,
            'roc': indicators.roc,
            'bb_upper': indicators.bb_upper,
            'bb_middle': indicators.bb_middle,
            'bb_lower': indicators.bb_lower,
            'bb_width': indicators.bb_width,
            'atr': indicators.atr,
            'volume_sma': indicators.volume_sma,
            'volume_ratio': indicators.volume_ratio,
            'obv': indicators.obv,
            'pivot_point': indicators.pivot_point,
            'resistance_1': indicators.resistance_1,
            'resistance_2': indicators.resistance_2,
            'support_1': indicators.support_1,
            'support_2': indicators.support_2
        }
    
    # Fallback-Implementierungen für technische Indikatoren (wenn TA-Lib nicht verfügbar)
    def _sma(self, data: np.ndarray, period: int) -> float:
        """Simple Moving Average"""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0.0
        return np.mean(data[-period:])
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Exponential Moving Average"""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0.0
        
        alpha = 2 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _rsi(self, data: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return 50.0
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _stochastic(self, high: np.ndarray, low: np.ndarray, 
                   close: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        if len(close) < period:
            return 50.0, 50.0
        
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Vereinfachte %D Berechnung (normalerweise 3-Perioden SMA von %K)
        d_percent = k_percent
        
        return k_percent, d_percent
    
    def _williams_r(self, high: np.ndarray, low: np.ndarray, 
                   close: np.ndarray, period: int = 14) -> float:
        """Williams %R"""
        if len(close) < period:
            return -50.0
        
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - close[-1]) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    def _bollinger_bands(self, data: np.ndarray, period: int = 20, 
                        std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands"""
        if len(data) < period:
            current_price = data[-1] if len(data) > 0 else 0.0
            return current_price, current_price, current_price
        
        sma = np.mean(data[-period:])
        std = np.std(data[-period:], ddof=1)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    def _atr(self, high: np.ndarray, low: np.ndarray, 
            close: np.ndarray, period: int = 14) -> float:
        """Average True Range"""
        if len(close) < 2:
            return 0.0
        
        true_ranges = []
        
        for i in range(1, len(close)):
            high_low = high[i] - low[i]
            high_close_prev = abs(high[i] - close[i-1])
            low_close_prev = abs(low[i] - close[i-1])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.0
        
        return np.mean(true_ranges[-period:])
    
    def _obv(self, close: np.ndarray, volume: np.ndarray) -> float:
        """On-Balance Volume"""
        if len(close) < 2 or len(volume) != len(close):
            return 0.0
        
        obv = 0
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]
            # Bei gleichem Preis wird das Volumen ignoriert
        
        return obv
    
    def get_analysis_summary(self, lookback_days: int = 7) -> Dict[str, Any]:
        """Erstellt Zusammenfassung der Analysen der letzten Tage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Relevante Analysen filtern
            recent_analyses = [
                analysis for analysis in self.analysis_history
                if analysis['timestamp'] >= cutoff_date
            ]
            
            if not recent_analyses:
                return {'error': 'Keine Analysen im gewählten Zeitraum'}
            
            # Statistiken berechnen
            conditions = [analysis['signal'].condition.value for analysis in recent_analyses]
            confidences = [analysis['signal'].confidence for analysis in recent_analyses]
            strengths = [analysis['signal'].strength for analysis in recent_analyses]
            
            # Häufigkeiten der Marktbedingungen
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            # Durchschnittswerte
            avg_confidence = np.mean(confidences) if confidences else 0
            avg_strength = np.mean(strengths) if strengths else 0
            
            # Empfehlungsverteilung
            recommendations = [analysis['signal'].recommendation for analysis in recent_analyses]
            recommendation_counts = {}
            for rec in recommendations:
                rec_type = rec.split(' -')[0]  # Z.B. "BUY" aus "BUY - Aufwärtstrend"
                recommendation_counts[rec_type] = recommendation_counts.get(rec_type, 0) + 1
            
            # Symbole analysiert
            symbols = list(set([analysis['symbol'] for analysis in recent_analyses]))
            
            summary = {
                'period_days': lookback_days,
                'total_analyses': len(recent_analyses),
                'symbols_analyzed': symbols,
                'market_conditions': condition_counts,
                'recommendations': recommendation_counts,
                'average_confidence': round(avg_confidence, 3),
                'average_trend_strength': round(avg_strength, 3),
                'most_common_condition': max(condition_counts.items(), key=lambda x: x[1])[0] if condition_counts else 'unknown',
                'analysis_frequency': round(len(recent_analyses) / lookback_days, 1)
            }
            
            return summary
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler bei Analyse-Zusammenfassung")
            return {'error': str(e)}
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Aktualisiert Analyse-Schwellenwerte"""
        try:
            valid_keys = ['trend_strength', 'volatility_high', 'volatility_low', 
                         'rsi_oversold', 'rsi_overbought', 'confidence_min']
            
            for key, value in new_thresholds.items():
                if key in valid_keys and 0 <= value <= 1:
                    self.thresholds[key] = value
                    if self.logger:
                        self.logger.info(f"Schwellenwert {key} auf {value} aktualisiert", 'strategy')
                else:
                    if self.logger:
                        self.logger.warning(f"Ungültiger Schwellenwert: {key} = {value}", 'strategy')
                        
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler bei Schwellenwert-Update")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Gibt Performance-Metriken der ML-Modelle zurück"""
        try:
            if self.trend_classifier is None:
                return {'error': 'Keine Modelle trainiert'}
            
            # Modell-Metadaten laden
            metadata_path = self.model_dir / 'model_metadata.json'
            metadata = {}
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Feature-Importances (für RandomForest)
            feature_importance = {}
            if hasattr(self.trend_classifier, 'feature_importances_'):
                feature_names = [
                    'return_5d', 'return_10d', 'return_20d', 'rsi_norm', 'stoch_k_norm',
                    'williams_r_norm', 'macd_hist', 'bb_width', 'volume_ratio', 
                    'ma_ratio', 'volatility'
                ]
                
                for i, importance in enumerate(self.trend_classifier.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = round(importance, 4)
            
            performance = {
                'model_type': metadata.get('model_type', 'Unknown'),
                'trained_at': metadata.get('trained_at', 'Unknown'),
                'version': metadata.get('version', 'Unknown'),
                'features_count': metadata.get('features_count', 0),
                'feature_importance': feature_importance,
                'model_loaded': True
            }
            
            return performance
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler bei Model-Performance-Abfrage")
            return {'error': str(e)}
    
    def analyze_multiple_symbols(self, symbols: List[str], 
                               timeframe: str = '1h') -> Dict[str, MarketSignal]:
        """Analysiert mehrere Symbole gleichzeitig"""
        results = {}
        
        for symbol in symbols:
            try:
                signal = self.analyze_market(symbol, timeframe)
                results[symbol] = signal
                
                if self.logger:
                    self.logger.info(f"Analyse abgeschlossen für {symbol}: {signal.condition.value}", 'strategy')
                    
            except Exception as e:
                if self.logger:
                    self.logger.log_error(e, f"Fehler bei Analyse von {symbol}")
                results[symbol] = self._create_unknown_signal(symbol)
        
        return results
    
    def get_market_overview(self, symbols: List[str]) -> Dict[str, Any]:
        """Erstellt Marktübersicht für mehrere Symbole"""
        try:
            signals = self.analyze_multiple_symbols(symbols)
            
            # Zusammenfassung erstellen
            conditions = [signal.condition.value for signal in signals.values()]
            confidences = [signal.confidence for signal in signals.values()]
            
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            # Empfehlungen zusammenfassen
            recommendations = {}
            for symbol, signal in signals.items():
                rec_type = signal.recommendation.split(' -')[0]
                if rec_type not in recommendations:
                    recommendations[rec_type] = []
                recommendations[rec_type].append(symbol)
            
            overview = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(symbols),
                'market_conditions': condition_counts,
                'average_confidence': round(np.mean(confidences), 3) if confidences else 0,
                'recommendations': recommendations,
                'individual_signals': {symbol: {
                    'condition': signal.condition.value,
                    'confidence': signal.confidence,
                    'recommendation': signal.recommendation,
                    'strength': signal.strength
                } for symbol, signal in signals.items()}
            }
            
            return overview
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler bei Marktübersicht")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Bereinigt alte Analysedaten"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            original_count = len(self.analysis_history)
            self.analysis_history = [
                analysis for analysis in self.analysis_history
                if analysis['timestamp'] >= cutoff_date
            ]
            
            cleaned_count = original_count - len(self.analysis_history)
            
            if self.logger and cleaned_count > 0:
                self.logger.info(f"{cleaned_count} alte Analysedaten bereinigt", 'system')
                
            # Cache bereinigen
            self.indicator_cache.clear()
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler bei Datenbereinigung")
    
    def __del__(self):
        """Cleanup beim Zerstören des Objekts"""
        try:
            # Letzte Modelle speichern
            if hasattr(self, 'trend_classifier') and self.trend_classifier is not None:
                self._save_models()
        except:
            pass