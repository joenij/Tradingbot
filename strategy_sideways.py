"""
Trading Bot - Sideways Grid Trading Strategy
Intelligentes Grid Trading System für Seitwärtsmärkte mit dynamischer Grid-Anpassung
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
import time
from enum import Enum
import math

class GridType(Enum):
    ARITHMETIC = "arithmetic"  # Gleichmäßige Abstände
    GEOMETRIC = "geometric"    # Prozentuale Abstände
    ADAPTIVE = "adaptive"      # Volatilitätsbasiert
    FIBONACCI = "fibonacci"    # Fibonacci-Levels

class GridDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class GridLevel:
    """Einzelnes Grid-Level"""
    price: float
    level_id: str
    is_buy: bool
    is_sell: bool
    order_id: Optional[str] = None
    filled: bool = False
    fill_time: Optional[datetime] = None
    fill_price: Optional[float] = None
    amount: float = 0.0
    profit_target: Optional[float] = None
    
class GridOrder:
    """Grid-Order Management"""
    def __init__(self, level: GridLevel, symbol: str, exchange: str):
        self.level = level
        self.symbol = symbol
        self.exchange = exchange
        self.created_at = datetime.now()
        self.status = "pending"  # pending, filled, cancelled, error
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level_id': self.level.level_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'price': self.level.price,
            'amount': self.level.amount,
            'is_buy': self.level.is_buy,
            'is_sell': self.level.is_sell,
            'order_id': self.level.order_id,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'filled': self.level.filled,
            'fill_time': self.level.fill_time.isoformat() if self.level.fill_time else None,
            'fill_price': self.level.fill_price
        }

class SidewaysGridStrategy:
    """
    Intelligente Grid Trading Strategie für Seitwärtsmärkte
    
    Features:
    - Dynamische Grid-Anpassung basierend auf Volatilität
    - Verschiedene Grid-Typen (arithmetisch, geometrisch, adaptiv)
    - Automatische Take-Profit Levels
    - Position-Size Management
    - Stop-Loss Integration
    - Grid-Rebalancing bei Marktänderungen
    """
    
    def __init__(self, config_manager=None, logger=None):
        self.config_manager = config_manager
        self.logger = logger
        
        # Strategy Konfiguration
        self.strategy_name = "sideways_grid"
        self.version = "1.0.0"
        
        # Grid Konfiguration
        self.grid_config = self._load_grid_config()
        
        # Aktive Grids pro Symbol
        self.active_grids: Dict[str, Dict[str, Any]] = {}
        
        # Grid Orders Management
        self.grid_orders: Dict[str, List[GridOrder]] = {}
        
        # Performance Tracking
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0.0,
            'grid_adjustments': 0,
            'active_positions': 0
        }
        
        # Thread-Safety
        self.lock = threading.Lock()
        
        # Market Data Cache
        self.market_data_cache = {}
        self.volatility_cache = {}
        
        if self.logger:
            self.logger.info(f"Sideways Grid Strategy initialisiert - Version: {self.version}", 'strategy')
    
    def _load_grid_config(self) -> Dict[str, Any]:
        """Lädt Grid-Konfiguration"""
        default_config = {
            # Grid Grundeinstellungen
            'grid_type': GridType.ADAPTIVE.value,
            'grid_levels': 10,  # Anzahl Grid-Levels pro Seite
            'grid_spacing_percent': 0.5,  # Basis-Abstand zwischen Levels in %
            'max_grid_range_percent': 10.0,  # Maximaler Grid-Bereich in %
            
            # Positionsgrößen
            'base_position_size': 0.1,  # Basis-Positionsgröße
            'max_total_position': 1.0,  # Maximale Gesamtposition
            'position_scaling': 'linear',  # linear, exponential, fibonacci
            
            # Risk Management  
            'stop_loss_percent': 15.0,  # Stop-Loss in %
            'take_profit_percent': 2.0,  # Take-Profit pro Trade in %
            'max_drawdown_percent': 10.0,
            
            # Grid-Anpassungen
            'volatility_lookback': 24,  # Stunden für Volatilitätsberechnung
            'grid_rebalance_threshold': 0.3,  # Schwelle für Grid-Neuausrichtung
            'min_rebalance_interval': 3600,  # Min. Sekunden zwischen Rebalancing
            
            # Marktbedingungen
            'sideways_detection_period': 48,  # Stunden für Seitwärtstrend-Erkennung
            'trend_strength_threshold': 0.02,  # Schwelle für Trenddetektion
            'volatility_multiplier': 1.5,  # Multiplikator für adaptive Grids
            
            # Execution
            'order_timeout': 300,  # Sekunden bis Order-Timeout
            'partial_fill_handling': True,
            'slippage_tolerance': 0.1,  # % Slippage-Toleranz
            
            # Advanced Features
            'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
            'support_resistance_integration': True,
            'volume_profile_weighting': True
        }
        
        if self.config_manager:
            return self.config_manager.get('strategy_sideways', default_config)
        return default_config
    
    def analyze_market_condition(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analysiert Marktbedingungen für Grid Trading
        
        Returns:
            Dict mit Marktanalyse und Grid-Empfehlungen
        """
        try:
            if data.empty or len(data) < self.grid_config['sideways_detection_period']:
                return {'suitable': False, 'reason': 'Insufficient data'}
            
            # Preis-Daten extrahieren
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values if 'volume' in data.columns else None
            
            current_price = closes[-1]
            
            # 1. Trend-Analyse
            trend_analysis = self._analyze_trend(closes)
            
            # 2. Volatilitäts-Analyse
            volatility_analysis = self._analyze_volatility(closes, highs, lows)
            
            # 3. Support/Resistance Levels
            sr_levels = self._identify_support_resistance(data)
            
            # 4. Seitwärtsmarkt-Detektion
            sideways_score = self._calculate_sideways_score(closes, trend_analysis, volatility_analysis)
            
            # 5. Grid-Parameter berechnen
            grid_params = self._calculate_optimal_grid_params(
                current_price, volatility_analysis, sr_levels, sideways_score
            )
            
            # 6. Volume Profile (falls verfügbar)
            volume_profile = self._analyze_volume_profile(data) if volumes is not None else {}
            
            analysis_result = {
                'suitable': sideways_score > 0.6,
                'confidence': sideways_score,
                'current_price': current_price,
                'trend_analysis': trend_analysis,
                'volatility_analysis': volatility_analysis,
                'support_resistance': sr_levels,
                'grid_parameters': grid_params,
                'volume_profile': volume_profile,
                'recommendation': self._generate_grid_recommendation(sideways_score, grid_params),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache für spätere Verwendung
            self.market_data_cache[symbol] = analysis_result
            
            if self.logger:
                self.logger.log_market_analysis(
                    symbol=symbol,
                    timeframe='1h',
                    market_condition='sideways' if analysis_result['suitable'] else 'trending',
                    confidence=sideways_score,
                    indicators={
                        'volatility': volatility_analysis['current_volatility'],
                        'trend_strength': trend_analysis['strength'],
                        'support_level': sr_levels.get('nearest_support', current_price),
                        'resistance_level': sr_levels.get('nearest_resistance', current_price)
                    }
                )
            
            return analysis_result
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler bei Marktanalyse für {symbol}")
            return {'suitable': False, 'error': str(e)}
    
    def _analyze_trend(self, closes: np.ndarray) -> Dict[str, Any]:
        """Analysiert Trend-Stärke und -Richtung"""
        
        # Linear Regression für Trend
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        
        # Trend-Stärke basierend auf R²
        y_pred = slope * x + intercept
        ss_res = np.sum((closes - y_pred) ** 2)
        ss_tot = np.sum((closes - np.mean(closes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Relative Strength Index (RSI)
        rsi = self._calculate_rsi(closes)
        
        # Moving Average Convergence
        ema_short = self._ema(closes, 12)
        ema_long = self._ema(closes, 26)
        macd_line = ema_short - ema_long
        
        return {
            'slope': slope,
            'strength': r_squared,
            'direction': 'up' if slope > 0 else 'down' if slope < 0 else 'sideways',
            'rsi': rsi[-1] if len(rsi) > 0 else 50,
            'macd': macd_line[-1] if len(macd_line) > 0 else 0,
            'trend_score': abs(slope) / closes[-1] * 100  # Trend in %
        }
    
    def _analyze_volatility(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict[str, Any]:
        """Analysiert Volatilität für Grid-Spacing"""
        
        # True Range Volatility
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        
        # Price Returns Volatility
        returns = np.diff(closes) / closes[:-1]
        volatility_std = np.std(returns[-self.grid_config['volatility_lookback']:])
        
        # Bollinger Band Width
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
        std_20 = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes)
        bb_width = (std_20 * 2) / ma_20 * 100 if ma_20 != 0 else 0
        
        # Volatility Regime
        volatility_percentile = self._calculate_volatility_percentile(volatility_std, closes)
        
        current_price = closes[-1]
        
        return {
            'atr': atr,
            'atr_percent': (atr / current_price * 100) if current_price != 0 else 0,
            'current_volatility': volatility_std,
            'bb_width': bb_width,
            'volatility_percentile': volatility_percentile,
            'regime': self._classify_volatility_regime(volatility_percentile),
            'optimal_grid_spacing': self._calculate_optimal_spacing(atr, current_price, volatility_std)
        }
    
    def _identify_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Identifiziert Support/Resistance Levels"""
        
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        current_price = closes[-1]
        
        # Lokale Maxima und Minima finden
        from scipy.signal import find_peaks
        
        # Resistance Levels (lokale Maxima)
        resistance_peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
        resistance_levels = highs[resistance_peaks] if len(resistance_peaks) > 0 else []
        
        # Support Levels (lokale Minima)  
        support_peaks, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
        support_levels = lows[support_peaks] if len(support_peaks) > 0 else []
        
        # Nächste Levels zum aktuellen Preis
        nearest_resistance = None
        nearest_support = None
        
        if len(resistance_levels) > 0:
            resistance_above = resistance_levels[resistance_levels > current_price]
            nearest_resistance = np.min(resistance_above) if len(resistance_above) > 0 else np.max(resistance_levels)
        
        if len(support_levels) > 0:
            support_below = support_levels[support_levels < current_price]
            nearest_support = np.max(support_below) if len(support_below) > 0 else np.min(support_levels)
        
        # Trading Range
        if nearest_resistance and nearest_support:
            trading_range = (nearest_resistance - nearest_support) / current_price * 100
        else:
            trading_range = np.std(closes[-50:]) / current_price * 100 * 4  # Fallback
        
        return {
            'resistance_levels': resistance_levels.tolist() if len(resistance_levels) > 0 else [],
            'support_levels': support_levels.tolist() if len(support_levels) > 0 else [],
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'trading_range_percent': trading_range,
            'price_position': self._calculate_price_position(current_price, nearest_support, nearest_resistance)
        }
    
    def _calculate_sideways_score(self, closes: np.ndarray, trend_analysis: Dict, volatility_analysis: Dict) -> float:
        """Berechnet Score für Seitwärtsmarkt-Eignung (0-1)"""
        
        # Trend-Component (niedriger Trend = besser für Sideways)
        trend_score = max(0, 1 - abs(trend_analysis['trend_score']) / 2.0)
        
        # Volatilitäts-Component (moderate Volatilität ideal)
        vol_regime = volatility_analysis['volatility_percentile']
        if 0.3 <= vol_regime <= 0.7:
            volatility_score = 1.0
        elif vol_regime < 0.3:
            volatility_score = vol_regime / 0.3 * 0.7  # Zu niedrige Volatilität
        else:
            volatility_score = (1 - vol_regime) / 0.3 * 0.7  # Zu hohe Volatilität
        
        # Range-Bound Behaviour
        range_score = min(1.0, volatility_analysis['bb_width'] / 5.0)  # Normalisiert auf BB-Width
        
        # RSI Mean Reversion (RSI um 50 ideal)
        rsi = trend_analysis['rsi']
        rsi_score = 1 - abs(rsi - 50) / 50
        
        # Kombinierte Bewertung
        sideways_score = (
            trend_score * 0.4 +      # 40% Trend
            volatility_score * 0.3 +  # 30% Volatilität
            range_score * 0.2 +       # 20% Range
            rsi_score * 0.1           # 10% RSI
        )
        
        return min(1.0, max(0.0, sideways_score))
    
    def _calculate_optimal_grid_params(self, price: float, volatility: Dict, sr_levels: Dict, sideways_score: float) -> Dict[str, Any]:
        """Berechnet optimale Grid-Parameter"""
        
        # Basis Grid-Spacing basierend auf Volatilität
        base_spacing = volatility['optimal_grid_spacing']
        
        # Anpassung basierend auf Support/Resistance
        if sr_levels['trading_range_percent'] > 0:
            range_spacing = sr_levels['trading_range_percent'] / (self.grid_config['grid_levels'] * 2)
            optimal_spacing = min(base_spacing, range_spacing)
        else:
            optimal_spacing = base_spacing
        
        # Grid-Typ basierend auf Marktbedingungen
        if volatility['regime'] == 'high':
            grid_type = GridType.GEOMETRIC
        elif sr_levels['resistance_levels'] and sr_levels['support_levels']:
            grid_type = GridType.ADAPTIVE
        else:
            grid_type = GridType.ARITHMETIC
        
        # Grid-Range berechnen
        max_range = min(
            self.grid_config['max_grid_range_percent'],
            sr_levels['trading_range_percent'] * 0.8 if sr_levels['trading_range_percent'] > 0 else 10
        )
        
        # Position Sizing
        position_multiplier = min(2.0, sideways_score * 1.5)  # Mehr Position bei besseren Bedingungen
        
        return {
            'grid_type': grid_type.value,
            'spacing_percent': optimal_spacing,
            'grid_levels': self.grid_config['grid_levels'],
            'max_range_percent': max_range,
            'position_multiplier': position_multiplier,
            'center_price': price,
            'upper_bound': price * (1 + max_range / 200),
            'lower_bound': price * (1 - max_range / 200),
            'recommended_capital': self._calculate_required_capital(price, optimal_spacing, max_range)
        }
    
    def create_grid(self, symbol: str, exchange: str, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erstellt ein neues Grid für ein Symbol
        """
        try:
            if not market_analysis['suitable']:
                return {'success': False, 'reason': 'Market not suitable for grid trading'}
            
            grid_params = market_analysis['grid_parameters']
            current_price = market_analysis['current_price']
            
            # Grid ID generieren
            grid_id = f"{symbol}_{exchange}_{int(datetime.now().timestamp())}"
            
            # Grid Levels berechnen
            grid_levels = self._calculate_grid_levels(
                center_price=current_price,
                grid_type=GridType(grid_params['grid_type']),
                spacing_percent=grid_params['spacing_percent'],
                num_levels=grid_params['grid_levels'],
                max_range=grid_params['max_range_percent']
            )
            
            # Position Sizes berechnen
            position_sizes = self._calculate_position_sizes(grid_levels, grid_params['position_multiplier'])
            
            # Grid-Struktur erstellen
            grid_structure = {
                'grid_id': grid_id,
                'symbol': symbol,
                'exchange': exchange,
                'created_at': datetime.now(),
                'grid_type': grid_params['grid_type'],
                'center_price': current_price,
                'spacing_percent': grid_params['spacing_percent'],
                'levels': [],
                'orders': [],
                'status': 'active',
                'total_invested': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_trades': 0,
                'last_rebalance': datetime.now(),
                'market_analysis': market_analysis
            }
            
            # Grid Levels mit Position Sizes kombinieren
            for i, (level_price, position_size) in enumerate(zip(grid_levels, position_sizes)):
                level = GridLevel(
                    price=level_price,
                    level_id=f"{grid_id}_L{i}",
                    is_buy=level_price < current_price,
                    is_sell=level_price > current_price,
                    amount=position_size
                )
                grid_structure['levels'].append(level)
            
            # Grid registrieren
            with self.lock:
                self.active_grids[grid_id] = grid_structure
                self.grid_orders[grid_id] = []
            
            if self.logger:
                self.logger.info(
                    f"Grid erstellt: {grid_id} | Symbol: {symbol} | Levels: {len(grid_levels)} | "
                    f"Spacing: {grid_params['spacing_percent']:.2f}% | Type: {grid_params['grid_type']}",
                    'strategy'
                )
            
            return {
                'success': True,
                'grid_id': grid_id,
                'grid_structure': grid_structure,
                'recommended_orders': self._generate_initial_orders(grid_structure)
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler beim Erstellen des Grids für {symbol}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_grid_levels(self, center_price: float, grid_type: GridType, 
                              spacing_percent: float, num_levels: int, max_range: float) -> List[float]:
        """Berechnet Grid-Level basierend auf Grid-Typ"""
        
        levels = []
        
        if grid_type == GridType.ARITHMETIC:
            # Gleichmäßige Abstände
            spacing = center_price * spacing_percent / 100
            
            # Levels oberhalb des aktuellen Preises
            for i in range(1, num_levels + 1):
                level = center_price + (spacing * i)
                if (level - center_price) / center_price * 100 <= max_range / 2:
                    levels.append(level)
            
            # Levels unterhalb des aktuellen Preises
            for i in range(1, num_levels + 1):
                level = center_price - (spacing * i)
                if (center_price - level) / center_price * 100 <= max_range / 2:
                    levels.append(level)
        
        elif grid_type == GridType.GEOMETRIC:
            # Prozentuale Abstände
            multiplier = 1 + spacing_percent / 100
            
            # Levels oberhalb
            for i in range(1, num_levels + 1):
                level = center_price * (multiplier ** i)
                if (level - center_price) / center_price * 100 <= max_range / 2:
                    levels.append(level)
            
            # Levels unterhalb
            for i in range(1, num_levels + 1):
                level = center_price / (multiplier ** i)
                if (center_price - level) / center_price * 100 <= max_range / 2:
                    levels.append(level)
        
        elif grid_type == GridType.FIBONACCI:
            # Fibonacci-basierte Levels
            fib_levels = self.grid_config['fibonacci_levels']
            range_size = center_price * max_range / 200  # Halber Range
            
            for fib_ratio in fib_levels:
                # Oberhalb
                level_up = center_price + (range_size * fib_ratio)
                levels.append(level_up)
                
                # Unterhalb
                level_down = center_price - (range_size * fib_ratio)
                levels.append(level_down)
        
        elif grid_type == GridType.ADAPTIVE:
            # Volatilitäts-adaptive Levels
            # Engere Levels bei niedriger Volatilität, weitere bei hoher
            volatility_multiplier = self.grid_config['volatility_multiplier']
            
            # Basis-Spacing anpassen
            adaptive_spacing = spacing_percent * volatility_multiplier
            spacing = center_price * adaptive_spacing / 100
            
            # Progressive Abstände (näher zum Zentrum enger)
            for i in range(1, num_levels + 1):
                progression_factor = math.sqrt(i)  # Progressiver Faktor
                
                level_up = center_price + (spacing * progression_factor)
                level_down = center_price - (spacing * progression_factor)
                
                if (level_up - center_price) / center_price * 100 <= max_range / 2:
                    levels.append(level_up)
                
                if (center_price - level_down) / center_price * 100 <= max_range / 2:
                    levels.append(level_down)
        
        # Sortieren und deduplizieren
        levels = sorted(list(set(levels)))
        
        # Aktuellen Preis hinzufügen falls nicht vorhanden
        if center_price not in levels:
            levels.append(center_price)
            levels.sort()
        
        return levels
    
    def _calculate_position_sizes(self, levels: List[float], multiplier: float) -> List[float]:
        """Berechnet Position Sizes für Grid Levels"""
        
        base_size = self.grid_config['base_position_size'] * multiplier
        scaling_type = self.grid_config['position_scaling']
        
        sizes = []
        
        if scaling_type == 'linear':
            # Gleichmäßige Größen
            sizes = [base_size] * len(levels)
        
        elif scaling_type == 'exponential':
            # Größere Positionen weiter vom Zentrum
            center_index = len(levels) // 2
            for i, level in enumerate(levels):
                distance_from_center = abs(i - center_index)
                size = base_size * (1.2 ** distance_from_center)
                sizes.append(min(size, base_size * 3))  # Cap bei 3x
        
        elif scaling_type == 'fibonacci':
            # Fibonacci-Sequenz für Position Sizes
            fib_sequence = [1, 1]
            while len(fib_sequence) < len(levels):
                fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
            
            # Normalisieren
            max_fib = max(fib_sequence[:len(levels)])
            for fib_val in fib_sequence[:len(levels)]:
                size = base_size * (fib_val / max_fib) * 2
                sizes.append(size)
        
        else:
            # Default: linear
            sizes = [base_size] * len(levels)
        
        return sizes
    
    def _generate_initial_orders(self, grid_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generiert initiale Orders für das Grid"""
        
        orders = []
        current_price = grid_structure['center_price']
        
        for level in grid_structure['levels']:
            if level.price == current_price:
                continue  # Skip center price
            
            order_type = 'limit_buy' if level.is_buy else 'limit_sell'
            
            order = {
                'symbol': grid_structure['symbol'],
                'exchange': grid_structure['exchange'],
                'type': order_type,
                'amount': level.amount,
                'price': level.price,
                'level_id': level.level_id,
                'grid_id': grid_structure['grid_id'],
                'time_in_force': 'GTC',  # Good Till Cancelled
                'reduce_only': False
            }
            
            orders.append(order)
        
        return orders
    
    def update_grid(self, grid_id: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Aktualisiert ein existierendes Grid basierend auf Marktbedingungen
        """
        try:
            if grid_id not in self.active_grids:
                return {'success': False, 'reason': 'Grid not found'}
            
            grid = self.active_grids[grid_id]
            current_price = market_data['close'].iloc[-1]
            
            # Prüfen ob Rebalancing notwendig
            rebalance_needed = self._check_rebalance_needed(grid, current_price, market_data)
            
            if rebalance_needed['needed']:
                return self._rebalance_grid(grid_id, current_price, market_data, rebalance_needed['reason'])
            
            # Standard Update - nur Preis-Updates
            grid['current_price'] = current_price
            grid['last_update'] = datetime.now()
            
            # PnL berechnen
            pnl_data = self._calculate_grid_pnl(grid, current_price)
            grid.update(pnl_data)
            
            return {
                'success': True,
                'grid_id': grid_id,
                'rebalanced': False,
                'current_price': current_price,
            	'pnl': pnl_data,
       	    	'active_orders': len([o for o in grid['orders'] if o.status == 'pending']),
            	'filled_orders': len([o for o in grid['orders'] if o.status == 'filled'])
        }
        
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler beim Grid-Update für {grid_id}")
            return {'success': False, 'error': str(e)}
    
    def _check_rebalance_needed(self, grid: Dict[str, Any], current_price: float, 
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Prüft ob Grid-Rebalancing notwendig ist"""
        
        # Zeit seit letztem Rebalancing
        time_since_rebalance = (datetime.now() - grid['last_rebalance']).total_seconds()
        min_interval = self.grid_config['min_rebalance_interval']
        
        if time_since_rebalance < min_interval:
            return {'needed': False, 'reason': 'Too soon since last rebalance'}
        
        # Preis-Drift vom Grid-Zentrum
        center_price = grid['center_price']
        price_drift = abs(current_price - center_price) / center_price
        drift_threshold = self.grid_config['grid_rebalance_threshold']
        
        if price_drift > drift_threshold:
            return {'needed': True, 'reason': f'Price drift {price_drift:.3f} exceeds threshold {drift_threshold}'}
        
        # Volatilitäts-Änderung
        current_volatility = self._calculate_current_volatility(market_data)
        grid_volatility = grid.get('initial_volatility', current_volatility)
        
        vol_change = abs(current_volatility - grid_volatility) / grid_volatility
        if vol_change > 0.5:  # 50% Volatilitäts-Änderung
            return {'needed': True, 'reason': f'Volatility change {vol_change:.3f} exceeds 50%'}
        
        # Zu viele gefüllte Orders an einer Seite
        filled_levels = [level for level in grid['levels'] if level.filled]
        if len(filled_levels) > len(grid['levels']) * 0.7:  # 70% der Levels gefüllt
            return {'needed': True, 'reason': 'Too many levels filled, grid exhausted'}
        
        # Market Regime Change
        new_analysis = self.analyze_market_condition(grid['symbol'], market_data)
        if not new_analysis['suitable']:
            return {'needed': True, 'reason': 'Market no longer suitable for grid trading'}
        
        return {'needed': False, 'reason': 'No rebalancing needed'}
    
    def _rebalance_grid(self, grid_id: str, current_price: float, 
                       market_data: pd.DataFrame, reason: str) -> Dict[str, Any]:
        """Führt Grid-Rebalancing durch"""
        
        try:
            grid = self.active_grids[grid_id]
            
            if self.logger:
                self.logger.info(f"Grid Rebalancing gestartet: {grid_id} | Grund: {reason}", 'strategy')
            
            # Neue Marktanalyse
            new_analysis = self.analyze_market_condition(grid['symbol'], market_data)
            
            if not new_analysis['suitable']:
                # Grid deaktivieren wenn Markt nicht mehr geeignet
                return self._deactivate_grid(grid_id, "Market no longer suitable")
            
            # Offene Orders stornieren
            cancelled_orders = self._cancel_grid_orders(grid_id)
            
            # Neue Grid-Parameter berechnen
            new_params = new_analysis['grid_parameters']
            
            # Neue Grid-Levels berechnen
            new_levels = self._calculate_grid_levels(
                center_price=current_price,
                grid_type=GridType(new_params['grid_type']),
                spacing_percent=new_params['spacing_percent'],
                num_levels=new_params['grid_levels'],
                max_range=new_params['max_range_percent']
            )
            
            # Position Sizes neu berechnen
            new_position_sizes = self._calculate_position_sizes(new_levels, new_params['position_multiplier'])
            
            # Grid aktualisieren
            with self.lock:
                # Alte Levels durch neue ersetzen
                grid['levels'] = []
                for i, (level_price, position_size) in enumerate(zip(new_levels, new_position_sizes)):
                    level = GridLevel(
                        price=level_price,
                        level_id=f"{grid_id}_R{int(datetime.now().timestamp())}_L{i}",
                        is_buy=level_price < current_price,
                        is_sell=level_price > current_price,
                        amount=position_size
                    )
                    grid['levels'].append(level)
                
                # Grid-Eigenschaften aktualisieren
                grid['center_price'] = current_price
                grid['spacing_percent'] = new_params['spacing_percent']
                grid['grid_type'] = new_params['grid_type']
                grid['last_rebalance'] = datetime.now()
                grid['market_analysis'] = new_analysis
                
                # Performance-Tracking
                self.performance_metrics['grid_adjustments'] += 1
            
            # Neue Orders generieren
            new_orders = self._generate_initial_orders(grid)
            
            if self.logger:
                self.logger.info(
                    f"Grid rebalanciert: {grid_id} | Neue Levels: {len(new_levels)} | "
                    f"Neue Spacing: {new_params['spacing_percent']:.2f}% | "
                    f"Stornierte Orders: {len(cancelled_orders)}",
                    'strategy'
                )
            
            return {
                'success': True,
                'grid_id': grid_id,
                'rebalanced': True,
                'reason': reason,
                'new_levels': len(new_levels),
                'cancelled_orders': len(cancelled_orders),
                'new_orders': new_orders,
                'new_parameters': new_params
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler beim Grid-Rebalancing für {grid_id}")
            return {'success': False, 'error': str(e)}
    
    def process_order_fill(self, grid_id: str, order_id: str, fill_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verarbeitet gefüllte Orders und aktualisiert Grid
        """
        try:
            if grid_id not in self.active_grids:
                return {'success': False, 'reason': 'Grid not found'}
            
            grid = self.active_grids[grid_id]
            
            # Entsprechenden Level finden
            filled_level = None
            for level in grid['levels']:
                if level.order_id == order_id:
                    filled_level = level
                    break
            
            if not filled_level:
                return {'success': False, 'reason': 'Order level not found'}
            
            # Level als gefüllt markieren
            with self.lock:
                filled_level.filled = True
                filled_level.fill_time = datetime.now()
                filled_level.fill_price = fill_data.get('price', filled_level.price)
                
                # Order Status aktualisieren
                for order in self.grid_orders[grid_id]:
                    if order.level.order_id == order_id:
                        order.status = 'filled'
                        break
                
                # Grid-Statistiken aktualisieren
                grid['total_trades'] += 1
                self.performance_metrics['total_trades'] += 1
            
            # Take-Profit Order erstellen (falls konfiguriert)
            take_profit_order = None
            if self.grid_config['take_profit_percent'] > 0:
                take_profit_order = self._create_take_profit_order(filled_level, fill_data)
            
            # Neue Grid-Order für das gleiche Level erstellen (Grid-Replenishment)
            replacement_order = self._create_replacement_order(filled_level, grid)
            
            # Profit/Loss berechnen
            pnl_data = self._calculate_trade_pnl(filled_level, fill_data)
            
            with self.lock:
                grid['realized_pnl'] += pnl_data['realized_pnl']
                if pnl_data['realized_pnl'] > 0:
                    self.performance_metrics['profitable_trades'] += 1
                self.performance_metrics['total_profit'] += pnl_data['realized_pnl']
            
            if self.logger:
                self.logger.log_trade(
                    symbol=grid['symbol'],
                    exchange=grid['exchange'],
                    side='buy' if filled_level.is_buy else 'sell',
                    amount=fill_data.get('amount', filled_level.amount),
                    price=filled_level.fill_price,
                    pnl=pnl_data['realized_pnl'],
                    strategy='sideways_grid',
                    metadata={
                        'grid_id': grid_id,
                        'level_id': filled_level.level_id,
                        'grid_level_price': filled_level.price
                    }
                )
            
            return {
                'success': True,
                'grid_id': grid_id,
                'filled_level': filled_level.level_id,
                'pnl': pnl_data,
                'take_profit_order': take_profit_order,
                'replacement_order': replacement_order,
                'grid_trades': grid['total_trades']
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler bei Order-Fill Verarbeitung für {grid_id}")
            return {'success': False, 'error': str(e)}
    
    def _create_take_profit_order(self, filled_level: GridLevel, fill_data: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Take-Profit Order für gefüllten Level"""
        
        fill_price = fill_data.get('price', filled_level.price)
        tp_percent = self.grid_config['take_profit_percent'] / 100
        
        if filled_level.is_buy:
            # Long Position -> Sell Take-Profit
            tp_price = fill_price * (1 + tp_percent)
            side = 'sell'
        else:
            # Short Position -> Buy Take-Profit
            tp_price = fill_price * (1 - tp_percent)
            side = 'buy'
        
        return {
            'type': f'limit_{side}',
            'price': tp_price,
            'amount': fill_data.get('amount', filled_level.amount),
            'level_id': filled_level.level_id,
            'order_type': 'take_profit',
            'parent_fill_price': fill_price,
            'reduce_only': True
        }
    
    def _create_replacement_order(self, filled_level: GridLevel, grid: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Ersatz-Order für gefüllten Grid-Level"""
        
        # Neue Order-ID generieren
        new_order_id = f"{filled_level.level_id}_R{int(datetime.now().timestamp())}"
        
        # Gleicher Level, gleiche Parameter
        return {
            'type': 'limit_buy' if filled_level.is_buy else 'limit_sell',
            'price': filled_level.price,
            'amount': filled_level.amount,
            'level_id': new_order_id,
            'grid_id': grid['grid_id'],
            'time_in_force': 'GTC',
            'reduce_only': False,
            'order_type': 'grid_replacement'
        }
    
    def _calculate_trade_pnl(self, level: GridLevel, fill_data: Dict[str, Any]) -> Dict[str, Any]:
        """Berechnet PnL für einzelnen Trade"""
        
        fill_price = fill_data.get('price', level.price)
        fill_amount = fill_data.get('amount', level.amount)
        
        # Vereinfachte PnL-Berechnung (ohne Gegenpositionen)
        # In der Realität würde hier eine komplexere Berechnung mit Positionen erfolgen
        
        if level.is_buy:
            # Kauforder - potentieller Gewinn bei späterem Verkauf
            potential_profit = 0  # Wird erst bei Verkauf realisiert
            unrealized_pnl = 0
        else:
            # Verkaufsorder - Gewinn falls vorher gekauft
            potential_profit = 0  # Vereinfacht
            unrealized_pnl = 0
        
        return {
            'realized_pnl': 0,  # Wird erst bei Schließung der Position berechnet
            'unrealized_pnl': unrealized_pnl,
            'fill_price': fill_price,
            'fill_amount': fill_amount,
            'commission': fill_amount * fill_price * 0.001  # 0.1% geschätzte Gebühr
        }
    
    def _calculate_grid_pnl(self, grid: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Berechnet Gesamt-PnL für ein Grid"""
        
        total_unrealized = 0
        total_invested = 0
        open_positions = 0
        
        for level in grid['levels']:
            if level.filled and level.fill_price:
                position_value = level.amount * level.fill_price
                current_value = level.amount * current_price
                
                if level.is_buy:
                    # Long Position
                    unrealized_pnl = current_value - position_value
                    total_invested += position_value
                    open_positions += 1
                else:
                    # Short Position (vereinfacht)
                    unrealized_pnl = position_value - current_value
                    open_positions += 1
                
                total_unrealized += unrealized_pnl
        
        return {
            'unrealized_pnl': total_unrealized,
            'total_invested': total_invested,
            'open_positions': open_positions,
            'current_value': total_invested + total_unrealized if total_invested > 0 else 0,
            'roi_percent': (total_unrealized / total_invested * 100) if total_invested > 0 else 0
        }
    
    def _cancel_grid_orders(self, grid_id: str) -> List[str]:
        """Storniert alle offenen Orders für ein Grid"""
        
        cancelled_orders = []
        
        if grid_id in self.grid_orders:
            for order in self.grid_orders[grid_id]:
                if order.status == 'pending':
                    order.status = 'cancelled'
                    cancelled_orders.append(order.level.order_id)
        
        return cancelled_orders
    
    def _deactivate_grid(self, grid_id: str, reason: str) -> Dict[str, Any]:
        """Deaktiviert ein Grid"""
        
        try:
            if grid_id not in self.active_grids:
                return {'success': False, 'reason': 'Grid not found'}
            
            # Orders stornieren
            cancelled_orders = self._cancel_grid_orders(grid_id)
            
            # Grid deaktivieren
            with self.lock:
                self.active_grids[grid_id]['status'] = 'deactivated'
                self.active_grids[grid_id]['deactivated_at'] = datetime.now()
                self.active_grids[grid_id]['deactivation_reason'] = reason
            
            if self.logger:
                self.logger.info(f"Grid deaktiviert: {grid_id} | Grund: {reason}", 'strategy')
            
            return {
                'success': True,
                'grid_id': grid_id,
                'reason': reason,
                'cancelled_orders': len(cancelled_orders)
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler bei Grid-Deaktivierung für {grid_id}")
            return {'success': False, 'error': str(e)}
    
    # Hilfsmethoden für technische Analyse
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Berechnet Relative Strength Index"""
        if len(prices) < period + 1:
            return np.array([50])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Berechnet Exponential Moving Average"""
        if len(prices) < period:
            return np.array([np.mean(prices)])
        
        alpha = 2.0 / (period + 1)
        ema_values = [prices[0]]
        
        for price in prices[1:]:
            ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
        
        return np.array(ema_values)
    
    def _calculate_volatility_percentile(self, current_vol: float, prices: np.ndarray, lookback: int = 252) -> float:
        """Berechnet Volatilitäts-Perzentil"""
        if len(prices) < lookback:
            return 0.5
        
        returns = np.diff(prices[-lookback:]) / prices[-lookback:-1]
        historical_vols = []
        
        # Rolling Volatility berechnen
        window = 20
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i])
            historical_vols.append(vol)
        
        if not historical_vols:
            return 0.5
        
        # Perzentil berechnen
        percentile = np.sum(np.array(historical_vols) <= current_vol) / len(historical_vols)
        return percentile
    
    def _classify_volatility_regime(self, percentile: float) -> str:
        """Klassifiziert Volatilitäts-Regime"""
        if percentile < 0.2:
            return 'very_low'
        elif percentile < 0.4:
            return 'low'
        elif percentile < 0.6:
            return 'medium'
        elif percentile < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_optimal_spacing(self, atr: float, price: float, volatility: float) -> float:
        """Berechnet optimalen Grid-Abstand"""
        
        # ATR-basierter Spacing
        atr_spacing = (atr / price) * 100
        
        # Volatilitäts-basierter Spacing
        vol_spacing = volatility * 100 * 2  # 2x Volatilität
        
        # Konservativer Ansatz: Minimum nehmen
        optimal_spacing = max(
            min(atr_spacing, vol_spacing),
            self.grid_config['grid_spacing_percent'] * 0.5  # Minimum 50% der Basis-Spacing
        )
        
        return min(optimal_spacing, self.grid_config['grid_spacing_percent'] * 2)  # Maximum 200% der Basis-Spacing
    
    def _calculate_price_position(self, current_price: float, support: Optional[float], resistance: Optional[float]) -> str:
        """Berechnet Position des Preises in der Range"""
        
        if not support or not resistance:
            return 'unknown'
        
        if current_price <= support:
            return 'at_support'
        elif current_price >= resistance:
            return 'at_resistance'
        else:
            # Position in der Range (0 = Support, 1 = Resistance)
            position_ratio = (current_price - support) / (resistance - support)
            
            if position_ratio < 0.3:
                return 'near_support'
            elif position_ratio > 0.7:
                return 'near_resistance'
            else:
                return 'mid_range'
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analysiert Volume Profile (falls Volume-Daten verfügbar)"""
        
        if 'volume' not in data.columns:
            return {}
        
        try:
            prices = data['close'].values
            volumes = data['volume'].values
            
            # Price-Volume Histogram
            price_bins = 20
            price_min, price_max = np.min(prices), np.max(prices)
            bin_edges = np.linspace(price_min, price_max, price_bins + 1)
            
            volume_profile = []
            for i in range(len(bin_edges) - 1):
                bin_low, bin_high = bin_edges[i], bin_edges[i + 1]
                
                # Volume in diesem Preisbereich
                mask = (prices >= bin_low) & (prices < bin_high)
                bin_volume = np.sum(volumes[mask])
                
                volume_profile.append({
                    'price_low': bin_low,
                    'price_high': bin_high,
                    'volume': bin_volume,
                    'price_center': (bin_low + bin_high) / 2
                })
            
            # High Volume Nodes (HVN) und Low Volume Nodes (LVN) identifizieren
            volumes_only = [vp['volume'] for vp in volume_profile]
            volume_threshold_high = np.percentile(volumes_only, 80)
            volume_threshold_low = np.percentile(volumes_only, 20)
            
            hvn_levels = [vp['price_center'] for vp in volume_profile if vp['volume'] >= volume_threshold_high]
            lvn_levels = [vp['price_center'] for vp in volume_profile if vp['volume'] <= volume_threshold_low]
            
            # Point of Control (POC) - Preis mit höchstem Volume
            poc_entry = max(volume_profile, key=lambda x: x['volume'])
            poc_price = poc_entry['price_center']
            
            return {
                'volume_profile': volume_profile,
                'poc_price': poc_price,
                'hvn_levels': hvn_levels,
                'lvn_levels': lvn_levels,
                'total_volume': np.sum(volumes),
                'avg_volume': np.mean(volumes)
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Fehler bei Volume Profile Analyse")
            return {}
    
    def _generate_grid_recommendation(self, sideways_score: float, grid_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generiert Grid-Trading Empfehlung"""
        
        if sideways_score < 0.3:
            recommendation = 'not_recommended'
            confidence = 'low'
            reason = 'Market shows strong directional movement'
        elif sideways_score < 0.6:
            recommendation = 'caution'
            confidence = 'medium'
            reason = 'Market conditions moderately suitable'
        else:
            recommendation = 'recommended'
            confidence = 'high'
            reason = 'Excellent sideways market conditions'
        
        # Empfohlene Parameter-Anpassungen
        adjustments = {}
        
        if sideways_score < 0.5:
            adjustments['spacing_percent'] = grid_params['spacing_percent'] * 1.5
            adjustments['position_multiplier'] = grid_params['position_multiplier'] * 0.7
            adjustments['note'] = 'Increased spacing and reduced position size due to uncertain conditions'
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reason': reason,
            'score': sideways_score,
            'suggested_adjustments': adjustments,
            'optimal_capital_allocation': min(1.0, sideways_score * 1.2)
        }
    
    def _calculate_required_capital(self, price: float, spacing_percent: float, max_range: float) -> float:
        """Berechnet benötigtes Kapital für Grid"""
        
        # Anzahl möglicher Positionen
        num_positions = int(max_range / spacing_percent)
        
        # Durchschnittliche Positionsgröße
        avg_position_size = self.grid_config['base_position_size']
        
        # Durchschnittlicher Positionswert
        avg_position_value = avg_position_size * price
        
        # Sicherheitsmarge
        safety_margin = 1.5
        
        required_capital = num_positions * avg_position_value * safety_margin
        
        return required_capital
    
    def _calculate_current_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Berechnet aktuelle Volatilität"""
        
        if len(data) < period:
            return 0.02  # Default 2%
        
        closes = data['close'].values[-period:]
        returns = np.diff(closes) / closes[:-1]
        
        return np.std(returns)
    
    def get_active_grids(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle aktiven Grids zurück"""
        with self.lock:
            return {grid_id: grid for grid_id, grid in self.active_grids.items() 
                   if grid['status'] == 'active'}
    
    def get_grid_status(self, grid_id: str) -> Dict[str, Any]:
        """Gibt detaillierten Status eines Grids zurück"""
        
        if grid_id not in self.active_grids:
            return {'exists': False}
        
        grid = self.active_grids[grid_id]
        
        # Statistiken berechnen
        filled_levels = [level for level in grid['levels'] if level.filled]
        pending_orders = [order for order in self.grid_orders.get(grid_id, []) if order.status == 'pending']
        
        return {
            'exists': True,
            'grid_id': grid_id,
            'symbol': grid['symbol'],
            'exchange': grid['exchange'],
            'status': grid['status'],
            'created_at': grid['created_at'].isoformat(),
            'center_price': grid['center_price'],
            'current_price': grid.get('current_price', grid['center_price']),
            'grid_type': grid['grid_type'],
            'spacing_percent': grid['spacing_percent'],
            'total_levels': len(grid['levels']),
            'filled_levels': len(filled_levels),
            'pending_orders': len(pending_orders),
            'total_trades': grid['total_trades'],
            'realized_pnl': grid['realized_pnl'],
            'unrealized_pnl': grid['unrealized_pnl'],
            'last_rebalance': grid['last_rebalance'].isoformat(),
            'last_update': grid.get('last_update', grid['created_at']).isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gibt Performance-Metriken der Strategie zurück"""
        
        with self.lock:
            active_grids_count = len([g for g in self.active_grids.values() if g['status'] == 'active'])
            total_grids = len(self.active_grids)
            
            metrics = self.performance_metrics.copy()
            metrics.update({
                'active_grids': active_grids_count,
                'total_grids': total_grids,
                'success_rate': (metrics['profitable_trades'] / max(1, metrics['total_trades'])) * 100,
                'avg_profit_per_trade': metrics['total_profit'] / max(1, metrics['total_trades']),
                'strategy_name': self.strategy_name,
                'version': self.version,
                'last_update': datetime.now().isoformat()
            })
            
            return metrics
    
    def cleanup_inactive_grids(self, max_age_hours: int = 24) -> int:
        """Räumt alte inaktive Grids auf"""
        
        cleanup_threshold = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self.lock:
            grids_to_remove = []
            
            for grid_id, grid in self.active_grids.items():
                if (grid['status'] != 'active' and 
                    grid.get('deactivated_at', grid['created_at']) < cleanup_threshold):
                    grids_to_remove.append(grid_id)
            
            for grid_id in grids_to_remove:
                del self.active_grids[grid_id]
                if grid_id in self.grid_orders:
                    del self.grid_orders[grid_id]
                cleaned_count += 1
        
        if self.logger and cleaned_count > 0:
            self.logger.info(f"Cleanup: {cleaned_count} inaktive Grids entfernt", 'strategy')
        
        return cleaned_count
    
    def save_grid_state(self, filepath: str) -> bool:
        """Speichert Grid-Zustand in Datei"""
        
        try:
            state_data = {
                'active_grids': {},
                'performance_metrics': self.performance_metrics,
                'version': self.version,
                'timestamp': datetime.now().isoformat()
            }
            
                        # Nur serialisierbare Daten extrahieren
            for grid_id, grid in self.active_grids.items():
                state_data['active_grids'][grid_id] = {
                    'symbol': grid['symbol'],
                    'exchange': grid['exchange'],
                    'status': grid['status'],
                    'created_at': grid['created_at'].isoformat(),
                    'center_price': grid['center_price'],
                    'grid_type': grid['grid_type'],
                    'spacing_percent': grid['spacing_percent'],
                    'total_trades': grid['total_trades'],
                    'realized_pnl': grid['realized_pnl'],
                    'unrealized_pnl': grid['unrealized_pnl'],
                    'last_rebalance': grid['last_rebalance'].isoformat(),
                    'levels': [
                        {
                            'price': level.price,
                            'level_id': level.level_id,
                            'is_buy': level.is_buy,
                            'is_sell': level.is_sell,
                            'order_id': level.order_id,
                            'filled': level.filled,
                            'fill_time': level.fill_time.isoformat() if level.fill_time else None,
                            'fill_price': level.fill_price,
                            'amount': level.amount,
                            'profit_target': level.profit_target
                        }
                        for level in grid['levels']
                    ]
                }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=4)

            if self.logger:
                self.logger.info(f"Grid-Status gespeichert in {filepath}", 'strategy')

            return True

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Fehler beim Speichern des Grid-Zustands in {filepath}")
            return False
