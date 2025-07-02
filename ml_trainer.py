"""
Trading Bot ML Trainer System
Machine Learning für Strategieoptimierung und Marktvorhersagen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn Imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import TimeSeriesSplit

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

# Technical Analysis
import talib
from scipy import stats
import threading
import time
from enum import Enum

class ModelType(Enum):
    MARKET_CONDITION = "market_condition"
    PRICE_DIRECTION = "price_direction"
    VOLATILITY_PREDICTION = "volatility_prediction"
    STRATEGY_SELECTION = "strategy_selection"
    ENTRY_SIGNAL = "entry_signal"
    EXIT_SIGNAL = "exit_signal"
    RISK_ASSESSMENT = "risk_assessment"

class MLTrainer:
    def __init__(self, config_manager, logger, data_manager):
        self.config = config_manager
        self.logger = logger
        self.data_manager = data_manager
        
        # Model Storage
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_metadata = {}
        
        # Pfade
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Training History
        self.training_history = {}
        
        # Feature Engineering
        self.feature_columns = []
        self.target_columns = []
        
        # Model Configurations
        self.model_configs = self._get_model_configurations()
        
        # Threading
        self.training_lock = threading.Lock()
        
        self.logger.info("ML Trainer initialisiert", 'ml')
    
    def _get_model_configurations(self) -> Dict[str, Dict]:
        """Definiert Konfigurationen für verschiedene ML-Modelle"""
        return {
            'random_forest_classifier': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'type': 'classifier'
            },
            'random_forest_regressor': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'type': 'regressor'
            },
            'gradient_boosting_classifier': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'type': 'classifier'
            },
            'gradient_boosting_regressor': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'type': 'regressor'
            },
            'svm_classifier': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                'type': 'classifier'
            },
            'neural_network_classifier': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'max_iter': [500]
                },
                'type': 'classifier'
            },
            'neural_network_regressor': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'max_iter': [500]
                },
                'type': 'regressor'
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'type': 'classifier'
            }
        }
    
    def create_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Erstellt umfassende Features für ML-Training"""
        try:
            self.logger.info(f"Erstelle Features für {symbol}", 'ml')
            
            features_df = df.copy()
            
            # Basis-Features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            features_df['volatility'] = features_df['returns'].rolling(20).std()
            
            # Technische Indikatoren - Trend
            features_df['sma_5'] = talib.SMA(features_df['close'], 5)
            features_df['sma_10'] = talib.SMA(features_df['close'], 10)
            features_df['sma_20'] = talib.SMA(features_df['close'], 20)
            features_df['sma_50'] = talib.SMA(features_df['close'], 50)
            features_df['ema_12'] = talib.EMA(features_df['close'], 12)
            features_df['ema_26'] = talib.EMA(features_df['close'], 26)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(features_df['close'])
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_histogram'] = macd_hist
            
            # RSI
            features_df['rsi'] = talib.RSI(features_df['close'], 14)
            features_df['rsi_oversold'] = (features_df['rsi'] < 30).astype(int)
            features_df['rsi_overbought'] = (features_df['rsi'] > 70).astype(int)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(features_df['close'])
            features_df['bb_upper'] = bb_upper
            features_df['bb_middle'] = bb_middle
            features_df['bb_lower'] = bb_lower
            features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features_df['bb_position'] = (features_df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(features_df['high'], features_df['low'], features_df['close'])
            features_df['stoch_k'] = slowk
            features_df['stoch_d'] = slowd
            
            # Williams %R
            features_df['williams_r'] = talib.WILLR(features_df['high'], features_df['low'], features_df['close'])
            
            # ATR (Average True Range)
            features_df['atr'] = talib.ATR(features_df['high'], features_df['low'], features_df['close'])
            
            # ADX (Average Directional Index)
            features_df['adx'] = talib.ADX(features_df['high'], features_df['low'], features_df['close'])
            
            # Volumen-Indikatoren
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # On-Balance Volume
            features_df['obv'] = talib.OBV(features_df['close'], features_df['volume'])
            
            # Price-Volume Features
            features_df['price_volume'] = features_df['close'] * features_df['volume']
            features_df['vwap'] = (features_df['price_volume'].rolling(20).sum() / 
                                  features_df['volume'].rolling(20).sum())
            
            # Momentum Features
            features_df['momentum_5'] = features_df['close'] / features_df['close'].shift(5) - 1
            features_df['momentum_10'] = features_df['close'] / features_df['close'].shift(10) - 1
            features_df['momentum_20'] = features_df['close'] / features_df['close'].shift(20) - 1
            
            # Support/Resistance Levels
            features_df['high_20'] = features_df['high'].rolling(20).max()
            features_df['low_20'] = features_df['low'].rolling(20).min()
            features_df['resistance_distance'] = (features_df['high_20'] - features_df['close']) / features_df['close']
            features_df['support_distance'] = (features_df['close'] - features_df['low_20']) / features_df['close']
            
            # Volatility Measures
            features_df['volatility_5'] = features_df['returns'].rolling(5).std()
            features_df['volatility_20'] = features_df['returns'].rolling(20).std()
            features_df['volatility_ratio'] = features_df['volatility_5'] / features_df['volatility_20']
            
            # Gap Analysis
            features_df['gap'] = (features_df['open'] - features_df['close'].shift(1)) / features_df['close'].shift(1)
            features_df['gap_up'] = (features_df['gap'] > 0.005).astype(int)
            features_df['gap_down'] = (features_df['gap'] < -0.005).astype(int)
            
            # Candlestick Patterns (vereinfacht)
            features_df['doji'] = (abs(features_df['open'] - features_df['close']) / 
                                  (features_df['high'] - features_df['low']) < 0.1).astype(int)
            features_df['hammer'] = ((features_df['close'] > features_df['open']) & 
                                    ((features_df['open'] - features_df['low']) > 
                                     2 * (features_df['close'] - features_df['open']))).astype(int)
            
            # Time-based Features
            if 'timestamp' in features_df.columns:
                features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
                features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
                features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            
            # Lag Features
            for lag in [1, 2, 3, 5]:
                features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
                features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
                features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            
            # Rolling Statistics
            for window in [5, 10, 20]:
                features_df[f'returns_mean_{window}'] = features_df['returns'].rolling(window).mean()
                features_df[f'returns_std_{window}'] = features_df['returns'].rolling(window).std()
                features_df[f'returns_skew_{window}'] = features_df['returns'].rolling(window).skew()
                features_df[f'returns_kurt_{window}'] = features_df['returns'].rolling(window).kurt()
            
            # Relative Strength
            features_df['rs_5'] = features_df['close'] / features_df['sma_5']
            features_df['rs_20'] = features_df['close'] / features_df['sma_20']
            features_df['rs_50'] = features_df['close'] / features_df['sma_50']
            
            # Market Condition Features
            features_df['trend_strength'] = abs(features_df['close'] - features_df['sma_20']) / features_df['sma_20']
            features_df['is_uptrend'] = (features_df['close'] > features_df['sma_20']).astype(int)
            features_df['is_downtrend'] = (features_df['close'] < features_df['sma_20']).astype(int)
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            self.logger.info(f"Features erstellt: {len(features_df.columns)} Spalten, {len(features_df)} Zeilen", 'ml')
            return features_df
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Erstellen der Features für {symbol}")
            return df
    
    def create_targets(self, df: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
        """Erstellt Target-Variablen basierend auf dem Modell-Typ"""
        try:
            targets_df = df.copy()
            
            if model_type == ModelType.MARKET_CONDITION:
                # Marktbedingungen: 0=Abwärts, 1=Seitwärts, 2=Aufwärts
                returns_5 = df['close'].pct_change(5)
                volatility = df['close'].pct_change().rolling(20).std()
                
                conditions = []
                for i, (ret, vol) in enumerate(zip(returns_5, volatility)):
                    if pd.isna(ret) or pd.isna(vol):
                        conditions.append(np.nan)
                    elif ret > 0.02 and vol < 0.05:  # Starker Aufwärtstrend
                        conditions.append(2)
                    elif ret < -0.02 and vol < 0.05:  # Starker Abwärtstrend
                        conditions.append(0)
                    else:  # Seitwärts oder unklarer Trend
                        conditions.append(1)
                
                targets_df['market_condition'] = conditions
            
            elif model_type == ModelType.PRICE_DIRECTION:
                # Preisrichtung: 0=Abwärts, 1=Aufwärts
                future_returns = df['close'].shift(-5) / df['close'] - 1
                targets_df['price_direction'] = (future_returns > 0).astype(int)
            
            elif model_type == ModelType.VOLATILITY_PREDICTION:
                # Volatilitätsvorhersage
                future_volatility = df['close'].pct_change().rolling(5).std().shift(-5)
                targets_df['future_volatility'] = future_volatility
            
            elif model_type == ModelType.STRATEGY_SELECTION:
                # Beste Strategie basierend auf Marktbedingungen
                returns_5 = df['close'].pct_change(5)
                volatility = df['close'].pct_change().rolling(20).std()
                
                strategies = []
                for ret, vol in zip(returns_5, volatility):
                    if pd.isna(ret) or pd.isna(vol):
                        strategies.append(np.nan)
                    elif ret > 0.015 and vol < 0.04:  # Uptrend Strategy
                        strategies.append(0)
                    elif abs(ret) < 0.01 and vol > 0.02:  # Grid Strategy
                        strategies.append(1)
                    else:  # Conservative/Downtrend Strategy
                        strategies.append(2)
                
                targets_df['best_strategy'] = strategies
            
            elif model_type == ModelType.ENTRY_SIGNAL:
                # Entry-Signale basierend auf zukünftigen Returns
                future_returns = df['close'].shift(-3) / df['close'] - 1
                targets_df['entry_signal'] = (future_returns > 0.01).astype(int)
            
            elif model_type == ModelType.EXIT_SIGNAL:
                # Exit-Signale basierend auf negativen zukünftigen Returns
                future_returns = df['close'].shift(-2) / df['close'] - 1
                targets_df['exit_signal'] = (future_returns < -0.005).astype(int)
            
            elif model_type == ModelType.RISK_ASSESSMENT:
                # Risikobewertung basierend auf Volatilität und Drawdown
                volatility = df['close'].pct_change().rolling(10).std()
                returns = df['close'].pct_change()
                
                # Einfache Risikokategorien: 0=Niedrig, 1=Mittel, 2=Hoch
                risk_levels = []
                for vol, ret in zip(volatility, returns):
                    if pd.isna(vol) or pd.isna(ret):
                        risk_levels.append(np.nan)
                    elif vol < 0.02:
                        risk_levels.append(0)  # Niedriges Risiko
                    elif vol < 0.05:
                        risk_levels.append(1)  # Mittleres Risiko
                    else:
                        risk_levels.append(2)  # Hohes Risiko
                
                targets_df['risk_level'] = risk_levels
            
            return targets_df
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Erstellen der Targets für {model_type}")
            return df
    
    def prepare_training_data(self, symbol: str, model_type: ModelType, 
                            start_date: str = None, end_date: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Bereitet Trainingsdaten vor"""
        try:
            self.logger.info(f"Bereite Trainingsdaten vor für {symbol}, Modell: {model_type}", 'ml')
            
            # Daten laden
            df = self.data_manager.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) < 100:
                raise ValueError(f"Nicht genügend Daten für {symbol}")
            
            # Features erstellen
            features_df = self.create_features(df, symbol)
            
            # Targets erstellen
            targets_df = self.create_targets(features_df, model_type)
            
            # Feature-Spalten definieren (alles außer Targets und Basis-Spalten)
            exclude_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                             'market_condition', 'price_direction', 'future_volatility',
                             'best_strategy', 'entry_signal', 'exit_signal', 'risk_level']
            
            feature_columns = [col for col in targets_df.columns if col not in exclude_columns]
            
            # Target-Spalte basierend auf Modell-Typ
            target_mapping = {
                ModelType.MARKET_CONDITION: 'market_condition',
                ModelType.PRICE_DIRECTION: 'price_direction',
                ModelType.VOLATILITY_PREDICTION: 'future_volatility',
                ModelType.STRATEGY_SELECTION: 'best_strategy',
                ModelType.ENTRY_SIGNAL: 'entry_signal',
                ModelType.EXIT_SIGNAL: 'exit_signal',
                ModelType.RISK_ASSESSMENT: 'risk_level'
            }
            
            target_column = target_mapping[model_type]
            
            if target_column not in targets_df.columns:
                raise ValueError(f"Target-Spalte {target_column} nicht gefunden")
            
            # Daten extrahieren
            X = targets_df[feature_columns].values
            y = targets_df[target_column].values
            
            # NaN-Werte entfernen
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            self.feature_columns = feature_columns
            
            self.logger.info(f"Trainingsdaten vorbereitet: {X.shape[0]} Samples, {X.shape[1]} Features", 'ml')
            return X, y
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Vorbereiten der Trainingsdaten für {symbol}")
            return None, None
    
    def train_model(self, symbol: str, model_type: ModelType, model_name: str,
                   start_date: str = None, end_date: str = None, 
                   use_grid_search: bool = True) -> Dict[str, Any]:
        """Trainiert ein ML-Modell"""
        with self.training_lock:
            try:
                self.logger.info(f"Starte Training: {model_name} für {symbol}, Typ: {model_type}", 'ml')
                
                # Trainingsdaten vorbereiten
                X, y = self.prepare_training_data(symbol, model_type, start_date, end_date)
                
                if X is None or len(X) < 50:
                    raise ValueError("Nicht genügend Trainingsdaten")
                
                # Modell-Konfiguration laden
                if model_name not in self.model_configs:
                    raise ValueError(f"Modell {model_name} nicht in Konfiguration gefunden")
                
                config = self.model_configs[model_name]
                
                # Daten skalieren
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y if config['type'] == 'classifier' else None
                )
                
                # Feature Selection (optional)
                if len(self.feature_columns) > 50:
                    if config['type'] == 'classifier':
                        selector = SelectKBest(f_classif, k=min(30, len(self.feature_columns)))
                    else:
                        selector = SelectKBest(f_regression, k=min(30, len(self.feature_columns)))
                    
                    X_train = selector.fit_transform(X_train, y_train)
                    X_test = selector.transform(X_test)
                    
                    # Ausgewählte Features speichern
                    selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
                    self.logger.info(f"Feature Selection: {len(selected_features)} von {len(self.feature_columns)} Features ausgewählt", 'ml')
                else:
                    selector = None
                    selected_features = self.feature_columns
                
                # Modell trainieren
                if use_grid_search and len(X_train) > 100:
                    # Grid Search für Hyperparameter-Optimierung
                    model = config['model']()
                    
                    if config['type'] == 'classifier':
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        scoring = 'accuracy'
                    else:
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        scoring = 'neg_mean_squared_error'
                    
                    grid_search = GridSearchCV(
                        model, config['params'], cv=cv, scoring=scoring, n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                    self.logger.info(f"Grid Search abgeschlossen. Beste Parameter: {best_params}", 'ml')
                
                else:
                    # Standard-Training ohne Grid Search
                    best_model = config['model']()
                    best_model.fit(X_train, y_train)
                    best_params = {}
                
                # Modell evaluieren
                train_predictions = best_model.predict(X_train)
                test_predictions = best_model.predict(X_test)
                
                # Metriken berechnen
                if config['type'] == 'classifier':
                    train_accuracy = accuracy_score(y_train, train_predictions)
                    test_accuracy = accuracy_score(y_test, test_predictions)
                    
                    metrics = {
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'precision': precision_score(y_test, test_predictions, average='weighted'),
                        'recall': recall_score(y_test, test_predictions, average='weighted'),
                        'f1_score': f1_score(y_test, test_predictions, average='weighted')
                    }
                    
                    # Classification Report
                    class_report = classification_report(y_test, test_predictions, output_dict=True)
                    
                else:
                    train_mse = mean_squared_error(y_train, train_predictions)
                    test_mse = mean_squared_error(y_test, test_predictions)
                    
                    metrics = {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_r2': r2_score(y_train, train_predictions),
                        'test_r2': r2_score(y_test, test_predictions),
                        'mae': mean_absolute_error(y_test, test_predictions)
                    }
                    
                    class_report = None
                
                # Modell speichern
                model_key = f"{symbol}_{model_type.value}_{model_name}"
                
                self.models[model_key] = best_model
                self.scalers[model_key] = scaler
                self.feature_selectors[model_key] = selector
                
                # Metadata speichern
                self.model_metadata[model_key] = {
                    'symbol': symbol,
                    'model_type': model_type.value,
                    'model_name': model_name,
                    'trained_date': datetime.now().isoformat(),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': selected_features,
                    'metrics': metrics,
                    'best_params': best_params,
                    'classification_report': class_report
                }
                
                # Modell auf Festplatte speichern
                self._save_model_to_disk(model_key)
                
                # Training loggen
                self.logger.log_ml_training(
                    model_name=model_key,
                    accuracy=metrics.get('test_accuracy', metrics.get('test_r2', 0)),
                    loss=metrics.get('test_mse', 1 - metrics.get('test_accuracy', 0)),
                    training_samples=len(X_train),
                    validation_accuracy=metrics.get('test_accuracy', metrics.get('test_r2'))
                )
                
                return {
                    'success': True,
                    'model_key': model_key,
                    'metrics': metrics,
                    'features_used': len(selected_features),
                    'training_samples': len(X_train)
                }
                
            except Exception as e:
                self.logger.log_error(e, f"Fehler beim Training von {model_name} für {symbol}")
                return {'success': False, 'error': str(e)}
    
    def predict(self, symbol: str, model_type: ModelType, model_name: str, 
               current_data: pd.DataFrame) -> Dict[str, Any]:
        """Macht Vorhersagen mit einem trainierten Modell"""
        try:
            model_key = f"{symbol}_{model_type.value}_{model_name}"
            
            if model_key not in self.models:
                raise ValueError(f"Modell {model_key} nicht gefunden")
            
            # Features aus aktuellen Daten erstellen
            features_df = self.create_features(current_data, symbol)
            
            if len(features_df) == 0:
                raise ValueError("Keine Features aus aktuellen Daten erstellt")
            
            # Letzte Zeile für Prediction verwenden
            last_row = features_df.iloc[-1]
            
            # Feature-Vektor erstellen
            X = np.array([last_row[self.model_metadata[model_key]['features']].values])
            
            # Skalieren
            X_scaled = self.scalers[model_key].transform(X)
            
            # Feature Selection anwenden (falls verwendet)
            if self.feature_selectors[model_key] is not None:
                X_scaled = self.feature_selectors[model_key].transform(X_scaled)
            
            # Vorhersage
            model = self.models[model_key]
            prediction = model.predict(X_scaled)[0]
            
            # Wahrscheinlichkeiten für Klassifikation
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
            else:
                probabilities = None
            
            return {
                'prediction': prediction,
                'probabilities': probabilities,
                'confidence': max(probabilities) if probabilities is not None else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler bei Vorhersage für {symbol}")
            return None
    
    def batch_predict(self, symbol: str, model_type: ModelType, model_name: str,
                     data_df: pd.DataFrame) -> np.ndarray:
        """Macht Batch-Vorhersagen für mehrere Datenpunkte"""
        try:
            model_key = f"{symbol}_{model_type.value}_{model_name}"
            
            if model_key not in self.models:
                raise ValueError(f"Modell {model_key} nicht gefunden")
            
            # Features erstellen
            features_df = self.create_features(data_df, symbol)
            
            if len(features_df) == 0:
                return np.array([])
            
            # Feature-Matrix erstellen
            X = features_df[self.model_metadata[model_key]['features']].values
            
            # NaN-Werte behandeln
            mask = ~np.isnan(X).any(axis=1)
            X_clean = X[mask]
            
            if len(X_clean) == 0:
                return np.array([])
            
            # Skalieren
            X_scaled = self.scalers[model_key].transform(X_clean)
            
            # Feature Selection anwenden
            if self.feature_selectors[model_key] is not None:
                X_scaled = self.feature_selectors[model_key].transform(X_scaled)
            
            # Vorhersagen
            predictions = self.models[model_key].predict(X_scaled)
            
            # Ergebnis-Array mit NaN für ausgelassene Zeilen
            full_predictions = np.full(len(features_df), np.nan)
            full_predictions[mask] = predictions
            
            return full_predictions
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler bei Batch-Vorhersage für {symbol}")
            return np.array([])
    
    def evaluate_model(self, symbol: str, model_type: ModelType, model_name: str,
                      test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Evaluiert ein trainiertes Modell"""
        try:
            model_key = f"{symbol}_{model_type.value}_{model_name}"
            
            if model_key not in self.models:
                raise ValueError(f"Modell {model_key} nicht gefunden")
            
            # Test-Daten vorbereiten
            if test_data is None:
                # Aktuelle Daten für Test verwenden
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                test_data = self.data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe='1h',
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
            
            if test_data is None or len(test_data) < 20:
                raise ValueError("Nicht genügend Test-Daten")
            
            # Features und Targets erstellen
            features_df = self.create_features(test_data, symbol)
            targets_df = self.create_targets(features_df, model_type)
            
            # Target-Spalte bestimmen
            target_mapping = {
                ModelType.MARKET_CONDITION: 'market_condition',
                ModelType.PRICE_DIRECTION: 'price_direction',
                ModelType.VOLATILITY_PREDICTION: 'future_volatility',
                ModelType.STRATEGY_SELECTION: 'best_strategy',
                ModelType.ENTRY_SIGNAL: 'entry_signal',
                ModelType.EXIT_SIGNAL: 'exit_signal',
                ModelType.RISK_ASSESSMENT: 'risk_level'
            }
            
            target_column = target_mapping[model_type]
            
            # Daten extrahieren
            X = targets_df[self.model_metadata[model_key]['features']].values
            y = targets_df[target_column].values
            
            # NaN entfernen
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                raise ValueError("Nicht genügend saubere Test-Daten")
            
            # Skalieren und transformieren
            X_scaled = self.scalers[model_key].transform(X)
            
            if self.feature_selectors[model_key] is not None:
                X_scaled = self.feature_selectors[model_key].transform(X_scaled)
            
            # Vorhersagen
            model = self.models[model_key]
            predictions = model.predict(X_scaled)
            
            # Metriken berechnen
            model_config = self.model_configs[model_name]
            
            if model_config['type'] == 'classifier':
                metrics = {
                    'accuracy': accuracy_score(y, predictions),
                    'precision': precision_score(y, predictions, average='weighted'),
                    'recall': recall_score(y, predictions, average='weighted'),
                    'f1_score': f1_score(y, predictions, average='weighted')
                }
                
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X_scaled)
                        if len(np.unique(y)) == 2:  # Binary classification
                            metrics['auc_roc'] = roc_auc_score(y, probabilities[:, 1])
                    except:
                        pass
                        
            else:
                metrics = {
                    'mse': mean_squared_error(y, predictions),
                    'mae': mean_absolute_error(y, predictions),
                    'r2': r2_score(y, predictions)
                }
            
            # Evaluation speichern
            evaluation_result = {
                'model_key': model_key,
                'evaluation_date': datetime.now().isoformat(),
                'test_samples': len(y),
                'metrics': metrics,
                'predictions_sample': predictions[:10].tolist(),
                'actual_sample': y[:10].tolist()
            }
            
            self.logger.info(f"Modell evaluiert: {model_key}, Metriken: {metrics}", 'ml')
            
            return evaluation_result
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler bei Modell-Evaluation für {symbol}")
            return None
    
    def retrain_model(self, symbol: str, model_type: ModelType, model_name: str,
                     days_back: int = 30) -> Dict[str, Any]:
        """Trainiert ein Modell mit aktuellen Daten neu"""
        try:
            self.logger.info(f"Starte Retraining für {model_name}, {symbol}", 'ml')
            
            # Aktuelle Daten laden
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            result = self.train_model(
                symbol=symbol,
                model_type=model_type,
                model_name=model_name,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                use_grid_search=False  # Schnelleres Retraining
            )
            
            if result['success']:
                self.logger.info(f"Retraining erfolgreich: {result['model_key']}", 'ml')
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Retraining für {symbol}")
            return {'success': False, 'error': str(e)}
    
    def get_model_importance(self, symbol: str, model_type: ModelType, 
                           model_name: str) -> Dict[str, float]:
        """Ermittelt Feature-Wichtigkeiten eines Modells"""
        try:
            model_key = f"{symbol}_{model_type.value}_{model_name}"
            
            if model_key not in self.models:
                raise ValueError(f"Modell {model_key} nicht gefunden")
            
            model = self.models[model_key]
            features = self.model_metadata[model_key]['features']
            
            # Feature Importance extrahieren
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return {}
            
            # Features und Wichtigkeiten kombinieren
            feature_importance = dict(zip(features, importances))
            
            # Nach Wichtigkeit sortieren
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler bei Feature-Importance für {symbol}")
            return {}
    
    def _save_model_to_disk(self, model_key: str):
        """Speichert Modell und Metadaten auf Festplatte"""
        try:
            model_path = self.model_dir / f"{model_key}"
            model_path.mkdir(exist_ok=True)
            
            # Modell speichern
            joblib.dump(self.models[model_key], model_path / "model.pkl")
            
            # Scaler speichern
            joblib.dump(self.scalers[model_key], model_path / "scaler.pkl")
            
            # Feature Selector speichern
            if self.feature_selectors[model_key] is not None:
                joblib.dump(self.feature_selectors[model_key], model_path / "selector.pkl")
            
            # Metadaten speichern
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(self.model_metadata[model_key], f, indent=2)
            
            self.logger.info(f"Modell gespeichert: {model_key}", 'ml')
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Speichern von {model_key}")
    
    def load_model_from_disk(self, model_key: str) -> bool:
        """Lädt Modell von Festplatte"""
        try:
            model_path = self.model_dir / f"{model_key}"
            
            if not model_path.exists():
                return False
            
            # Modell laden
            self.models[model_key] = joblib.load(model_path / "model.pkl")
            
            # Scaler laden
            self.scalers[model_key] = joblib.load(model_path / "scaler.pkl")
            
            # Feature Selector laden
            selector_path = model_path / "selector.pkl"
            if selector_path.exists():
                self.feature_selectors[model_key] = joblib.load(selector_path)
            else:
                self.feature_selectors[model_key] = None
            
            # Metadaten laden
            with open(model_path / "metadata.json", 'r') as f:
                self.model_metadata[model_key] = json.load(f)
            
            self.logger.info(f"Modell geladen: {model_key}", 'ml')
            return True
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Laden von {model_key}")
            return False
    
    def load_all_models(self):
        """Lädt alle gespeicherten Modelle"""
        try:
            loaded_count = 0
            
            for model_dir in self.model_dir.iterdir():
                if model_dir.is_dir():
                    model_key = model_dir.name
                    if self.load_model_from_disk(model_key):
                        loaded_count += 1
            
            self.logger.info(f"{loaded_count} Modelle geladen", 'ml')
            
        except Exception as e:
            self.logger.log_error(e, "Fehler beim Laden aller Modelle")
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """Gibt Liste aller verfügbaren Modelle zurück"""
        model_list = []
        
        for model_key, metadata in self.model_metadata.items():
            model_info = {
                'model_key': model_key,
                'symbol': metadata['symbol'],
                'model_type': metadata['model_type'],
                'model_name': metadata['model_name'],
                'trained_date': metadata['trained_date'],
                'training_samples': metadata['training_samples'],
                'metrics': metadata['metrics']
            }
            model_list.append(model_info)
        
        return model_list
    
    def delete_model(self, model_key: str) -> bool:
        """Löscht ein Modell"""
        try:
            # Aus Memory löschen
            if model_key in self.models:
                del self.models[model_key]
            if model_key in self.scalers:
                del self.scalers[model_key]
            if model_key in self.feature_selectors:
                del self.feature_selectors[model_key]
            if model_key in self.model_metadata:
                del self.model_metadata[model_key]
            
            # Von Festplatte löschen
            model_path = self.model_dir / f"{model_key}"
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            
            self.logger.info(f"Modell gelöscht: {model_key}", 'ml')
            return True
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Löschen von {model_key}")
            return False
    
    def auto_train_models(self, symbols: List[str], days_back: int = 90) -> Dict[str, Any]:
        """Trainiert automatisch Modelle für alle Symbole und Typen"""
        try:
            self.logger.info(f"Starte Auto-Training für {len(symbols)} Symbole", 'ml')
            
            results = {}
            total_models = len(symbols) * len(ModelType) * 3  # 3 verschiedene Algorithmen
            trained_models = 0
            failed_models = 0
            
            # Datums-Range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            for symbol in symbols:
                results[symbol] = {}
                
                # Für jeden Model-Type
                for model_type in ModelType:
                    results[symbol][model_type.value] = {}
                    
                    # Verschiedene Algorithmen testen
                    algorithms = ['random_forest_classifier', 'gradient_boosting_classifier', 'neural_network_classifier']
                    if model_type == ModelType.VOLATILITY_PREDICTION:
                        algorithms = ['random_forest_regressor', 'gradient_boosting_regressor', 'neural_network_regressor']
                    
                    for algorithm in algorithms:
                        try:
                            result = self.train_model(
                                symbol=symbol,
                                model_type=model_type,
                                model_name=algorithm,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'),
                                use_grid_search=True
                            )
                            
                            results[symbol][model_type.value][algorithm] = result
                            
                            if result['success']:
                                trained_models += 1
                                self.logger.info(f"✓ {symbol} - {model_type.value} - {algorithm}", 'ml')
                            else:
                                failed_models += 1
                                self.logger.error(f"✗ {symbol} - {model_type.value} - {algorithm}: {result.get('error', 'Unbekannter Fehler')}", 'ml')
                        
                        except Exception as e:
                            failed_models += 1
                            self.logger.log_error(e, f"Auto-Training Fehler: {symbol} - {model_type.value} - {algorithm}")
                
                # Zwischenbericht
                self.logger.info(f"Symbol {symbol} abgeschlossen. Erfolg: {trained_models}, Fehler: {failed_models}", 'ml')
            
            # Zusammenfassung
            summary = {
                'total_attempted': total_models,
                'successful': trained_models,
                'failed': failed_models,
                'success_rate': (trained_models / total_models) * 100 if total_models > 0 else 0,
                'results': results
            }
            
            self.logger.info(f"Auto-Training abgeschlossen: {trained_models}/{total_models} erfolgreich ({summary['success_rate']:.1f}%)", 'ml')
            
            return summary
            
        except Exception as e:
            self.logger.log_error(e, "Fehler beim Auto-Training")
            return {'success': False, 'error': str(e)}
    
    def get_best_model_for_type(self, symbol: str, model_type: ModelType) -> str:
        """Ermittelt das beste Modell für einen bestimmten Typ"""
        try:
            best_model = None
            best_score = -np.inf
            
            # Alle Modelle für diesen Symbol und Typ durchsuchen
            for model_key, metadata in self.model_metadata.items():
                if (metadata['symbol'] == symbol and 
                    metadata['model_type'] == model_type.value):
                    
                    # Score basierend auf Modell-Typ
                    if model_type in [ModelType.VOLATILITY_PREDICTION]:
                        # Für Regression: R²-Score verwenden
                        score = metadata['metrics'].get('test_r2', -np.inf)
                    else:
                        # Für Klassifikation: Accuracy verwenden
                        score = metadata['metrics'].get('test_accuracy', -np.inf)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_key
            
            return best_model
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler bei Suche nach bestem Modell für {symbol} - {model_type}")
            return None
    
    def cleanup_old_models(self, days_old: int = 30):
        """Löscht alte Modelle"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            models_to_delete = []
            
            for model_key, metadata in self.model_metadata.items():
                trained_date = datetime.fromisoformat(metadata['trained_date'])
                
                if trained_date < cutoff_date:
                    models_to_delete.append(model_key)
            
            for model_key in models_to_delete:
                if self.delete_model(model_key):
                    deleted_count += 1
            
            self.logger.info(f"{deleted_count} alte Modelle gelöscht", 'ml')
            
        except Exception as e:
            self.logger.log_error(e, "Fehler beim Cleanup alter Modelle")
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Erstellt einen Performance-Report aller Modelle"""
        try:
            report = {
                'total_models': len(self.models),
                'by_symbol': {},
                'by_type': {},
                'by_algorithm': {},
                'average_metrics': {}
            }
            
            # Sammle Statistiken
            all_accuracies = []
            all_r2_scores = []
            
            for model_key, metadata in self.model_metadata.items():
                symbol = metadata['symbol']
                model_type = metadata['model_type']
                model_name = metadata['model_name']
                metrics = metadata['metrics']
                
                # Nach Symbol
                if symbol not in report['by_symbol']:
                    report['by_symbol'][symbol] = []
                report['by_symbol'][symbol].append({
                    'model_key': model_key,
                    'type': model_type,
                    'algorithm': model_name,
                    'metrics': metrics
                })
                
                # Nach Typ
                if model_type not in report['by_type']:
                    report['by_type'][model_type] = []
                report['by_type'][model_type].append({
                    'model_key': model_key,
                    'symbol': symbol,
                    'metrics': metrics
                })
                
                # Nach Algorithmus
                if model_name not in report['by_algorithm']:
                    report['by_algorithm'][model_name] = []
                report['by_algorithm'][model_name].append({
                    'model_key': model_key,
                    'symbol': symbol,
                    'metrics': metrics
                })
                
                # Sammle Metriken
                if 'test_accuracy' in metrics:
                    all_accuracies.append(metrics['test_accuracy'])
                if 'test_r2' in metrics:
                    all_r2_scores.append(metrics['test_r2'])
            
            # Durchschnittliche Metriken
            if all_accuracies:
                report['average_metrics']['accuracy'] = {
                    'mean': np.mean(all_accuracies),
                    'std': np.std(all_accuracies),
                    'min': np.min(all_accuracies),
                    'max': np.max(all_accuracies)
                }
            
            if all_r2_scores:
                report['average_metrics']['r2_score'] = {
                    'mean': np.mean(all_r2_scores),
                    'std': np.std(all_r2_scores),
                    'min': np.min(all_r2_scores),
                    'max': np.max(all_r2_scores)
                }
            
            return report
            
        except Exception as e:
            self.logger.log_error(e, "Fehler beim Erstellen des Performance-Reports")
            return {}
    
    def shutdown(self):
        """Beendet den ML Trainer sauber"""
        try:
            self.logger.info("ML Trainer wird beendet...", 'ml')
            
            # Alle aktuellen Modelle speichern
            for model_key in self.models.keys():
                self._save_model_to_disk(model_key)
            
            # Training History speichern
            history_file = self.model_dir / "training_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            self.logger.info("ML Trainer beendet", 'ml')
            
        except Exception as e:
            self.logger.log_error(e, "Fehler beim Beenden des ML Trainers")