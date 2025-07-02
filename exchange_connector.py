"""
Trading Bot Exchange Connector
Sichere Anbindung an Binance und KuCoin mit Rate Limiting, Fehlerbehandlung und Retry-Logik
"""

import ccxt
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import base64
import json
from threading import Lock, Timer
import traceback

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    PARTIALLY_FILLED = "partially_filled"
    PENDING = "pending"

@dataclass
class Order:
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    type: str
    status: str
    filled: float = 0.0
    remaining: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: int = 0
    exchange: str = ""
    client_order_id: str = ""

@dataclass
class Balance:
    asset: str
    free: float
    used: float
    total: float

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    timestamp: int

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = Lock()
    
    def can_make_request(self) -> bool:
        with self.lock:
            now = time.time()
            # Entferne alte Requests
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            return len(self.requests) < self.max_requests
    
    def add_request(self):
        with self.lock:
            self.requests.append(time.time())
    
    def wait_if_needed(self):
        if not self.can_make_request():
            wait_time = self.time_window - (time.time() - min(self.requests))
            time.sleep(max(0, wait_time))

class ExchangeConnector:
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        
        # Exchange-Instanzen
        self.exchanges = {}
        self.active_exchange = None
        
        # Rate Limiting
        self.rate_limiters = {}
        
        # Retry-Konfiguration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Marktdaten-Cache
        self.market_cache = {}
        self.cache_timeout = 300  # 5 Minuten
        
        # Verbindungsstatus
        self.connection_status = {}
        
        # Emergency Stop
        self.emergency_stop = False
        
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialisiert alle konfigurierten Exchanges"""
        try:
            # Binance initialisieren
            if self._setup_binance():
                self.logger.info("Binance erfolgreich initialisiert", "exchange")
            
            # KuCoin initialisieren
            if self._setup_kucoin():
                self.logger.info("KuCoin erfolgreich initialisiert", "exchange")
            
            # Primäre Exchange setzen
            primary_exchange = self.config_manager.get_config_value('EXCHANGE', 'primary_exchange', 'binance')
            if primary_exchange in self.exchanges:
                self.active_exchange = primary_exchange
                self.logger.info(f"Primäre Exchange gesetzt: {primary_exchange}", "exchange")
            else:
                self.logger.warning(f"Primäre Exchange {primary_exchange} nicht verfügbar", "exchange")
                
        except Exception as e:
            self.logger.log_error(e, "Exchange-Initialisierung fehlgeschlagen")
    
    def _setup_binance(self) -> bool:
        """Binance Exchange Setup"""
        try:
            credentials = self.config_manager.get_api_credentials('binance')
            if not credentials.get('api_key') or not credentials.get('api_secret'):
                self.logger.warning("Binance API-Credentials fehlen", "exchange")
                return False
            
            # Testnet oder Mainnet
            is_testnet = self.config_manager.is_testnet_mode()
            
            binance_config = {
                'apiKey': credentials['api_key'],
                'secret': credentials['api_secret'],
                'sandbox': is_testnet,
                'enableRateLimit': True,
                'rateLimit': 1200,  # Requests per minute
                'options': {
                    'defaultType': 'spot',  # spot, margin, future
                    'recvWindow': 10000,
                    'timeDifference': 0,
                    'adjustForTimeDifference': True
                }
            }
            
            if is_testnet:
                binance_config['urls'] = {
                    'api': 'https://testnet.binance.vision',
                    'public': 'https://testnet.binance.vision',
                    'private': 'https://testnet.binance.vision'
                }
            
            self.exchanges['binance'] = ccxt.binance(binance_config)
            
            # Rate Limiter für Binance
            self.rate_limiters['binance'] = RateLimiter(1200, 60)  # 1200 requests per minute
            
            # Verbindung testen
            self._test_connection('binance')
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, "Binance Setup fehlgeschlagen")
            return False
    
    def _setup_kucoin(self) -> bool:
        """KuCoin Exchange Setup"""
        try:
            credentials = self.config_manager.get_api_credentials('kucoin')
            if not credentials.get('api_key') or not credentials.get('api_secret'):
                self.logger.warning("KuCoin API-Credentials fehlen", "exchange")
                return False
            
            is_testnet = self.config_manager.is_testnet_mode()
            
            kucoin_config = {
                'apiKey': credentials['api_key'],
                'secret': credentials['api_secret'],
                'password': credentials.get('passphrase', ''),
                'sandbox': is_testnet,
                'enableRateLimit': True,
                'rateLimit': 1800,  # Requests per minute
                'options': {
                    'defaultType': 'spot'
                }
            }
            
            self.exchanges['kucoin'] = ccxt.kucoin(kucoin_config)
            
            # Rate Limiter für KuCoin
            self.rate_limiters['kucoin'] = RateLimiter(1800, 60)  # 1800 requests per minute
            
            # Verbindung testen
            self._test_connection('kucoin')
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, "KuCoin Setup fehlgeschlagen")
            return False
    
    def _test_connection(self, exchange_name: str) -> bool:
        """Testet die Verbindung zu einer Exchange"""
        try:
            if exchange_name not in self.exchanges:
                return False
            
            exchange = self.exchanges[exchange_name]
            
            # Einfacher API-Test
            exchange.load_markets()
            balance = exchange.fetch_balance()
            
            self.connection_status[exchange_name] = {
                'connected': True,
                'last_check': datetime.now().isoformat(),
                'error': None
            }
            
            self.logger.info(f"{exchange_name} Verbindung erfolgreich getestet", "exchange")
            return True
            
        except Exception as e:
            self.connection_status[exchange_name] = {
                'connected': False,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
            
            self.logger.error(f"{exchange_name} Verbindungstest fehlgeschlagen: {e}", "exchange")
            return False
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """Führt eine Funktion mit Retry-Logik aus"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except ccxt.NetworkError as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Netzwerkfehler (Versuch {attempt + 1}), warte {wait_time}s: {e}", "exchange")
                    time.sleep(wait_time)
                else:
                    raise e
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 60  # 1 Minute bei Rate Limit
                    self.logger.warning(f"Rate Limit erreicht (Versuch {attempt + 1}), warte {wait_time}s", "exchange")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                # Andere Fehler werden direkt weitergegeben
                raise e
        
        raise last_exception
    
    def get_exchange(self, exchange_name: str = None):
        """Gibt eine Exchange-Instanz zurück"""
        if exchange_name is None:
            exchange_name = self.active_exchange
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} nicht verfügbar")
        
        return self.exchanges[exchange_name]
    
    def switch_exchange(self, exchange_name: str) -> bool:
        """Wechselt die aktive Exchange"""
        if exchange_name not in self.exchanges:
            self.logger.error(f"Exchange {exchange_name} nicht verfügbar", "exchange")
            return False
        
        if not self.connection_status.get(exchange_name, {}).get('connected', False):
            self.logger.error(f"Exchange {exchange_name} nicht verbunden", "exchange")
            return False
        
        old_exchange = self.active_exchange
        self.active_exchange = exchange_name
        
        self.logger.info(f"Exchange gewechselt von {old_exchange} zu {exchange_name}", "exchange")
        return True
    
    def get_available_exchanges(self) -> List[str]:
        """Gibt alle verfügbaren und verbundenen Exchanges zurück"""
        return [name for name, status in self.connection_status.items() 
                if status.get('connected', False)]
    
    def fetch_ticker(self, symbol: str, exchange_name: str = None) -> Dict[str, Any]:
        """Holt Ticker-Daten für ein Symbol"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            ticker = self._execute_with_retry(exchange.fetch_ticker, symbol)
            self.rate_limiters[exchange_name].add_request()
            
            return {
                'symbol': ticker['symbol'],
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'high': ticker['high'],
                'low': ticker['low'],
                'timestamp': ticker['timestamp'],
                'exchange': exchange_name
            }
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen des Tickers für {symbol} auf {exchange_name}")
            raise e
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                    since: int = None, limit: int = 500, 
                    exchange_name: str = None) -> pd.DataFrame:
        """Holt OHLCV-Daten"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            ohlcv = self._execute_with_retry(
                exchange.fetch_ohlcv, 
                symbol, 
                timeframe, 
                since, 
                limit
            )
            self.rate_limiters[exchange_name].add_request()
            
            # Konvertiere zu DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen der OHLCV-Daten für {symbol} auf {exchange_name}")
            raise e
    
    def fetch_balance(self, exchange_name: str = None) -> Dict[str, Balance]:
        """Holt Kontosaldo"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            balance_data = self._execute_with_retry(exchange.fetch_balance)
            self.rate_limiters[exchange_name].add_request()
            
            balances = {}
            for asset, data in balance_data.items():
                if isinstance(data, dict) and 'free' in data:
                    balances[asset] = Balance(
                        asset=asset,
                        free=float(data.get('free', 0)),
                        used=float(data.get('used', 0)),
                        total=float(data.get('total', 0))
                    )
            
            return balances
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen des Kontosaldos auf {exchange_name}")
            raise e
    
    def create_order(self, symbol: str, side: str, amount: float, 
                    price: float = None, order_type: str = 'market',
                    params: Dict[str, Any] = None, 
                    exchange_name: str = None) -> Order:
        """Erstellt eine Order"""
        if self.emergency_stop:
            raise RuntimeError("Emergency Stop aktiv - keine neuen Orders")
        
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        params = params or {}
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            # Order erstellen
            if order_type == 'market':
                order_result = self._execute_with_retry(
                    exchange.create_market_order,
                    symbol, side, amount, None, None, params
                )
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("Preis erforderlich für Limit-Order")
                order_result = self._execute_with_retry(
                    exchange.create_limit_order,
                    symbol, side, amount, price, None, params
                )
            else:
                raise ValueError(f"Nicht unterstützter Order-Typ: {order_type}")
            
            self.rate_limiters[exchange_name].add_request()
            
            # Order-Objekt erstellen
            order = Order(
                id=order_result['id'],
                symbol=order_result['symbol'],
                side=order_result['side'],
                amount=float(order_result['amount']),
                price=float(order_result.get('price', 0)),
                type=order_result['type'],
                status=order_result['status'],
                filled=float(order_result.get('filled', 0)),
                remaining=float(order_result.get('remaining', 0)),
                cost=float(order_result.get('cost', 0)),
                fee=float(order_result.get('fee', {}).get('cost', 0)),
                timestamp=order_result.get('timestamp', int(time.time() * 1000)),
                exchange=exchange_name,
                client_order_id=order_result.get('clientOrderId', '')
            )
            
            # Trade loggen
            self.logger.log_trade(
                action=side.upper(),
                symbol=symbol,
                amount=amount,
                price=price or order.price,
                strategy="manual",  # Wird später von der Strategie überschrieben
                exchange=exchange_name,
                order_id=order.id
            )
            
            return order
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Erstellen der Order: {symbol} {side} {amount} auf {exchange_name}")
            raise e
    
    def cancel_order(self, order_id: str, symbol: str, 
                    exchange_name: str = None) -> bool:
        """Storniert eine Order"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            result = self._execute_with_retry(exchange.cancel_order, order_id, symbol)
            self.rate_limiters[exchange_name].add_request()
            
            self.logger.info(f"Order {order_id} für {symbol} auf {exchange_name} storniert", "exchange")
            return True
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Stornieren der Order {order_id} für {symbol} auf {exchange_name}")
            return False
    
    def fetch_order(self, order_id: str, symbol: str, 
                   exchange_name: str = None) -> Optional[Order]:
        """Holt Order-Status"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            order_result = self._execute_with_retry(exchange.fetch_order, order_id, symbol)
            self.rate_limiters[exchange_name].add_request()
            
            return Order(
                id=order_result['id'],
                symbol=order_result['symbol'],
                side=order_result['side'],
                amount=float(order_result['amount']),
                price=float(order_result.get('price', 0)),
                type=order_result['type'],
                status=order_result['status'],
                filled=float(order_result.get('filled', 0)),
                remaining=float(order_result.get('remaining', 0)),
                cost=float(order_result.get('cost', 0)),
                fee=float(order_result.get('fee', {}).get('cost', 0)),
                timestamp=order_result.get('timestamp', int(time.time() * 1000)),
                exchange=exchange_name,
                client_order_id=order_result.get('clientOrderId', '')
            )
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen der Order {order_id} für {symbol} auf {exchange_name}")
            return None
    
    def fetch_open_orders(self, symbol: str = None, 
                         exchange_name: str = None) -> List[Order]:
        """Holt alle offenen Orders"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            orders_result = self._execute_with_retry(exchange.fetch_open_orders, symbol)
            self.rate_limiters[exchange_name].add_request()
            
            orders = []
            for order_result in orders_result:
                orders.append(Order(
                    id=order_result['id'],
                    symbol=order_result['symbol'],
                    side=order_result['side'],
                    amount=float(order_result['amount']),
                    price=float(order_result.get('price', 0)),
                    type=order_result['type'],
                    status=order_result['status'],
                    filled=float(order_result.get('filled', 0)),
                    remaining=float(order_result.get('remaining', 0)),
                    cost=float(order_result.get('cost', 0)),
                    fee=float(order_result.get('fee', {}).get('cost', 0)),
                    timestamp=order_result.get('timestamp', int(time.time() * 1000)),
                    exchange=exchange_name,
                    client_order_id=order_result.get('clientOrderId', '')
                ))
            
            return orders
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen der offenen Orders auf {exchange_name}")
            return []
    
    def fetch_my_trades(self, symbol: str, since: int = None, 
                       limit: int = 100, exchange_name: str = None) -> List[Dict[str, Any]]:
        """Holt eigene Trades"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        # Rate Limiting
        self.rate_limiters[exchange_name].wait_if_needed()
        
        try:
            trades = self._execute_with_retry(exchange.fetch_my_trades, symbol, since, limit)
            self.rate_limiters[exchange_name].add_request()
            
            return trades
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen der Trades für {symbol} auf {exchange_name}")
            return []
    
    def get_trading_pairs(self, exchange_name: str = None) -> List[str]:
        """Holt alle verfügbaren Trading-Paare"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        try:
            markets = exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen der Trading-Paare auf {exchange_name}")
            return []
    
    def get_symbol_info(self, symbol: str, exchange_name: str = None) -> Dict[str, Any]:
        """Holt Informationen über ein Symbol"""
        exchange_name = exchange_name or self.active_exchange
        exchange = self.get_exchange(exchange_name)
        
        try:
            markets = exchange.load_markets()
            if symbol not in markets:
                raise ValueError(f"Symbol {symbol} nicht verfügbar auf {exchange_name}")
            
            market = markets[symbol]
            return {
                'symbol': market['symbol'],
                'base': market['base'],
                'quote': market['quote'],
                'active': market['active'],
                'type': market['type'],
                'spot': market.get('spot', False),
                'future': market.get('future', False),
                'option': market.get('option', False),
                'precision': {
                    'amount': market['precision']['amount'],
                    'price': market['precision']['price']
                },
                'limits': {
                    'amount': {
                        'min': market['limits']['amount']['min'],
                        'max': market['limits']['amount']['max']
                    },
                    'price': {
                        'min': market['limits']['price']['min'],
                        'max': market['limits']['price']['max']
                    },
                    'cost': {
                        'min': market['limits']['cost']['min'],
                        'max': market['limits']['cost']['max']
                    }
                },
                'maker_fee': market.get('maker', 0),
                'taker_fee': market.get('taker', 0)
            }
            
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Abrufen der Symbol-Info für {symbol} auf {exchange_name}")
            return {}
    
    def check_order_requirements(self, symbol: str, amount: float, 
                               price: float = None, exchange_name: str = None) -> Dict[str, Any]:
        """Prüft ob eine Order den Mindestanforderungen entspricht"""
        symbol_info = self.get_symbol_info(symbol, exchange_name)
        
        if not symbol_info:
            return {'valid': False, 'reason': 'Symbol-Info nicht verfügbar'}
        
        limits = symbol_info.get('limits', {})
        precision = symbol_info.get('precision', {})
        
        # Mindestmenge prüfen
        min_amount = limits.get('amount', {}).get('min', 0)
        if amount < min_amount:
            return {
                'valid': False, 
                'reason': f'Menge {amount} unter Minimum {min_amount}'
            }
        
        # Mindestkosten prüfen (falls Preis angegeben)
        if price:
            cost = amount * price
            min_cost = limits.get('cost', {}).get('min', 0)
            if cost < min_cost:
                return {
                    'valid': False,
                    'reason': f'Kosten {cost} unter Minimum {min_cost}'
                }
        
        # Precision prüfen
        amount_precision = precision.get('amount', 8)
        price_precision = precision.get('price', 8)
        
        return {
            'valid': True,
            'adjusted_amount': round(amount, amount_precision),
            'adjusted_price': round(price, price_precision) if price else None,
            'min_amount': min_amount,
            'min_cost': limits.get('cost', {}).get('min', 0),
            'fees': {
                'maker': symbol_info.get('maker_fee', 0),
                'taker': symbol_info.get('taker_fee', 0)
            }
        }
    
    def emergency_stop_all(self):
        """Notfall-Stop: Storniert alle offenen Orders"""
        self.emergency_stop = True
        self.logger.critical("EMERGENCY STOP aktiviert - alle Orders werden storniert", "exchange")
        
        for exchange_name in self.get_available_exchanges():
            try:
                open_orders = self.fetch_open_orders(exchange_name=exchange_name)
                for order in open_orders:
                    try:
                        self.cancel_order(order.id, order.symbol, exchange_name)
                        self.logger.info(f"Order {order.id} storniert (Emergency Stop)", "exchange")
                    except Exception as e:
                        self.logger.error(f"Fehler beim Stornieren von Order {order.id}: {e}", "exchange")
                        
            except Exception as e:
                self.logger.error(f"Fehler beim Emergency Stop auf {exchange_name}: {e}", "exchange")
    
    def reset_emergency_stop(self):
        """Setzt Emergency Stop zurück"""
        self.emergency_stop = False
        self.logger.info("Emergency Stop deaktiviert", "exchange")
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt den Status aller Exchanges zurück"""
        status = {
            'active_exchange': self.active_exchange,
            'emergency_stop': self.emergency_stop,
            'exchanges': {},
            'rate_limits': {}
        }
        
        for exchange_name in self.exchanges.keys():
            status['exchanges'][exchange_name] = self.connection_status.get(exchange_name, {})
            
            # Rate Limit Status
            rate_limiter = self.rate_limiters.get(exchange_name)
            if rate_limiter:
                with rate_limiter.lock:
                    status['rate_limits'][exchange_name] = {
                        'current_requests': len(rate_limiter.requests),
                        'max_requests': rate_limiter.max_requests,
                        'time_window': rate_limiter.time_window,
                        'can_make_request': rate_limiter.can_make_request()
                    }
        
        return status
    
    def reconnect_exchange(self, exchange_name: str) -> bool:
        """Versucht Verbindung zu einer Exchange wiederherzustellen"""
        try:
            if exchange_name == 'binance':
                return self._setup_binance()
            elif exchange_name == 'kucoin':
                return self._setup_kucoin()
            else:
                self.logger.error(f"Unbekannte Exchange: {exchange_name}", "exchange")
                return False
                
        except Exception as e:
            self.logger.log_error(e, f"Fehler beim Wiederherstellen der Verbindung zu {exchange_name}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Führt Gesundheitscheck für alle Exchanges durch"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': True,
            'exchanges': {}
        }
        
        for exchange_name in self.exchanges.keys():
            try:
                # Einfacher API-Test
                is_connected = self._test_connection(exchange_name)
                
                health_status['exchanges'][exchange_name] = {
                    'connected': is_connected,
                    'last_