"""
Position Manager für Trading Bot
Verwaltet alle offenen Positionen, Orders und synchronisiert nach Neustarts
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
import pickle

class PositionStatus(Enum):
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Repräsentiert eine einzelne Order"""
    order_id: str
    exchange_order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: float
    filled_amount: float
    remaining_amount: float
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    strategy: str
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    fee: float = 0.0
    average_price: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für JSON-Serialisierung"""
        data = asdict(self)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Erstellt Order aus Dictionary"""
        data['side'] = OrderSide(data['side'])
        data['order_type'] = OrderType(data['order_type'])
        data['status'] = OrderStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

@dataclass
class Position:
    """Repräsentiert eine Position"""
    position_id: str
    symbol: str
    exchange: str
    strategy: str
    side: OrderSide
    entry_price: float
    current_price: float
    amount: float
    filled_amount: float
    remaining_amount: float
    value: float
    unrealized_pnl: float
    realized_pnl: float
    status: PositionStatus
    created_at: datetime
    updated_at: datetime
    entry_orders: List[str]  # Order IDs
    exit_orders: List[str]   # Order IDs
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_loss: float = 0.0
    max_gain: float = 0.0
    
    def update_pnl(self, current_price: float):
        """Aktualisiert PnL basierend auf aktuellem Preis"""
        self.current_price = current_price
        
        if self.filled_amount > 0:
            if self.side == OrderSide.BUY:
                self.unrealized_pnl = (current_price - self.entry_price) * self.filled_amount
            else:  # SHORT
                self.unrealized_pnl = (self.entry_price - current_price) * self.filled_amount
            
            # Max Gain/Loss tracking
            if self.unrealized_pnl > self.max_gain:
                self.max_gain = self.unrealized_pnl
            if self.unrealized_pnl < self.max_loss:
                self.max_loss = self.unrealized_pnl
        
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        data = asdict(self)
        data['side'] = self.side.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Erstellt Position aus Dictionary"""
        data['side'] = OrderSide(data['side'])
        data['status'] = PositionStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

class PositionManager:
    """Hauptklasse für Position Management"""
    
    def __init__(self, config_manager, logger, exchange_connector):
        self.config = config_manager
        self.logger = logger.get_logger('trading')
        self.exchange_connector = exchange_connector
        
        # Datenstrukturen
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.active_strategies: Dict[str, List[str]] = {}  # strategy -> position_ids
        
        # Thread-Safety
        self.lock = threading.RLock()
        
        # Persistenz
        self.data_dir = Path("data/positions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Statistiken
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'win_rate': 0.0
        }
        
        # Initialisierung
        self._load_positions()
        self._load_orders()
        self._start_monitoring()
    
    def _generate_position_id(self, symbol: str, strategy: str) -> str:
        """Generiert eindeutige Position ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"{symbol}_{strategy}_{timestamp}"
    
    def _generate_order_id(self, symbol: str, side: str) -> str:
        """Generiert eindeutige Order ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"{symbol}_{side}_{timestamp}"
    
    async def create_position(self, symbol: str, strategy: str, side: OrderSide,
                            amount: float, price: float = None, 
                            stop_loss: float = None, take_profit: float = None,
                            order_type: OrderType = OrderType.MARKET) -> Optional[str]:
        """Erstellt eine neue Position"""
        
        try:
            with self.lock:
                # Position ID generieren
                position_id = self._generate_position_id(symbol, strategy)
                
                # Preis bestimmen
                if price is None:
                    ticker = await self.exchange_connector.get_ticker(symbol)
                    price = ticker['last']
                
                # Position erstellen
                position = Position(
                    position_id=position_id,
                    symbol=symbol,
                    exchange=self.exchange_connector.current_exchange,
                    strategy=strategy,
                    side=side,
                    entry_price=price,
                    current_price=price,
                    amount=amount,
                    filled_amount=0.0,
                    remaining_amount=amount,
                    value=amount * price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    status=PositionStatus.OPEN,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    entry_orders=[],
                    exit_orders=[],
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit
                )
                
                # Entry Order erstellen
                order_id = await self._create_entry_order(position, order_type)
                if order_id:
                    position.entry_orders.append(order_id)
                    self.positions[position_id] = position
                    
                    # Zu Strategie-Tracking hinzufügen
                    if strategy not in self.active_strategies:
                        self.active_strategies[strategy] = []
                    self.active_strategies[strategy].append(position_id)
                    
                    # Persistieren
                    self._save_positions()
                    
                    self.logger.info(f"Position erstellt: {position_id} | {symbol} | {side.value} | {amount} @ {price}")
                    
                    return position_id
                
                return None
                
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Position: {e}")
            return None
    
    async def _create_entry_order(self, position: Position, order_type: OrderType) -> Optional[str]:
        """Erstellt Entry Order für Position"""
        
        try:
            order_id = self._generate_order_id(position.symbol, position.side.value)
            
            # Order bei Exchange platzieren
            exchange_order = await self.exchange_connector.place_order(
                symbol=position.symbol,
                side=position.side.value.lower(),
                order_type=order_type.value.lower(),
                amount=position.amount,
                price=position.entry_price if order_type == OrderType.LIMIT else None
            )
            
            if exchange_order and 'id' in exchange_order:
                # Order-Objekt erstellen
                order = Order(
                    order_id=order_id,
                    exchange_order_id=exchange_order['id'],
                    exchange=position.exchange,
                    symbol=position.symbol,
                    side=position.side,
                    order_type=order_type,
                    amount=position.amount,
                    price=position.entry_price,
                    filled_amount=0.0,
                    remaining_amount=position.amount,
                    status=OrderStatus.PENDING,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    strategy=position.strategy
                )
                
                self.orders[order_id] = order
                self._save_orders()
                
                self.logger.info(f"Entry Order erstellt: {order_id} | {exchange_order['id']}")
                return order_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Entry Order: {e}")
            return None
    
    async def close_position(self, position_id: str, amount: float = None, 
                           order_type: OrderType = OrderType.MARKET) -> bool:
        """Schließt eine Position"""
        
        try:
            with self.lock:
                if position_id not in self.positions:
                    self.logger.warning(f"Position {position_id} nicht gefunden")
                    return False
                
                position = self.positions[position_id]
                
                if position.status != PositionStatus.OPEN:
                    self.logger.warning(f"Position {position_id} ist nicht offen")
                    return False
                
                # Menge bestimmen
                close_amount = amount if amount else position.filled_amount
                if close_amount > position.filled_amount:
                    close_amount = position.filled_amount
                
                # Gegenposition erstellen
                close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
                
                # Aktuellen Preis holen
                ticker = await self.exchange_connector.get_ticker(position.symbol)
                current_price = ticker['last']
                
                # Exit Order erstellen
                order_id = await self._create_exit_order(position, close_side, close_amount, 
                                                       current_price, order_type)
                
                if order_id:
                    position.exit_orders.append(order_id)
                    position.updated_at = datetime.now()
                    self._save_positions()
                    
                    self.logger.info(f"Position wird geschlossen: {position_id} | {close_amount} @ {current_price}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Fehler beim Schließen der Position: {e}")
            return False
    
    async def _create_exit_order(self, position: Position, side: OrderSide, 
                               amount: float, price: float, order_type: OrderType) -> Optional[str]:
        """Erstellt Exit Order für Position"""
        
        try:
            order_id = self._generate_order_id(position.symbol, side.value)
            
            # Order bei Exchange platzieren
            exchange_order = await self.exchange_connector.place_order(
                symbol=position.symbol,
                side=side.value.lower(),
                order_type=order_type.value.lower(),
                amount=amount,
                price=price if order_type == OrderType.LIMIT else None
            )
            
            if exchange_order and 'id' in exchange_order:
                # Order-Objekt erstellen
                order = Order(
                    order_id=order_id,
                    exchange_order_id=exchange_order['id'],
                    exchange=position.exchange,
                    symbol=position.symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    filled_amount=0.0,
                    remaining_amount=amount,
                    status=OrderStatus.PENDING,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    strategy=position.strategy
                )
                
                self.orders[order_id] = order
                self._save_orders()
                
                self.logger.info(f"Exit Order erstellt: {order_id} | {exchange_order['id']}")
                return order_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Exit Order: {e}")
            return None
    
    async def update_stop_loss(self, position_id: str, stop_price: float) -> bool:
        """Aktualisiert Stop Loss für Position"""
        
        try:
            with self.lock:
                if position_id not in self.positions:
                    return False
                
                position = self.positions[position_id]
                old_stop = position.stop_loss_price
                position.stop_loss_price = stop_price
                position.updated_at = datetime.now()
                
                self._save_positions()
                
                self.logger.info(f"Stop Loss aktualisiert: {position_id} | {old_stop} -> {stop_price}")
                return True
                
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren des Stop Loss: {e}")
            return False
    
    async def update_take_profit(self, position_id: str, take_profit_price: float) -> bool:
        """Aktualisiert Take Profit für Position"""
        
        try:
            with self.lock:
                if position_id not in self.positions:
                    return False
                
                position = self.positions[position_id]
                old_tp = position.take_profit_price
                position.take_profit_price = take_profit_price
                position.updated_at = datetime.now()
                
                self._save_positions()
                
                self.logger.info(f"Take Profit aktualisiert: {position_id} | {old_tp} -> {take_profit_price}")
                return True
                
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren des Take Profit: {e}")
            return False
    
    async def check_stop_loss_take_profit(self, position_id: str, current_price: float) -> bool:
        """Prüft Stop Loss und Take Profit Trigger"""
        
        try:
            with self.lock:
                if position_id not in self.positions:
                    return False
                
                position = self.positions[position_id]
                
                if position.status != PositionStatus.OPEN or position.filled_amount <= 0:
                    return False
                
                should_close = False
                reason = ""
                
                # Stop Loss prüfen
                if position.stop_loss_price:
                    if position.side == OrderSide.BUY and current_price <= position.stop_loss_price:
                        should_close = True
                        reason = f"Stop Loss triggered: {current_price} <= {position.stop_loss_price}"
                    elif position.side == OrderSide.SELL and current_price >= position.stop_loss_price:
                        should_close = True
                        reason = f"Stop Loss triggered: {current_price} >= {position.stop_loss_price}"
                
                # Take Profit prüfen
                if not should_close and position.take_profit_price:
                    if position.side == OrderSide.BUY and current_price >= position.take_profit_price:
                        should_close = True
                        reason = f"Take Profit triggered: {current_price} >= {position.take_profit_price}"
                    elif position.side == OrderSide.SELL and current_price <= position.take_profit_price:
                        should_close = True
                        reason = f"Take Profit triggered: {current_price} <= {position.take_profit_price}"
                
                if should_close:
                    self.logger.info(f"Automatisches Schließen: {position_id} | {reason}")
                    await self.close_position(position_id)
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Fehler bei Stop Loss/Take Profit Prüfung: {e}")
            return False
    
    async def sync_with_exchange(self, symbol: str = None) -> bool:
        """Synchronisiert Positionen und Orders mit Exchange"""
        
        try:
            self.logger.info("Synchronisierung mit Exchange gestartet...")
            
            # Alle aktiven Orders vom Exchange holen
            active_orders = await self.exchange_connector.get_open_orders(symbol)
            
            # Orders synchronisieren
            for exchange_order in active_orders:
                await self._sync_order(exchange_order)
            
            # Positionen basierend auf Balances synchronisieren
            if symbol:
                symbols = [symbol]
            else:
                symbols = list(set([pos.symbol for pos in self.positions.values()]))
            
            for sym in symbols:
                await self._sync_positions_for_symbol(sym)
            
            self.logger.info("Synchronisierung abgeschlossen")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler bei Exchange-Synchronisierung: {e}")
            return False
    
    async def _sync_order(self, exchange_order: Dict[str, Any]):
        """Synchronisiert einzelne Order mit Exchange-Daten"""
        
        try:
            # Suche Order in unseren Daten
            our_order = None
            for order in self.orders.values():
                if order.exchange_order_id == exchange_order['id']:
                    our_order = order
                    break
            
            if our_order:
                # Order-Status aktualisieren
                old_status = our_order.status
                our_order.filled_amount = float(exchange_order.get('filled', 0))
                our_order.remaining_amount = our_order.amount - our_order.filled_amount
                our_order.average_price = float(exchange_order.get('average', our_order.price))
                our_order.fee = float(exchange_order.get('fee', {}).get('cost', 0))
                
                # Status mapping
                if exchange_order['status'] == 'closed':
                    our_order.status = OrderStatus.FILLED
                elif exchange_order['status'] == 'canceled':
                    our_order.status = OrderStatus.CANCELLED
                elif our_order.filled_amount > 0:
                    our_order.status = OrderStatus.PARTIALLY_FILLED
                
                our_order.updated_at = datetime.now()
                
                if old_status != our_order.status:
                    self.logger.info(f"Order Status Update: {our_order.order_id} | {old_status.value} -> {our_order.status.value}")
                
                # Position aktualisieren falls Order gefüllt
                if our_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    await self._update_position_from_order(our_order)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Order-Synchronisierung: {e}")
    
    async def _sync_positions_for_symbol(self, symbol: str):
        """Synchronisiert Positionen für ein Symbol"""
        
        try:
            # Balance vom Exchange holen
            balance = await self.exchange_connector.get_balance()
            
            # Base Asset aus Symbol extrahieren (z.B. BTC aus BTCUSDT)
            base_asset = symbol.replace('USDT', '').replace('BTC', '').replace('ETH', '')
            
            if base_asset in balance and balance[base_asset]['free'] > 0:
                # Wir haben Guthaben, prüfen ob wir offene Positionen haben
                current_balance = float(balance[base_asset]['free'])
                
                # Alle offenen Positionen für dieses Symbol finden
                open_positions = [pos for pos in self.positions.values() 
                                if pos.symbol == symbol and pos.status == PositionStatus.OPEN]
                
                total_position_amount = sum(pos.filled_amount for pos in open_positions)
                
                # Diskrepanz prüfen
                if abs(current_balance - total_position_amount) > 0.0001:  # Toleranz für Rundungsfehler
                    self.logger.warning(f"Balance-Diskrepanz bei {symbol}: Exchange: {current_balance}, Tracked: {total_position_amount}")
            
        except Exception as e:
            self.logger.error(f"Fehler bei Position-Synchronisierung für {symbol}: {e}")
    
    async def _update_position_from_order(self, order: Order):
        """Aktualisiert Position basierend auf Order-Fill"""
        
        try:
            # Position finden
            position = None
            for pos in self.positions.values():
                if order.order_id in pos.entry_orders or order.order_id in pos.exit_orders:
                    position = pos
                    break
            
            if not position:
                self.logger.warning(f"Position für Order {order.order_id} nicht gefunden")
                return
            
            # Entry Order
            if order.order_id in position.entry_orders:
                old_filled = position.filled_amount
                position.filled_amount += order.filled_amount - (order.filled_amount - order.remaining_amount)
                position.remaining_amount = position.amount - position.filled_amount
                
                if position.filled_amount >= position.amount:
                    position.status = PositionStatus.OPEN
                else:
                    position.status = PositionStatus.PARTIALLY_FILLED
                
                # Durchschnittspreis aktualisieren
                if order.average_price > 0:
                    total_cost = (old_filled * position.entry_price) + (order.filled_amount * order.average_price)
                    position.entry_price = total_cost / position.filled_amount if position.filled_amount > 0 else position.entry_price
                
                position.updated_at = datetime.now()
                
                self.logger.info(f"Position Entry Update: {position.position_id} | Filled: {position.filled_amount}/{position.amount}")
            
            # Exit Order
            elif order.order_id in position.exit_orders:
                # Realized PnL berechnen
                if position.side == OrderSide.BUY:
                    realized_pnl = (order.average_price - position.entry_price) * order.filled_amount
                else:
                    realized_pnl = (position.entry_price - order.average_price) * order.filled_amount
                
                position.realized_pnl += realized_pnl - order.fee
                position.filled_amount -= order.filled_amount
                
                if position.filled_amount <= 0:
                    position.status = PositionStatus.CLOSED
                    
                    # Statistiken aktualisieren
                    self._update_trade_statistics(position)
                    
                    # Aus aktiven Strategien entfernen
                    if position.strategy in self.active_strategies:
                        if position.position_id in self.active_strategies[position.strategy]:
                            self.active_strategies[position.strategy].remove(position.position_id)
                
                position.updated_at = datetime.now()
                
                self.logger.info(f"Position Exit Update: {position.position_id} | Remaining: {position.filled_amount} | PnL: {realized_pnl:.4f}")
            
            self._save_positions()
            
        except Exception as e:
            self.logger.error(f"Fehler bei Position-Update von Order: {e}")
    
    def _update_trade_statistics(self, position: Position):
        """Aktualisiert Handelsstatistiken"""
        
        try:
            with self.lock:
                self.stats['total_trades'] += 1
                
                if position.realized_pnl > 0:
                    self.stats['winning_trades'] += 1
                    self.stats['total_profit'] += position.realized_pnl
                    
                    if position.realized_pnl > self.stats['largest_win']:
                        self.stats['largest_win'] = position.realized_pnl
                else:
                    self.stats['losing_trades'] += 1
                    self.stats['total_loss'] += abs(position.realized_pnl)
                    
                    if abs(position.realized_pnl) > self.stats['largest_loss']:
                        self.stats['largest_loss'] = abs(position.realized_pnl)
                
                # Durchschnittswerte berechnen
                if self.stats['winning_trades'] > 0:
                    self.stats['average_win'] = self.stats['total_profit'] / self.stats['winning_trades']
                
                if self.stats['losing_trades'] > 0:
                    self.stats['average_loss'] = self.stats['total_loss'] / self.stats['losing_trades']
                
                # Win Rate
                if self.stats['total_trades'] > 0:
                    self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
                
                # Statistiken persistieren
                self._save_statistics()
                
        except Exception as e:
            self.logger.error(f"Fehler bei Statistik-Update: {e}")
    
    def get_positions(self, symbol: str = None, strategy: str = None, 
                     status: PositionStatus = None) -> List[Position]:
        """Gibt Positionen basierend auf Filtern zurück"""
        
        with self.lock:
            positions = list(self.positions.values())
            
            if symbol:
                positions = [pos for pos in positions if pos.symbol == symbol]
            
            if strategy:
                positions = [pos for pos in positions if pos.strategy == strategy]
            
            if status:
                positions = [pos for pos in positions if pos.status == status]
            
            return positions
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Gibt einzelne Position zurück"""
        with self.lock:
            return self.positions.get(position_id)
    
    def get_open_positions(self, symbol: str = None) -> List[Position]:
        """Gibt alle offenen Positionen zurück"""
        return self.get_positions(symbol=symbol, status=PositionStatus.OPEN)
    
    def get_orders(self, symbol: str = None, status: OrderStatus = None) -> List[Order]:
        """Gibt Orders basierend auf Filtern zurück"""
        
        with self.lock:
            orders = list(self.orders.values())
            
            if symbol:
                orders = [order for order in orders if order.symbol == symbol]
            
            if status:
                orders = [order for order in orders if order.status == status]
            
            return orders
    
    def get_pending_orders(self, symbol: str = None) -> List[Order]:
        """Gibt alle pending Orders zurück"""
        return self.get_orders(symbol=symbol, status=OrderStatus.PENDING)
    
    def get_portfolio_value(self) -> Dict[str, float]:
        """Berechnet aktuellen Portfolio-Wert"""
        
        try:
            with self.lock:
                total_value = 0.0
                unrealized_pnl = 0.0
                realized_pnl = 0.0
                
                for position in self.positions.values():
                    if position.status == PositionStatus.OPEN:
                        total_value += position.value
                        unrealized_pnl += position.unrealized_pnl
                    
                    realized_pnl += position.realized_pnl
                
                return {
                    'total_value': total_value,
                    'unrealized_pnl': unrealized_pnl,
                    'realized_pnl': realized_pnl,
                    'total_pnl': unrealized_pnl + realized_pnl
                }
                
	 except Exception as e:
            self.logger.error(f"Fehler bei Portfolio-Wert Berechnung: {e}")
            return {
                'total_value': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_pnl': 0.0
            }
    
    def get_strategy_performance(self, strategy: str) -> Dict[str, Any]:
        """Gibt Performance-Daten für eine Strategie zurück"""
        
        try:
            with self.lock:
                strategy_positions = [pos for pos in self.positions.values() 
                                    if pos.strategy == strategy]
                
                if not strategy_positions:
                    return {'error': f'Keine Positionen für Strategie {strategy} gefunden'}
                
                total_trades = len([pos for pos in strategy_positions 
                                  if pos.status == PositionStatus.CLOSED])
                winning_trades = len([pos for pos in strategy_positions 
                                    if pos.status == PositionStatus.CLOSED and pos.realized_pnl > 0])
                losing_trades = len([pos for pos in strategy_positions 
                                   if pos.status == PositionStatus.CLOSED and pos.realized_pnl < 0])
                
                total_profit = sum(pos.realized_pnl for pos in strategy_positions 
                                 if pos.status == PositionStatus.CLOSED and pos.realized_pnl > 0)
                total_loss = sum(abs(pos.realized_pnl) for pos in strategy_positions 
                               if pos.status == PositionStatus.CLOSED and pos.realized_pnl < 0)
                
                unrealized_pnl = sum(pos.unrealized_pnl for pos in strategy_positions 
                                   if pos.status == PositionStatus.OPEN)
                
                return {
                    'strategy': strategy,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'total_profit': total_profit,
                    'total_loss': total_loss,
                    'net_profit': total_profit - total_loss,
                    'unrealized_pnl': unrealized_pnl,
                    'profit_factor': (total_profit / total_loss) if total_loss > 0 else float('inf'),
                    'active_positions': len([pos for pos in strategy_positions 
                                           if pos.status == PositionStatus.OPEN])
                }
                
        except Exception as e:
            self.logger.error(f"Fehler bei Strategie-Performance Berechnung: {e}")
            return {'error': f'Fehler bei Berechnung: {str(e)}'}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt aktuelle Handelsstatistiken zurück"""
        with self.lock:
            return self.stats.copy()
    
    def _start_monitoring(self):
        """Startet Monitoring-Thread"""
        
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitoring_thread.start()
                self.logger.info("Position Monitoring gestartet")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Monitoring: {e}")
    
    def _monitoring_loop(self):
        """Hauptschleife für Position-Monitoring"""
        
        while self.monitoring_active:
            try:
                # Async-Event-Loop in Thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Monitoring-Tasks ausführen
                loop.run_until_complete(self._run_monitoring_tasks())
                
                # Kurz warten
                time.sleep(5)  # 5 Sekunden zwischen Checks
                
            except Exception as e:
                self.logger.error(f"Fehler im Monitoring-Loop: {e}")
                time.sleep(10)  # Länger warten bei Fehlern
    
    async def _run_monitoring_tasks(self):
        """Führt alle Monitoring-Aufgaben aus"""
        
        try:
            # Aktuelle Preise für alle offenen Positionen holen
            open_positions = self.get_open_positions()
            
            if not open_positions:
                return
            
            # Eindeutige Symbole sammeln
            symbols = list(set([pos.symbol for pos in open_positions]))
            
            # Preise für alle Symbole holen
            for symbol in symbols:
                try:
                    ticker = await self.exchange_connector.get_ticker(symbol)
                    current_price = ticker['last']
                    
                    # Alle Positionen für dieses Symbol aktualisieren
                    symbol_positions = [pos for pos in open_positions if pos.symbol == symbol]
                    
                    for position in symbol_positions:
                        # PnL aktualisieren
                        position.update_pnl(current_price)
                        
                        # Stop Loss / Take Profit prüfen
                        await self.check_stop_loss_take_profit(position.position_id, current_price)
                    
                except Exception as e:
                    self.logger.error(f"Fehler beim Aktualisieren von {symbol}: {e}")
            
            # Positionen speichern
            self._save_positions()
            
            # Regelmäßige Exchange-Synchronisierung (alle 30 Sekunden)
            if int(time.time()) % 30 == 0:
                await self.sync_with_exchange()
                
        except Exception as e:
            self.logger.error(f"Fehler bei Monitoring-Tasks: {e}")
    
    def stop_monitoring(self):
        """Stoppt das Monitoring"""
        
        try:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            self.logger.info("Position Monitoring gestoppt")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Monitoring: {e}")
    
    def _save_positions(self):
        """Speichert Positionen in Datei"""
        
        try:
            positions_file = self.data_dir / "positions.json"
            positions_data = {
                pos_id: pos.to_dict() for pos_id, pos in self.positions.items()
            }
            
            with open(positions_file, 'w', encoding='utf-8') as f:
                json.dump(positions_data, f, indent=2, ensure_ascii=False)
                
            # Backup erstellen
            backup_file = self.data_dir / f"positions_backup_{int(time.time())}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(positions_data, f, indent=2, ensure_ascii=False)
            
            # Alte Backups löschen (nur letzten 10 behalten)
            self._cleanup_old_backups("positions_backup_")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Positionen: {e}")
    
    def _load_positions(self):
        """Lädt Positionen aus Datei"""
        
        try:
            positions_file = self.data_dir / "positions.json"
            
            if positions_file.exists():
                with open(positions_file, 'r', encoding='utf-8') as f:
                    positions_data = json.load(f)
                
                for pos_id, pos_data in positions_data.items():
                    try:
                        position = Position.from_dict(pos_data)
                        self.positions[pos_id] = position
                        
                        # Zu Strategie-Tracking hinzufügen
                        if position.strategy not in self.active_strategies:
                            self.active_strategies[position.strategy] = []
                        if pos_id not in self.active_strategies[position.strategy]:
                            self.active_strategies[position.strategy].append(pos_id)
                            
                    except Exception as e:
                        self.logger.error(f"Fehler beim Laden der Position {pos_id}: {e}")
                
                self.logger.info(f"{len(self.positions)} Positionen geladen")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Positionen: {e}")
    
    def _save_orders(self):
        """Speichert Orders in Datei"""
        
        try:
            orders_file = self.data_dir / "orders.json"
            orders_data = {
                order_id: order.to_dict() for order_id, order in self.orders.items()
            }
            
            with open(orders_file, 'w', encoding='utf-8') as f:
                json.dump(orders_data, f, indent=2, ensure_ascii=False)
                
            # Backup erstellen
            backup_file = self.data_dir / f"orders_backup_{int(time.time())}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(orders_data, f, indent=2, ensure_ascii=False)
            
            # Alte Backups löschen
            self._cleanup_old_backups("orders_backup_")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Orders: {e}")
    
    def _load_orders(self):
        """Lädt Orders aus Datei"""
        
        try:
            orders_file = self.data_dir / "orders.json"
            
            if orders_file.exists():
                with open(orders_file, 'r', encoding='utf-8') as f:
                    orders_data = json.load(f)
                
                for order_id, order_data in orders_data.items():
                    try:
                        order = Order.from_dict(order_data)
                        self.orders[order_id] = order
                    except Exception as e:
                        self.logger.error(f"Fehler beim Laden der Order {order_id}: {e}")
                
                self.logger.info(f"{len(self.orders)} Orders geladen")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Orders: {e}")
    
    def _save_statistics(self):
        """Speichert Statistiken in Datei"""
        
        try:
            stats_file = self.data_dir / "statistics.json"
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Statistiken: {e}")
    
    def _load_statistics(self):
        """Lädt Statistiken aus Datei"""
        
        try:
            stats_file = self.data_dir / "statistics.json"
            
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    saved_stats = json.load(f)
                    self.stats.update(saved_stats)
                
                self.logger.info("Statistiken geladen")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Statistiken: {e}")
    
    def _cleanup_old_backups(self, prefix: str, keep_count: int = 10):
        """Löscht alte Backup-Dateien"""
        
        try:
            backup_files = list(self.data_dir.glob(f"{prefix}*.json"))
            
            if len(backup_files) > keep_count:
                # Nach Erstellungszeit sortieren (älteste zuerst)
                backup_files.sort(key=lambda x: x.stat().st_mtime)
                
                # Überschüssige Dateien löschen
                for file_to_delete in backup_files[:-keep_count]:
                    file_to_delete.unlink()
                    
        except Exception as e:
            self.logger.error(f"Fehler beim Aufräumen der Backups: {e}")
    
    async def emergency_close_all_positions(self, reason: str = "Emergency Close") -> bool:
        """Schließt alle offenen Positionen im Notfall"""
        
        try:
            open_positions = self.get_open_positions()
            
            if not open_positions:
                self.logger.info("Keine offenen Positionen für Notfall-Schließung")
                return True
            
            self.logger.warning(f"NOTFALL-SCHLIESSUNG ALLER POSITIONEN: {reason}")
            
            success_count = 0
            for position in open_positions:
                try:
                    success = await self.close_position(position.position_id)
                    if success:
                        success_count += 1
                        self.logger.info(f"Position {position.position_id} notfallmäßig geschlossen")
                    else:
                        self.logger.error(f"Fehler beim Schließen von Position {position.position_id}")
                        
                except Exception as e:
                    self.logger.error(f"Fehler beim Notfall-Schließen von {position.position_id}: {e}")
            
            self.logger.info(f"Notfall-Schließung abgeschlossen: {success_count}/{len(open_positions)} Positionen geschlossen")
            
            return success_count == len(open_positions)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Notfall-Schließung: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """Storniert alle pending Orders"""
        
        try:
            pending_orders = self.get_pending_orders(symbol)
            
            if not pending_orders:
                self.logger.info("Keine pending Orders zum Stornieren")
                return True
            
            success_count = 0
            for order in pending_orders:
                try:
                    # Order bei Exchange stornieren
                    result = await self.exchange_connector.cancel_order(
                        order.exchange_order_id, order.symbol
                    )
                    
                    if result:
                        order.status = OrderStatus.CANCELLED
                        order.updated_at = datetime.now()
                        success_count += 1
                        self.logger.info(f"Order {order.order_id} storniert")
                    
                except Exception as e:
                    self.logger.error(f"Fehler beim Stornieren von Order {order.order_id}: {e}")
            
            self._save_orders()
            
            self.logger.info(f"Order-Stornierung abgeschlossen: {success_count}/{len(pending_orders)} Orders storniert")
            
            return success_count == len(pending_orders)
            
        except Exception as e:
            self.logger.error(f"Fehler bei Order-Stornierung: {e}")
            return False
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung aller Positionen zurück"""
        
        try:
            with self.lock:
                open_positions = self.get_open_positions()
                closed_positions = self.get_positions(status=PositionStatus.CLOSED)
                
                # Nach Strategien gruppieren
                strategy_summary = {}
                for strategy in self.active_strategies.keys():
                    strategy_perf = self.get_strategy_performance(strategy)
                    strategy_summary[strategy] = strategy_perf
                
                # Portfolio-Werte
                portfolio = self.get_portfolio_value()
                
                return {
                    'total_positions': len(self.positions),
                    'open_positions': len(open_positions),
                    'closed_positions': len(closed_positions),
                    'active_strategies': len(self.active_strategies),
                    'portfolio_value': portfolio,
                    'strategy_performance': strategy_summary,
                    'overall_statistics': self.stats,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Fehler bei Position-Zusammenfassung: {e}")
            return {'error': f'Fehler bei Zusammenfassung: {str(e)}'}
    
    def export_positions_to_csv(self, filename: str = None) -> str:
        """Exportiert Positionen zu CSV-Datei"""
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"positions_export_{timestamp}.csv"
            
            export_file = self.data_dir / filename
            
            import csv
            
            with open(export_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'position_id', 'symbol', 'exchange', 'strategy', 'side', 
                    'entry_price', 'current_price', 'amount', 'filled_amount',
                    'value', 'unrealized_pnl', 'realized_pnl', 'status',
                    'created_at', 'updated_at', 'stop_loss_price', 'take_profit_price',
                    'max_gain', 'max_loss'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for position in self.positions.values():
                    row = {
                        'position_id': position.position_id,
                        'symbol': position.symbol,
                        'exchange': position.exchange,
                        'strategy': position.strategy,
                        'side': position.side.value,
                        'entry_price': position.entry_price,
                        'current_price': position.current_price,
                        'amount': position.amount,
                        'filled_amount': position.filled_amount,
                        'value': position.value,
                        'unrealized_pnl': position.unrealized_pnl,
                        'realized_pnl': position.realized_pnl,
                        'status': position.status.value,
                        'created_at': position.created_at.isoformat(),
                        'updated_at': position.updated_at.isoformat(),
                        'stop_loss_price': position.stop_loss_price,
                        'take_profit_price': position.take_profit_price,
                        'max_gain': position.max_gain,
                        'max_loss': position.max_loss
                    }
                    writer.writerow(row)
            
            self.logger.info(f"Positionen exportiert nach: {export_file}")
            return str(export_file)
            
        except Exception as e:
            self.logger.error(f"Fehler beim CSV-Export: {e}")
            return ""
    
    def __del__(self):
        """Destruktor - Monitoring stoppen"""
        try:
            self.stop_monitoring()
        except:
            pass


# Utility-Funktionen für Position Manager

def calculate_position_size(balance: float, risk_percent: float, 
                          entry_price: float, stop_loss_price: float) -> float:
    """
    Berechnet optimale Positionsgröße basierend auf Risikomanagement
    
    Args:
        balance: Verfügbares Guthaben
        risk_percent: Risiko in Prozent (z.B. 2.0 für 2%)
        entry_price: Einstiegspreis
        stop_loss_price: Stop-Loss Preis
    
    Returns:
        Positionsgröße
    """
    try:
        risk_amount = balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        position_size = risk_amount / price_diff
        return round(position_size, 8)
        
    except Exception:
        return 0

def calculate_stop_loss_price(entry_price: float, side: OrderSide, 
                            stop_loss_percent: float) -> float:
    """
    Berechnet Stop-Loss Preis basierend auf Prozentsatz
    
    Args:
        entry_price: Einstiegspreis
        side: BUY oder SELL
        stop_loss_percent: Stop-Loss in Prozent
    
    Returns:
        Stop-Loss Preis
    """
    try:
        if side == OrderSide.BUY:
            return entry_price * (1 - stop_loss_percent / 100)
        else:  # SELL
            return entry_price * (1 + stop_loss_percent / 100)
            
    except Exception:
        return entry_price

def calculate_take_profit_price(entry_price: float, side: OrderSide, 
                              take_profit_percent: float) -> float:
    """
    Berechnet Take-Profit Preis basierend auf Prozentsatz
    
    Args:
        entry_price: Einstiegspreis
        side: BUY oder SELL
        take_profit_percent: Take-Profit in Prozent
    
    Returns:
        Take-Profit Preis
    """
    try:
        if side == OrderSide.BUY:
            return entry_price * (1 + take_profit_percent / 100)
        else:  # SELL
            return entry_price * (1 - take_profit_percent / 100)
            
    except Exception:
        return entry_price