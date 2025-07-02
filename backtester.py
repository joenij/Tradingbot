import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, asdict
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestConfig:
    """Konfiguration für Backtests"""
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    commission: float = 0.001  # 0.1% Gebühren
    slippage: float = 0.001    # 0.1% Slippage
    max_positions: int = 1
    position_size_pct: float = 0.95  # 95% des verfügbaren Kapitals
    risk_per_trade: float = 0.02     # 2% Risiko pro Trade
    
@dataclass
class Trade:
    """Einzelner Trade"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    side: str = 'buy'  # 'buy' oder 'sell'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    strategy: str = ''
    trade_id: str = ''
    
@dataclass
class BacktestResult:
    """Backtest Ergebnisse"""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: List[float]
    timestamps: List[datetime]
    total_return: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    start_date: datetime = None
    end_date: datetime = None
    duration_days: int = 0
    annual_return: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    strategy_name: str = ''

class PortfolioState:
    """Portfolio Status während Backtest"""
    
    def __init__(self, initial_balance: float):
        self.cash = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.equity_history = [initial_balance]
        self.timestamp_history = []
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Berechnet aktuellen Portfoliowert"""
        position_value = sum(
            qty * prices.get(symbol, 0) 
            for symbol, qty in self.positions.items()
        )
        return self.cash + position_value
        
    def can_buy(self, symbol: str, price: float, quantity: float, commission: float) -> bool:
        """Prüft ob Kauf möglich ist"""
        required_cash = quantity * price * (1 + commission)
        return self.cash >= required_cash
        
    def execute_buy(self, symbol: str, price: float, quantity: float, 
                   commission: float, timestamp: datetime, strategy: str) -> Trade:
        """Führt Kauf aus"""
        cost = quantity * price
        total_cost = cost * (1 + commission)
        
        if not self.can_buy(symbol, price, quantity, commission):
            raise ValueError("Nicht genügend Cash für Kauf")
            
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            quantity=quantity,
            side='buy',
            commission=cost * commission,
            strategy=strategy,
            trade_id=f"{strategy}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        
        return trade
        
    def execute_sell(self, symbol: str, price: float, quantity: float, 
                    commission: float, timestamp: datetime, trade: Trade) -> Trade:
        """Führt Verkauf aus und schließt Trade ab"""
        if self.positions.get(symbol, 0) < quantity:
            raise ValueError("Nicht genügend Position für Verkauf")
            
        proceeds = quantity * price
        net_proceeds = proceeds * (1 - commission)
        
        self.cash += net_proceeds
        self.positions[symbol] -= quantity
        
        if self.positions[symbol] <= 0:
            del self.positions[symbol]
            
        # Trade abschließen
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.commission += proceeds * commission
        
        # PnL berechnen
        if trade.side == 'buy':
            trade.pnl = (price - trade.entry_price) * quantity - trade.commission
        else:
            trade.pnl = (trade.entry_price - price) * quantity - trade.commission
            
        trade.pnl_pct = trade.pnl / (trade.entry_price * quantity) * 100
        
        self.trades.append(trade)
        return trade

class BacktestEngine:
    """Haupt-Backtesting Engine"""
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger.get_logger(__name__)
        self.results_cache = {}
        
    def run_backtest(self, strategy_class, data: pd.DataFrame, 
                    config: BacktestConfig) -> BacktestResult:
        """Führt Backtest für eine Strategie aus"""
        try:
            self.logger.info(f"Starte Backtest für {strategy_class.__name__} auf {config.symbol}")
            
            # Portfolio initialisieren
            portfolio = PortfolioState(config.initial_balance)
            
            # Strategie initialisieren
            strategy = strategy_class(self.config_manager, self.logger)
            
            # Daten vorbereiten
            data = data.copy()
            data.index = pd.to_datetime(data.index)
            
            # Filter für Zeitraum
            start_date = pd.to_datetime(config.start_date)
            end_date = pd.to_datetime(config.end_date)
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if data.empty:
                raise ValueError("Keine Daten für angegebenen Zeitraum")
            
            open_trades = []  # Offene Trades
            
            # Durch alle Datenpunkte iterieren
            for i, (timestamp, row) in enumerate(data.iterrows()):
                current_data = data.iloc[:i+1] if i > 0 else data.iloc[:1]
                
                # Aktuelle Preise
                prices = {config.symbol: row['close']}
                
                # Portfolio Wert berechnen
                portfolio_value = portfolio.get_portfolio_value(prices)
                portfolio.equity_history.append(portfolio_value)
                portfolio.timestamp_history.append(timestamp)
                
                # Strategie analysieren
                try:
                    signals = strategy.analyze(current_data)
                    
                    if not signals:
                        continue
                        
                    signal = signals.get(config.symbol, {})
                    action = signal.get('action', 'hold')
                    
                    if action == 'buy' and len(open_trades) < config.max_positions:
                        # Kaufsignal
                        available_cash = portfolio.cash
                        position_size = available_cash * config.position_size_pct
                        quantity = position_size / row['close']
                        
                        if portfolio.can_buy(config.symbol, row['close'], quantity, config.commission):
                            trade = portfolio.execute_buy(
                                config.symbol, row['close'], quantity,
                                config.commission, timestamp, strategy_class.__name__
                            )
                            open_trades.append(trade)
                            
                            self.logger.debug(f"Kauf: {quantity:.4f} {config.symbol} @ {row['close']:.4f}")
                    
                    elif action == 'sell' and open_trades:
                        # Verkaufssignal - ältesten Trade schließen
                        trade_to_close = open_trades.pop(0)
                        
                        completed_trade = portfolio.execute_sell(
                            config.symbol, row['close'], trade_to_close.quantity,
                            config.commission, timestamp, trade_to_close
                        )
                        
                        self.logger.debug(f"Verkauf: {completed_trade.quantity:.4f} {config.symbol} @ {row['close']:.4f}, PnL: {completed_trade.pnl:.2f}")
                        
                except Exception as e:
                    self.logger.error(f"Fehler bei Strategie-Analyse: {e}")
                    continue
            
            # Offene Trades am Ende schließen
            final_price = data.iloc[-1]['close']
            final_timestamp = data.index[-1]
            
            for trade in open_trades:
                completed_trade = portfolio.execute_sell(
                    config.symbol, final_price, trade.quantity,
                    config.commission, final_timestamp, trade
                )
                self.logger.debug(f"Trade am Ende geschlossen: PnL: {completed_trade.pnl:.2f}")
            
            # Finaler Portfolio-Wert
            final_value = portfolio.get_portfolio_value({config.symbol: final_price})
            portfolio.equity_history.append(final_value)
            portfolio.timestamp_history.append(final_timestamp)
            
            # Ergebnisse berechnen
            result = self._calculate_results(portfolio, config, strategy_class.__name__)
            
            self.logger.info(f"Backtest abgeschlossen. Return: {result.total_return_pct:.2f}%, "
                           f"Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.total_trades}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler beim Backtest: {e}")
            raise
    
    def _calculate_results(self, portfolio: PortfolioState, 
                         config: BacktestConfig, strategy_name: str) -> BacktestResult:
        """Berechnet Backtest-Ergebnisse"""
        
        # Basis-Metriken
        initial_value = config.initial_balance
        final_value = portfolio.equity_history[-1]
        total_return = final_value - initial_value
        total_return_pct = (total_return / initial_value) * 100
        
        # Trade-Statistiken
        trades = portfolio.trades
        total_trades = len(trades)
        
        if total_trades == 0:
            return BacktestResult(
                config=config,
                trades=[],
                equity_curve=portfolio.equity_history,
                timestamps=portfolio.timestamp_history,
                strategy_name=strategy_name
            )
        
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # PnL Statistiken
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Equity Curve Analyse
        equity_series = pd.Series(portfolio.equity_history, 
                                index=portfolio.timestamp_history)
        returns = equity_series.pct_change().dropna()
        
        # Drawdown berechnen
        running_max = equity_series.expanding().max()
        drawdown = equity_series - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max[drawdown.idxmin()]) * 100
        
        # Risiko-Metriken
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252) * 100  # Annualisiert
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Zeitraum-Metriken
        start_date = portfolio.timestamp_history[0]
        end_date = portfolio.timestamp_history[-1]
        duration_days = (end_date - start_date).days
        
        # Annualisierte Rendite
        if duration_days > 0:
            annual_return = ((final_value / initial_value) ** (365.25 / duration_days) - 1) * 100
        else:
            annual_return = 0
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
        
        return BacktestResult(
            config=config,
            trades=trades,
            equity_curve=portfolio.equity_history,
            timestamps=portfolio.timestamp_history,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            annual_return=annual_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            strategy_name=strategy_name
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Berechnet Sharpe Ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)  # Täglicher Risk-free Rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Berechnet Sortino Ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0
        
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    def run_multiple_backtests(self, strategies: List, data_dict: Dict[str, pd.DataFrame],
                             configs: List[BacktestConfig]) -> Dict[str, List[BacktestResult]]:
        """Führt mehrere Backtests parallel aus"""
        results = {}
        
        # Alle Kombinationen erstellen
        tasks = []
        for strategy_class in strategies:
            for config in configs:
                if config.symbol in data_dict:
                    tasks.append((strategy_class, data_dict[config.symbol], config))
        
        self.logger.info(f"Starte {len(tasks)} Backtests parallel")
        
        # Parallel ausführen
        with ThreadPoolExecutor(max_workers=min(len(tasks), multiprocessing.cpu_count())) as executor:
            futures = []
            for strategy_class, data, config in tasks:
                future = executor.submit(self.run_backtest, strategy_class, data, config)
                futures.append((future, strategy_class.__name__, config.symbol))
            
            for future, strategy_name, symbol in futures:
                try:
                    result = future.result()
                    key = f"{strategy_name}_{symbol}"
                    if key not in results:
                        results[key] = []
                    results[key].append(result)
                except Exception as e:
                    self.logger.error(f"Backtest fehlgeschlagen für {strategy_name} auf {symbol}: {e}")
        
        return results
    
    def optimize_parameters(self, strategy_class, data: pd.DataFrame, 
                          base_config: BacktestConfig, 
                          param_ranges: Dict[str, List]) -> Tuple[Dict, BacktestResult]:
        """Optimiert Strategie-Parameter"""
        self.logger.info(f"Starte Parameter-Optimierung für {strategy_class.__name__}")
        
        best_result = None
        best_params = {}
        
        # Alle Parameter-Kombinationen generieren
        param_combinations = self._generate_param_combinations(param_ranges)
        
        self.logger.info(f"Teste {len(param_combinations)} Parameter-Kombinationen")
        
        for i, params in enumerate(param_combinations):
            try:
                # Strategie mit neuen Parametern erstellen
                strategy = strategy_class(self.config_manager, self.logger)
                
                # Parameter setzen
                for param_name, param_value in params.items():
                    setattr(strategy, param_name, param_value)
                
                # Backtest durchführen
                result = self.run_backtest(strategy_class, data, base_config)
                
                # Bewertung (hier: Sharpe Ratio, kann angepasst werden)
                if best_result is None or result.sharpe_ratio > best_result.sharpe_ratio:
                    best_result = result
                    best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Progress: {i+1}/{len(param_combinations)} - "
                                   f"Beste Sharpe Ratio: {best_result.sharpe_ratio:.4f}")
                    
            except Exception as e:
                self.logger.error(f"Fehler bei Parameter-Kombination {params}: {e}")
                continue
        
        self.logger.info(f"Optimierung abgeschlossen. Beste Parameter: {best_params}")
        return best_params, best_result
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generiert alle Parameter-Kombinationen"""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
            
        return combinations
    
    def generate_report(self, results: Dict[str, List[BacktestResult]], 
                       output_dir: str = "backtest_results") -> str:
        """Generiert detaillierten Backtest-Report"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(output_dir, f"backtest_report_{timestamp}.html")
            
            html_content = self._generate_html_report(results)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Zusätzlich JSON-Export
            json_file = os.path.join(output_dir, f"backtest_data_{timestamp}.json")
            self._export_results_json(results, json_file)
            
            self.logger.info(f"Report generiert: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Fehler beim Generieren des Reports: {e}")
            raise
    
    def _generate_html_report(self, results: Dict[str, List[BacktestResult]]) -> str:
        """Generiert HTML-Report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                th { background-color: #f2f2f2; }
                .positive { color: green; }
                .negative { color: red; }
                .summary { background-color: #f9f9f9; padding: 15px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Trading Bot Backtest Report</h1>
            <div class="summary">
                <h2>Zusammenfassung</h2>
        """
        
        # Zusammenfassung erstellen
        all_results = [result for result_list in results.values() for result in result_list]
        
        if all_results:
            avg_return = np.mean([r.total_return_pct for r in all_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in all_results])
            best_strategy = max(all_results, key=lambda x: x.total_return_pct)
            
            html += f"""
                <p>Anzahl getesteter Strategien: {len(results)}</p>
                <p>Durchschnittliche Rendite: {avg_return:.2f}%</p>
                <p>Durchschnittliche Sharpe Ratio: {avg_sharpe:.2f}</p>
                <p>Beste Strategie: {best_strategy.strategy_name} ({best_strategy.total_return_pct:.2f}%)</p>
            </div>
            """
        
        # Detailtabelle
        html += """
            <h2>Detaillierte Ergebnisse</h2>
            <table>
                <tr>
                    <th>Strategie</th>
                    <th>Symbol</th>
                    <th>Zeitraum</th>
                    <th>Rendite (%)</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown (%)</th>
                    <th>Win Rate (%)</th>
                    <th>Trades</th>
                    <th>Profit Factor</th>
                </tr>
        """
        
        for key, result_list in results.items():
            for result in result_list:
                color_class = "positive" if result.total_return_pct > 0 else "negative"
                html += f"""
                    <tr>
                        <td>{result.strategy_name}</td>
                        <td>{result.config.symbol}</td>
                        <td>{result.start_date.strftime('%Y-%m-%d')} - {result.end_date.strftime('%Y-%m-%d')}</td>
                        <td class="{color_class}">{result.total_return_pct:.2f}</td>
                        <td>{result.sharpe_ratio:.2f}</td>
                        <td class="negative">{result.max_drawdown_pct:.2f}</td>
                        <td>{result.win_rate:.1f}</td>
                        <td>{result.total_trades}</td>
                        <td>{result.profit_factor:.2f}</td>
                    </tr>
                """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def _export_results_json(self, results: Dict[str, List[BacktestResult]], 
                           filename: str) -> None:
        """Exportiert Ergebnisse als JSON"""
        export_data = {}
        
        for key, result_list in results.items():
            export_data[key] = []
            for result in result_list:
                # BacktestResult zu Dict konvertieren
                result_dict = asdict(result)
                
                # Datetime-Objekte zu Strings konvertieren
                if result_dict['start_date']:
                    result_dict['start_date'] = result_dict['start_date'].isoformat()
                if result_dict['end_date']:
                    result_dict['end_date'] = result_dict['end_date'].isoformat()
                
                # Timestamps konvertieren
                result_dict['timestamps'] = [ts.isoformat() for ts in result.timestamps]
                
                # Trade-Timestamps konvertieren
                for trade_dict in result_dict['trades']:
                    if trade_dict['entry_time']:
                        trade_dict['entry_time'] = trade_dict['entry_time'].isoformat()
                    if trade_dict['exit_time']:
                        trade_dict['exit_time'] = trade_dict['exit_time'].isoformat()
                
                export_data[key].append(result_dict)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def compare_strategies(self, results: Dict[str, List[BacktestResult]]) -> pd.DataFrame:
        """Erstellt Vergleichstabelle für Strategien"""
        comparison_data = []
        
        for key, result_list in results.items():
            for result in result_list:
                comparison_data.append({
                    'Strategy': result.strategy_name,
                    'Symbol': result.config.symbol,
                    'Return_Pct': result.total_return_pct,
                    'Sharpe_Ratio': result.sharpe_ratio,
                    'Max_Drawdown_Pct': result.max_drawdown_pct,
                    'Win_Rate': result.win_rate,
                    'Total_Trades': result.total_trades,
                    'Profit_Factor': result.profit_factor,
                    'Annual_Return': result.annual_return,
                    'Volatility': result.volatility,
                    'Calmar_Ratio': result.calmar_ratio,
                    'Sortino_Ratio': result.sortino_ratio
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Nach Sharpe Ratio sortieren
        df = df.sort_values('Sharpe_Ratio', ascending=False)
        
        return df
    
    def save_results(self, results: Dict[str, List[BacktestResult]], 
                    filename: str = None) -> str:
        """Speichert Backtest-Ergebnisse"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
        
        try:
            self._export_results_json(results, filename)
            self.logger.info(f"Ergebnisse gespeichert: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern: {e}")
            raise
    
    def load_results(self, filename: str) -> Dict[str, List[BacktestResult]]:
        """Lädt gespeicherte Backtest-Ergebnisse"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = {}
            for key, result_list in data.items():
                results[key] = []
                for result_dict in result_list:
                    # Datetime-Strings zurück konvertieren
                    if result_dict['start_date']:
                        result_dict['start_date'] = datetime.fromisoformat(result_dict['start_date'])
                    if result_dict['end_date']:
                        result_dict['end_date'] = datetime.fromisoformat(result_dict['end_date'])
                    
                    result_dict['timestamps'] = [
                        datetime.fromisoformat(ts) for ts in result_dict['timestamps']
                    ]
                    
                    # Trades konvertieren
                    trades = []
                    for trade_dict in result_dict['trades']:
                        if trade_dict['entry_time']:
                            trade_dict['entry_time'] = datetime.fromisoformat(trade_dict['entry_time'])
                        if trade_dict['exit_time']:
                            trade_dict['exit_time'] = datetime.fromisoformat(trade_dict['exit_time'])
                        trades.append(Trade(**trade_dict))
                    
                    result_dict['trades'] = trades
                    
                    # Config rekonstruieren
                    config_dict = result_dict['config']
                    result_dict['config'] = BacktestConfig(**config_dict)
                    
                    results[key].append(BacktestResult(**result_dict))
            
            self.logger.info(f"Ergebnisse geladen: {filename}")
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden: {e}")
            raise

# Erweiterte Analyse-Klassen

class PerformanceAnalyzer:
    """Erweiterte Performance-Analyse"""
    
    def __init__(self, logger):
        self.logger = logger.get_logger(__name__)
    
    def analyze_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Detaillierte Trade-Analyse"""
        if not trades:
            return {}
        
        # Trade-Dauer Analyse
        trade_durations = []
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # Stunden
                trade_durations.append(duration)
        
        # PnL Verteilung
        pnls = [trade.pnl for trade in trades]
        pnl_pcts = [trade.pnl_pct for trade in trades]
        
        # Consecutive Wins/Losses
        consecutive_stats = self._analyze_consecutive_trades(trades)
        
        # Monthly Performance
        monthly_performance = self._analyze_monthly_performance(trades)
        
        return {
            'trade_count': len(trades),
            'avg_duration_hours': np.mean(trade_durations) if trade_durations else 0,
            'median_duration_hours': np.median(trade_durations) if trade_durations else 0,
            'pnl_stats': {
                'mean': np.mean(pnls),
                'median': np.median(pnls),
                'std': np.std(pnls),
                'skewness': stats.skew(pnls) if len(pnls) > 2 else 0,
                'kurtosis': stats.kurtosis(pnls) if len(pnls) > 2 else 0,
            },
            'pnl_pct_stats': {
                'mean': np.mean(pnl_pcts),
                'median': np.median(pnl_pcts),
                'std': np.std(pnl_pcts),
            },
            'consecutive_stats': consecutive_stats,
            'monthly_performance': monthly_performance,
            'best_trade': max(trades, key=lambda x: x.pnl),
            'worst_trade': min(trades, key=lambda x: x.pnl),
        }
    
    def _analyze_consecutive_trades(self, trades: List[Trade]) -> Dict[str, int]:
        """Analysiert aufeinanderfolgende Gewinne/Verluste"""
        if not trades:
            return {}
        
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif trade.pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'current_win_streak': current_win_streak,
            'current_loss_streak': current_loss_streak,
        }
    
    def _analyze_monthly_performance(self, trades: List[Trade]) -> Dict[str, float]:
        """Analysiert monatliche Performance"""
        if not trades:
            return {}
        
        monthly_pnl = {}
        for trade in trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime('%Y-%m')
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl
        
        if monthly_pnl:
            monthly_values = list(monthly_pnl.values())
            return {
                'avg_monthly_pnl': np.mean(monthly_values),
                'best_month': max(monthly_values),
                'worst_month': min(monthly_values),
                'profitable_months': len([x for x in monthly_values if x > 0]),
                'total_months': len(monthly_values),
                'monthly_win_rate': len([x for x in monthly_values if x > 0]) / len(monthly_values) * 100,
            }
        
        return {}

class RiskAnalyzer:
    """Risiko-Analyse für Backtests"""
    
    def __init__(self, logger):
        self.logger = logger.get_logger(__name__)
    
    def analyze_risk_metrics(self, equity_curve: List[float], 
                           timestamps: List[datetime]) -> Dict[str, float]:
        """Berechnet erweiterte Risiko-Metriken"""
        if len(equity_curve) < 2:
            return {}
        
        equity_series = pd.Series(equity_curve, index=timestamps)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) < 2:
            return {}
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum Drawdown Duration
        running_max = equity_series.expanding().max()
        drawdown = equity_series - running_max
        
        # Drawdown-Perioden finden
        drawdown_periods = self._find_drawdown_periods(drawdown)
        max_dd_duration = max([dd['duration'] for dd in drawdown_periods]) if drawdown_periods else 0
        
        # Tail Ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        tail_ratio = (np.percentile(positive_returns, 95) if len(positive_returns) > 0 else 0) / \
                    (abs(np.percentile(negative_returns, 5)) if len(negative_returns) > 0 else 1)
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2)) if len(drawdown) > 0 else 0
        
        return {
            'var_95': var_95 * 100,
            'var_99': var_99 * 100,
            'cvar_95': cvar_95 * 100,
            'cvar_99': cvar_99 * 100,
            'max_dd_duration_days': max_dd_duration,
            'tail_ratio': tail_ratio,
            'ulcer_index': ulcer_index,
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252) * 100,
            'upside_deviation': returns[returns > 0].std() * np.sqrt(252) * 100,
        }
    
    def _find_drawdown_periods(self, drawdown: pd.Series) -> List[Dict]:
        """Findet Drawdown-Perioden"""
        periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                # Start einer neuen Drawdown-Periode
                in_drawdown = True
                start_date = date
                max_dd = dd
            elif dd < 0 and in_drawdown:
                # Fortsetzung der Drawdown-Periode
                max_dd = min(max_dd, dd)
            elif dd == 0 and in_drawdown:
                # Ende der Drawdown-Periode
                in_drawdown = False
                duration = (date - start_date).days
                periods.append({
                    'start': start_date,
                    'end': date,
                    'duration': duration,
                    'max_drawdown': max_dd
                })
        
        return periods

class WalkForwardAnalyzer:
    """Walk-Forward Analyse für robuste Backtests"""
    
    def __init__(self, backtester, logger):
        self.backtester = backtester
        self.logger = logger.get_logger(__name__)
    
    def run_walk_forward_analysis(self, strategy_class, data: pd.DataFrame,
                                base_config: BacktestConfig, 
                                train_periods: int = 252,  # 1 Jahr Training
                                test_periods: int = 63,    # 3 Monate Test
                                step_size: int = 63) -> Dict[str, Any]:
        """Führt Walk-Forward Analyse durch"""
        
        self.logger.info("Starte Walk-Forward Analyse")
        
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        
        results = []
        oos_results = []  # Out-of-Sample Ergebnisse
        
        start_idx = train_periods
        
        while start_idx + test_periods < len(data):
            # Training Data
            train_data = data.iloc[start_idx - train_periods:start_idx]
            
            # Test Data (Out-of-Sample)
            test_data = data.iloc[start_idx:start_idx + test_periods]
            
            # Konfiguration für Test-Periode anpassen
            test_config = BacktestConfig(
                symbol=base_config.symbol,
                start_date=test_data.index[0].strftime('%Y-%m-%d'),
                end_date=test_data.index[-1].strftime('%Y-%m-%d'),
                initial_balance=base_config.initial_balance,
                commission=base_config.commission,
                slippage=base_config.slippage
            )
            
            try:
                # Backtest auf Test-Daten
                result = self.backtester.run_backtest(strategy_class, test_data, test_config)
                oos_results.append(result)
                
                self.logger.info(f"Walk-Forward Periode {len(oos_results)}: "
                               f"Return: {result.total_return_pct:.2f}%, "
                               f"Trades: {result.total_trades}")
                
            except Exception as e:
                self.logger.error(f"Fehler in Walk-Forward Periode: {e}")
            
            start_idx += step_size
        
        if not oos_results:
            return {'error': 'Keine erfolgreichen Walk-Forward Perioden'}
        
        # Gesamtergebnisse berechnen
        total_return = sum(r.total_return for r in oos_results)
        avg_return = np.mean([r.total_return_pct for r in oos_results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in oos_results])
        consistency = len([r for r in oos_results if r.total_return_pct > 0]) / len(oos_results)
        
        return {
            'periods': len(oos_results),
            'total_return': total_return,
            'avg_return_pct': avg_return,
            'avg_sharpe_ratio': avg_sharpe,
            'consistency': consistency * 100,
            'individual_results': oos_results,
            'best_period': max(oos_results, key=lambda x: x.total_return_pct),
            'worst_period': min(oos_results, key=lambda x: x.total_return_pct),
        }

class MonteCarloAnalyzer:
    """Monte Carlo Simulation für Backtest-Validierung"""
    
    def __init__(self, logger):
        self.logger = logger.get_logger(__name__)
    
    def run_monte_carlo_simulation(self, trades: List[Trade], 
                                 initial_balance: float = 10000,
                                 simulations: int = 1000) -> Dict[str, Any]:
        """Führt Monte Carlo Simulation basierend auf Trade-Historie durch"""
        
        if not trades:
            return {'error': 'Keine Trades für Monte Carlo Simulation'}
        
        self.logger.info(f"Starte Monte Carlo Simulation mit {simulations} Durchläufen")
        
        # Trade-Returns extrahieren
        trade_returns = [trade.pnl_pct / 100 for trade in trades]
        
        # Monte Carlo Simulationen
        final_balances = []
        max_drawdowns = []
        
        for sim in range(simulations):
            # Zufällige Reihenfolge der Trades
            shuffled_returns = np.random.choice(trade_returns, size=len(trades), replace=True)
            
            # Equity Curve simulieren
            balance = initial_balance
            equity_curve = [balance]
            peak = balance
            max_dd = 0
            
            for ret in shuffled_returns:
                balance *= (1 + ret)
                equity_curve.append(balance)
                
                # Drawdown berechnen
                if balance > peak:
                    peak = balance
                else:
                    dd = (peak - balance) / peak
                    max_dd = max(max_dd, dd)
            
            final_balances.append(balance)
            max_drawdowns.append(max_dd * 100)
        
        # Statistiken berechnen
        final_returns = [(b - initial_balance) / initial_balance * 100 for b in final_balances]
        
        results = {
            'simulations': simulations,
            'final_return_stats': {
                'mean': np.mean(final_returns),
                'median': np.median(final_returns),
                'std': np.std(final_returns),
                'min': np.min(final_returns),
                'max': np.max(final_returns),
                'percentile_5': np.percentile(final_returns, 5),
                'percentile_95': np.percentile(final_returns, 95),
            },
            'max_drawdown_stats': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'std': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns),
                'percentile_95': np.percentile(max_drawdowns, 95),
            },
            'probability_positive': len([r for r in final_returns if r > 0]) / len(final_returns) * 100,
            'probability_drawdown_over_20': len([dd for dd in max_drawdowns if dd > 20]) / len(max_drawdowns) * 100,
        }
        
        self.logger.info(f"Monte Carlo abgeschlossen. Erwartete Rendite: {results['final_return_stats']['mean']:.2f}%")
        
        return results

# Beispiel-Nutzung und Test-Framework
class BacktestFramework:
    """Hauptklasse für komplettes Backtesting-Framework"""
    
    def __init__(self, config_manager, logger):
        self.config_manager = config_manager
        self.logger = logger
        self.backtester = BacktestEngine(config_manager, logger)
        self.performance_analyzer = PerformanceAnalyzer(logger)
        self.risk_analyzer = RiskAnalyzer(logger)
        self.walk_forward_analyzer = WalkForwardAnalyzer(self.backtester, logger)
        self.monte_carlo_analyzer = MonteCarloAnalyzer(logger)
    
    def run_comprehensive_analysis(self, strategies: List, 
                                 data_dict: Dict[str, pd.DataFrame],
                                 configs: List[BacktestConfig]) -> Dict[str, Any]:
        """Führt umfassende Analyse durch"""
        
        # 1. Standard Backtests
        self.logger.info("Phase 1: Standard Backtests")
        standard_results = self.backtester.run_multiple_backtests(strategies, data_dict, configs)
        
        # 2. Beste Strategien identifizieren
        comparison_df = self.backtester.compare_strategies(standard_results)
        top_strategies = comparison_df.head(3)  # Top 3 Strategien
        
        # 3. Erweiterte Analysen für Top-Strategien
        detailed_analysis = {}
        
        for _, row in top_strategies.iterrows():
            strategy_name = row['Strategy']
            symbol = row['Symbol']
            key = f"{strategy_name}_{symbol}"
            
            if key in standard_results:
                result = standard_results[key][0]  # Erstes Ergebnis nehmen
                
                self.logger.info(f"Detailanalyse für {strategy_name} auf {symbol}")
                
                # Performance-Analyse
                trade_analysis = self.performance_analyzer.analyze_trades(result.trades)
                
                # Risiko-Analyse
                risk_analysis = self.risk_analyzer.analyze_risk_metrics(
                    result.equity_curve, result.timestamps
                )
                
                # Monte Carlo Simulation
                mc_analysis = self.monte_carlo_analyzer.run_monte_carlo_simulation(
                    result.trades, result.config.initial_balance
                )
                
                detailed_analysis[key] = {
                    'backtest_result': result,
                    'trade_analysis': trade_analysis,
                    'risk_analysis': risk_analysis,
                    'monte_carlo': mc_analysis,
                }
        
        return {
            'standard_results': standard_results,
            'comparison_table': comparison_df,
            'detailed_analysis': detailed_analysis,
            'summary': {
                'total_strategies_tested': len(standard_results),
                'best_strategy': top_strategies.iloc[0].to_dict() if not top_strategies.empty else None,
                'avg_return': comparison_df['Return_Pct'].mean(),
                'avg_sharpe': comparison_df['Sharpe_Ratio'].mean(),
            }
        }

# Beispiel für die Verwendung
if __name__ == "__main__":
    """
    Beispiel für die Verwendung des Backtesting-Systems:
    
    # 1. Initialisierung
    from config_manager import ConfigManager
    from logger import Logger
    
    config_manager = ConfigManager()
    logger = Logger(config_manager)
    
    # 2. Framework erstellen
    framework = BacktestFramework(config_manager, logger)
    
    # 3. Daten laden (beispielhaft)
    data_dict = {
        'BTCUSDT': load_historical_data('BTCUSDT'),
        'ETHUSDT': load_historical_data('ETHUSDT'),
    }
    
    # 4. Konfigurationen erstellen
    configs = [
        BacktestConfig('BTCUSDT', '2020-01-01', '2023-12-31'),
        BacktestConfig('ETHUSDT', '2020-01-01', '2023-12-31'),
    ]
    
    # 5. Strategien (müssen implementiert sein)
    from strategy_uptrend import UptrendStrategy
    from strategy_sideways import SidewaysStrategy
    from strategy_downtrend import DowntrendStrategy
    
    strategies = [UptrendStrategy, SidewaysStrategy, DowntrendStrategy]
    
    # 6. Umfassende Analyse durchführen
    results = framework.run_comprehensive_analysis(strategies, data_dict, configs)
    
    # 7. Report generieren
    report_file = framework.backtester.generate_report(results['standard_results'])
    
    print(f"Analyse abgeschlossen. Report: {report_file}")
    """
    pass