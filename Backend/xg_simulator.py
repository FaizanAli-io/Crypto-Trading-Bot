"""
Integrated Trading Simulator
Combines backtest predictions with price estimation for realistic trade simulation
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from tabulate import tabulate

from xg_validator import ModelValidator
from xg_predict import SignalPredictor


class IntegratedTradingSimulator:
    """Simulate trading using backtest predictions and price targets"""
    
    def __init__(self, binance_client=None):
        self.validator = ModelValidator(binance_client)
        self.predictor = SignalPredictor(binance_client)
        self.results_dir = Path("integrated_simulation_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def simulate_from_backtest(self, symbol, horizon, initial_capital=1000,
                               days=30, min_confidence=0.7, trading_fee=0.001,
                               position_size=0.95, stop_loss_pct=None,
                               take_profit_pct=None, use_smc=False, interval="1h"):
        """
        Run complete simulation using backtest predictions and price estimates
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            horizon: Prediction horizon in hours
            initial_capital: Starting capital in USD
            days: Days to backtest
            min_confidence: Minimum confidence threshold
            trading_fee: Trading fee per trade (0.001 = 0.1%)
            position_size: Fraction of capital to use per trade (0.95 = 95%)
            stop_loss_pct: Stop loss percentage (e.g., 0.02 = 2%)
            take_profit_pct: Take profit percentage (e.g., 0.03 = 3%)
            use_smc: If True, try to load SMC model first, fallback to simple model
            interval: Candle interval (e.g., "15m", "1h") - must match trained model
        
        Returns:
            dict: Complete simulation results with trades and metrics
        """
        logger.info("="*80)
        logger.info("INTEGRATED TRADING SIMULATION")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Horizon: {horizon}h")
        logger.info(f"Interval: {interval}")
        logger.info(f"Model Type: {'SMC (with fallback)' if use_smc else 'Simple'}")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"Position Size: {position_size:.0%}")
        logger.info(f"Trading Fee: {trading_fee:.2%} per trade")
        if stop_loss_pct:
            logger.info(f"Stop Loss: {stop_loss_pct:.1%}")
        if take_profit_pct:
            logger.info(f"Take Profit: {take_profit_pct:.1%}")
        logger.info("="*80)
        
        # Convert horizon from hours to minutes
        horizon_minutes = horizon
        
        # Step 1: Run backtest to get predictions
        logger.info("\nStep 1: Running backtest to get predictions...")
        backtest_results = self.validator.backtest_for_simulation(
            symbol=symbol,
            interval=interval,
            horizon_minutes=horizon_minutes,
            days=days,
            min_confidence=min_confidence,
            use_smc=use_smc
        )
        
        if backtest_results['num_predictions'] == 0:
            raise ValueError("No predictions generated from backtest")
        
        logger.info(f"✓ Generated {backtest_results['num_predictions']} predictions")
        
        # Step 2: Load prediction data and full dataframe
        predictions = backtest_results['predictions']
        actuals = backtest_results['actuals']
        confidences = backtest_results['confidences']
        current_prices = backtest_results['current_prices']
        future_prices = backtest_results['future_prices']
        timestamps = [datetime.fromisoformat(ts) for ts in backtest_results['timestamps']]
        
        # Get full historical dataframe for price estimation
        df_full = backtest_results.get('dataframe')  # Assume validator provides this
        if df_full is None:
            logger.warning("Full dataframe not available, falling back to simple estimation")
            df_full = None
        
        # Step 3: Simulate trades
        logger.info("\nStep 2: Simulating trades with P&L tracking...")
        
        capital = initial_capital
        trades = []
        portfolio_values = [initial_capital]
        portfolio_timestamps = [timestamps[0]]
        
        for i in range(len(predictions)):
            timestamp = timestamps[i]
            prediction = predictions[i]  # 0=DOWN, 1=UP
            actual = actuals[i]
            confidence = confidences[i]
            entry_price = current_prices[i]
            actual_exit_price = future_prices[i]
            
            # Calculate position amount
            position_amount = capital * position_size
            
            # Skip if insufficient capital
            if position_amount < 10:
                logger.warning(f"[{timestamp}] Insufficient capital: ${capital:.2f}")
                break
            
            # Determine trade direction
            if prediction == 1:
                position_type = "LONG"
            else:
                position_type = "SHORT"
            
            # Calculate expected exit price using historical pattern-based estimation
            # if df_full is not None:
            #     # Get historical data up to this point
            #     df_up_to_now = df_full[df_full.index <= timestamp]
            #     estimated_target = self._estimate_target_price(
            #         df_up_to_now, entry_price, prediction, confidence, horizon
            #     )
            # else:
            #     # Fallback to simple estimation
            #     estimated_target = self._estimate_target_price_simple(
            #         entry_price, prediction, confidence, horizon
            #     )
            df_up_to_now = df_full[df_full.index <= timestamp]
            estimated_target = self._estimate_target_price(
                    df_up_to_now, entry_price, prediction, confidence, horizon
                )
            # For realistic simulation, use actual future price
            exit_price = actual_exit_price
            
            # Calculate P&L based on position type
            if position_type == "LONG":
                price_change = exit_price - entry_price
                pnl_pct = price_change / entry_price
            else:  # SHORT
                price_change = entry_price - exit_price
                pnl_pct = price_change / entry_price
            
            # Calculate fees (entry + exit)
            fee = position_amount * trading_fee * 2
            
            # Calculate realized P&L
            gross_pnl = position_amount * pnl_pct
            realized_pnl = gross_pnl - fee
            
            # Check stop loss
            hit_stop_loss = False
            if stop_loss_pct and pnl_pct < -stop_loss_pct:
                exit_price = entry_price * (1 - stop_loss_pct) if position_type == "LONG" else entry_price * (1 + stop_loss_pct)
                realized_pnl = -position_amount * stop_loss_pct - fee
                hit_stop_loss = True
            
            # Check take profit
            hit_take_profit = False
            if take_profit_pct and pnl_pct > take_profit_pct:
                exit_price = entry_price * (1 + take_profit_pct) if position_type == "LONG" else entry_price * (1 - take_profit_pct)
                realized_pnl = position_amount * take_profit_pct - fee
                hit_take_profit = True
            
            # Update capital
            capital += realized_pnl
            
            # Determine exit reason
            if hit_stop_loss:
                exit_reason = "STOP_LOSS"
            elif hit_take_profit:
                exit_reason = "TAKE_PROFIT"
            else:
                exit_reason = "HORIZON_REACHED"
            
            # Extract estimated target price
            if isinstance(estimated_target, dict):
                estimated_price = estimated_target['target_price']
            else:
                estimated_price = estimated_target
            
            # Record trade
            trade = {
                'trade_number': i + 1,
                'entry_time': timestamp.isoformat(),
                'exit_time': (timestamp + timedelta(hours=horizon)).isoformat(),
                'type': position_type,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'estimated_target': round(estimated_price, 2),
                'actual_exit': round(actual_exit_price, 2),
                'position_amount': round(position_amount, 2),
                'confidence': round(confidence, 4),
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'actual_direction': 'UP' if actual == 1 else 'DOWN',
                'prediction_correct': prediction == actual,
                'pnl': round(realized_pnl, 2),
                'pnl_pct': round((realized_pnl / position_amount) * 100, 2),
                'gross_pnl': round(gross_pnl, 2),
                'fee': round(fee, 2),
                'exit_reason': exit_reason,
                'capital_after': round(capital, 2)
            }
            
            trades.append(trade)
            portfolio_values.append(capital)
            portfolio_timestamps.append(timestamp + timedelta(hours=horizon))
            
            # Log trade
            emoji = "✓" if trade['prediction_correct'] else "✗"
            pnl_emoji = "📈" if realized_pnl > 0 else "📉"
            logger.info(f"[{i+1:2d}] {emoji} {timestamp.strftime('%Y-%m-%d %H:%M')} | "
                       f"{position_type:5s} @ ${entry_price:7,.2f} → ${exit_price:7,.2f} | "
                       f"{pnl_emoji} ${realized_pnl:+7.2f} ({trade['pnl_pct']:+.2f}%) | "
                       f"Capital: ${capital:,.2f}")
        
        # Calculate final statistics
        final_capital = capital
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        correct_predictions = [t for t in trades if t['prediction_correct']]
        
        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
        accuracy = (len(correct_predictions) / len(trades) * 100) if trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = (sum([t['pnl'] for t in winning_trades]) / 
                        abs(sum([t['pnl'] for t in losing_trades]))) if losing_trades else float('inf')
        
        # Calculate max drawdown
        peak = initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Compile results
        summary = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'directional_accuracy': accuracy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor if profit_factor != float('inf') else None,
            'max_drawdown_pct': max_drawdown,
            'best_trade': max(trades, key=lambda x: x['pnl']) if trades else None,
            'worst_trade': min(trades, key=lambda x: x['pnl']) if trades else None,
            'total_fees_paid': sum([t['fee'] for t in trades]),
            'avg_confidence': np.mean([t['confidence'] for t in trades]) if trades else 0
        }
        
        results = {
            'simulation_params': {
                'symbol': symbol,
                'horizon': horizon,
                'initial_capital': initial_capital,
                'days': days,
                'min_confidence': min_confidence,
                'trading_fee': trading_fee,
                'position_size': position_size,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct
            },
            'backtest_metrics': backtest_results['metrics'],
            'trading_summary': summary,
            'trades': trades,
            'portfolio_history': {
                'values': portfolio_values,
                'timestamps': [ts.isoformat() for ts in portfolio_timestamps]
            }
        }
        
        # Print results
        self._print_trades_table(trades)
        self._print_summary(symbol, horizon, summary)
        
        # Save results
        # self._save_results(symbol, horizon, results)
        
        return results
    
    def _estimate_target_price(self, df, current_price, direction, confidence, horizon):
        """
        Estimate target price using historical movement patterns
        (Imported from SignalPredictor class)
        
        WARNING: This is a rough estimate and should NOT be treated as precise.
        The model only predicts DIRECTION, not exact price.
        """
        # Calculate historical movements for this horizon
        df_recent = df.tail(1440)  # Last 60 days (24*60 hours)
        
        if len(df_recent) < horizon + 1:
            logger.warning("Insufficient data for historical estimation, using simple method")
            return self._estimate_target_price_simple(current_price, direction, confidence, horizon)
        
        # Calculate actual price changes over horizon periods
        price_changes = []
        for i in range(len(df_recent) - horizon):
            current = df_recent.iloc[i]['close']
            future = df_recent.iloc[i + horizon]['close']
            pct_change = (future - current) / current
            price_changes.append(pct_change)
        
        price_changes = np.array(price_changes)
        
        if len(price_changes) == 0:
            return self._estimate_target_price_simple(current_price, direction, confidence, horizon)
        
        # Remove outliers (beyond 2 standard deviations)
        mean_change = np.mean(price_changes)
        std_change = np.std(price_changes)
        
        filtered_changes = price_changes[
            np.abs(price_changes - mean_change) < 2 * std_change
        ]
        
        if len(filtered_changes) == 0:
            filtered_changes = price_changes
        
        if direction == 1:  # UP prediction
            # Use positive changes only
            up_moves = filtered_changes[filtered_changes > 0]
            if len(up_moves) > 0:
                # Weight by confidence: higher confidence = use higher percentile
                percentile = 50 + (confidence - 0.5) * 60  # Range: 50th to 80th percentile
                expected_change = np.percentile(up_moves, percentile)
            else:
                expected_change = abs(mean_change)  # Fallback to absolute mean
        else:  # DOWN prediction (direction == 0)
            # Use negative changes only
            down_moves = filtered_changes[filtered_changes < 0]
            if len(down_moves) > 0:
                percentile = 50 - (confidence - 0.5) * 60  # Range: 50th to 20th percentile
                expected_change = np.percentile(down_moves, percentile)
            else:
                expected_change = -abs(mean_change)  # Fallback to negative mean
        
        # Calculate target price
        target_price = current_price * (1 + expected_change)
        
        # Calculate price range (confidence interval)
        price_std = current_price * std_change
        lower_bound = target_price - price_std
        upper_bound = target_price + price_std
        # if direction == 1:
        #     target_price = 0.7*target_price
        # else:
        #     target_price = 1.3*target_price
        return {
            'method': 'historical_average',
            'target_price': target_price,
            'expected_change_pct': expected_change * 100,
            'price_range': {
                'lower': lower_bound,
                'upper': upper_bound
            },
            'warning': 'This is a statistical estimate based on past patterns. NOT a guarantee.'
        }
    
    def _estimate_target_price_simple(self, current_price, direction, confidence, horizon):
        """
        Simple price estimation based on direction and confidence
        This is a rough estimate - fallback method
        """
        # Base volatility assumption (can be refined with actual data)
        base_volatility = 0.015 * horizon  # 1.5% per hour
        
        # Adjust by confidence
        expected_move = base_volatility * (0.5 + confidence * 0.5)
        
        if direction == 1:  # UP
            target = current_price * (1 + expected_move)
        else:  # DOWN
            target = current_price * (1 - expected_move)
        
        return target
    
    
    def _print_trades_table(self, trades):
        """Print trades in formatted table"""
        if not trades:
            logger.info("\nNo trades executed")
            return
        
        logger.info("\n" + "="*80)
        logger.info("TRADE HISTORY")
        logger.info("="*80)
        
        table_data = []
        for trade in trades:
            entry_time = datetime.fromisoformat(trade['entry_time'])
            table_data.append([
                trade['trade_number'],
                entry_time.strftime('%Y-%m-%d %H:%M'),
                trade['type'],
                f"${trade['entry_price']:,.2f}",
                f"${trade['exit_price']:,.2f}",
                f"{trade['confidence']:.1%}",
                "✓" if trade['prediction_correct'] else "✗",
                f"${trade['pnl']:+,.2f}",
                f"{trade['pnl_pct']:+.2f}%",
                trade['exit_reason']
            ])
        
        headers = ['#', 'Entry Time', 'Type', 'Entry $', 'Exit $', 
                  'Conf', 'Correct', 'P&L', 'P&L %', 'Exit']
        
        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def _print_summary(self, symbol, horizon, summary):
        """Print simulation summary"""
        logger.info("\n" + "="*80)
        logger.info("SIMULATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol} | Horizon: {horizon}h")
        
        logger.info(f"\nCAPITAL:")
        logger.info(f"  Initial:        ${summary['initial_capital']:,.2f}")
        logger.info(f"  Final:          ${summary['final_capital']:,.2f}")
        logger.info(f"  Total Return:   ${summary['total_return']:+,.2f} ({summary['total_return_pct']:+.2f}%)")
        
        profit_emoji = "🎉" if summary['total_return'] > 0 else "⚠️"
        logger.info(f"  {profit_emoji} {'PROFIT' if summary['total_return'] > 0 else 'LOSS'}")
        
        logger.info(f"\nTRADING STATS:")
        logger.info(f"  Total Trades:         {summary['num_trades']}")
        logger.info(f"  Winning Trades:       {summary['winning_trades']} ({summary['win_rate']:.1f}%)")
        logger.info(f"  Losing Trades:        {summary['losing_trades']}")
        logger.info(f"  Directional Accuracy: {summary['directional_accuracy']:.1f}%")
        logger.info(f"  Average Confidence:   {summary['avg_confidence']:.1%}")
        
        logger.info(f"\nPERFORMANCE METRICS:")
        logger.info(f"  Average Win:      ${summary['avg_win']:,.2f}")
        logger.info(f"  Average Loss:     ${summary['avg_loss']:,.2f}")
        if summary['profit_factor']:
            logger.info(f"  Profit Factor:    {summary['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown:     {summary['max_drawdown_pct']:.2f}%")
        logger.info(f"  Total Fees Paid:  ${summary['total_fees_paid']:,.2f}")
        
        if summary['best_trade']:
            best = summary['best_trade']
            logger.info(f"\nBEST TRADE:")
            logger.info(f"  Trade #{best['trade_number']} | {best['type']} | "
                       f"${best['pnl']:+,.2f} ({best['pnl_pct']:+.2f}%) | "
                       f"{best['entry_time']}")
        
        if summary['worst_trade']:
            worst = summary['worst_trade']
            logger.info(f"\nWORST TRADE:")
            logger.info(f"  Trade #{worst['trade_number']} | {worst['type']} | "
                       f"${worst['pnl']:+,.2f} ({worst['pnl_pct']:+.2f}%) | "
                       f"{worst['entry_time']}")
        
        logger.info("="*80 + "\n")
    
    def _save_results(self, symbol, horizon, results):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_sim_{symbol}_{horizon}h_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")


# ========================
# CLI Interface
# ========================
if __name__ == "__main__":
    import sys
    from binance.client import Client
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    
    if api_key:
        binance_client = Client(api_key, api_secret)
    else:
        binance_client = Client()
    
    simulator = IntegratedTradingSimulator(binance_client)
    
    # Example usage
    results = simulator.simulate_from_backtest(
        symbol="DOTUSDT",
        horizon=15,  # 1 hour prediction
        interval="15m",  # Must match trained model
        initial_capital=1000,
        days=20,
        min_confidence=0.85,
        trading_fee=0.001,  # 0.1%
        position_size=0.95,  # Use 95% of capital per trade
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.03,  # 3% take profit
        use_smc=False  # Try SMC model first, fallback to simple
    )
    
    print("\n✓ Simulation complete! Check the JSON file for detailed results.")