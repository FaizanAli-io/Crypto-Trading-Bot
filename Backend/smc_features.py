"""
Smart Money Concepts (SMC) Feature Engineering Module
Integrates with existing FeatureEngineer class
"""
import pandas as pd
import numpy as np
from loguru import logger

class SMCFeatureEngineer:
    """
    Smart Money Concepts indicators for institutional price action analysis
    Detects: Order Blocks, FVG, Liquidity Sweeps, BOS/ChoCh, Premium/Discount
    """
    
    def __init__(self, swing_lookback=5, fvg_threshold=0.001):
        """
        Args:
            swing_lookback: Candles to look back for swing points
            fvg_threshold: Minimum gap size (as % of price) to qualify as FVG
        """
        self.swing_lookback = swing_lookback
        self.fvg_threshold = fvg_threshold
    
    def add_all_smc_features(self, df):
        """
        Add all Smart Money Concepts features
        """
        try:
            df = df.copy()
            
            # Core SMC Features
            df = self._add_order_blocks(df)
            df = self._add_fair_value_gaps(df)
            df = self._add_liquidity_zones(df)
            df = self._add_market_structure(df)
            df = self._add_premium_discount_zones(df)
            df = self._add_liquidity_sweeps(df)
            df = self._add_inducement_zones(df)
            df = self._add_smc_composite_signal(df)
            
            logger.info(f"Added Smart Money Concepts features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding SMC features: {e}")
            raise
    
    def _add_order_blocks(self, df):
        """
        Order Blocks: Last opposing candle before strong move
        Bullish OB: Last down candle before strong up move
        Bearish OB: Last up candle before strong down move
        """
        df['bullish_ob'] = 0
        df['bearish_ob'] = 0
        df['ob_strength'] = 0.0
        df['current_ob_distance'] = 0.0
        
        # Identify strong moves (threshold: 1.5% move in 3 candles)
        df['strong_up_move'] = (df['close'].pct_change(3) > 0.015).astype(int)
        df['strong_down_move'] = (df['close'].pct_change(3) < -0.015).astype(int)
        
        for i in range(3, len(df)):
            # Bullish Order Block (before strong up move)
            if df['strong_up_move'].iloc[i]:
                # Find last bearish candle before the move
                for lookback in range(1, min(5, i)):
                    if df['close'].iloc[i - lookback] < df['open'].iloc[i - lookback]:
                        df.loc[df.index[i], 'bullish_ob'] = 1
                        # OB zone: low to high of that candle
                        ob_low = df['low'].iloc[i - lookback]
                        ob_high = df['high'].iloc[i - lookback]
                        df.loc[df.index[i], 'bullish_ob_low'] = ob_low
                        df.loc[df.index[i], 'bullish_ob_high'] = ob_high
                        # Strength = size of subsequent move
                        df.loc[df.index[i], 'ob_strength'] = abs(df['close'].iloc[i] - df['close'].iloc[i - lookback]) / df['close'].iloc[i - lookback] * 100
                        break
            
            # Bearish Order Block (before strong down move)
            if df['strong_down_move'].iloc[i]:
                for lookback in range(1, min(5, i)):
                    if df['close'].iloc[i - lookback] > df['open'].iloc[i - lookback]:
                        df.loc[df.index[i], 'bearish_ob'] = 1
                        ob_low = df['low'].iloc[i - lookback]
                        ob_high = df['high'].iloc[i - lookback]
                        df.loc[df.index[i], 'bearish_ob_low'] = ob_low
                        df.loc[df.index[i], 'bearish_ob_high'] = ob_high
                        df.loc[df.index[i], 'ob_strength'] = abs(df['close'].iloc[i] - df['close'].iloc[i - lookback]) / df['close'].iloc[i - lookback] * 100
                        break
        
        # Forward fill OB zones and calculate distance
        df['bullish_ob_low'] = df['bullish_ob_low'].ffill()
        df['bullish_ob_high'] = df['bullish_ob_high'].ffill()
        df['bearish_ob_low'] = df['bearish_ob_low'].ffill()
        df['bearish_ob_high'] = df['bearish_ob_high'].ffill()
        
        # Distance from nearest OB
        df['dist_to_bull_ob'] = ((df['close'] - df['bullish_ob_high']) / df['close']) * 100
        df['dist_to_bear_ob'] = ((df['bearish_ob_low'] - df['close']) / df['close']) * 100
        
        return df
    
    def _add_fair_value_gaps(self, df):
        """
        Fair Value Gaps (FVG/Imbalance): Price inefficiencies
        Bullish FVG: Gap between candle[i-2].high and candle[i].low
        Bearish FVG: Gap between candle[i-2].low and candle[i].high
        """
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        df['fvg_size'] = 0.0
        
        for i in range(2, len(df)):
            # Bullish FVG (gap up)
            gap_up = df['low'].iloc[i] - df['high'].iloc[i - 2]
            if gap_up > 0:
                gap_pct = (gap_up / df['close'].iloc[i]) * 100
                if gap_pct >= self.fvg_threshold * 100:
                    df.loc[df.index[i], 'bullish_fvg'] = 1
                    df.loc[df.index[i], 'fvg_size'] = gap_pct
                    df.loc[df.index[i], 'fvg_top'] = df['low'].iloc[i]
                    df.loc[df.index[i], 'fvg_bottom'] = df['high'].iloc[i - 2]
            
            # Bearish FVG (gap down)
            gap_down = df['low'].iloc[i - 2] - df['high'].iloc[i]
            if gap_down > 0:
                gap_pct = (gap_down / df['close'].iloc[i]) * 100
                if gap_pct >= self.fvg_threshold * 100:
                    df.loc[df.index[i], 'bearish_fvg'] = 1
                    df.loc[df.index[i], 'fvg_size'] = gap_pct
                    df.loc[df.index[i], 'fvg_top'] = df['low'].iloc[i - 2]
                    df.loc[df.index[i], 'fvg_bottom'] = df['high'].iloc[i]
        
        # Check if price is in FVG zone
        df['fvg_top'] = df['fvg_top'].ffill()
        df['fvg_bottom'] = df['fvg_bottom'].ffill()
        df['price_in_fvg'] = ((df['close'] >= df['fvg_bottom']) & 
                               (df['close'] <= df['fvg_top'])).astype(int)
        
        return df
    
    def _add_liquidity_zones(self, df):
        """
        Liquidity Zones: Areas where stops likely cluster
        Equal highs/lows = liquidity pools
        """
        window = self.swing_lookback
        
        # Equal highs (resistance/liquidity above)
        df['equal_highs'] = 0
        df['equal_lows'] = 0
        
        for i in range(window, len(df)):
            # Check for equal highs (within 0.2% tolerance)
            recent_highs = df['high'].iloc[i - window:i]
            max_high = recent_highs.max()
            equal_count = sum(abs(recent_highs - max_high) / max_high < 0.002)
            
            if equal_count >= 2:
                df.loc[df.index[i], 'equal_highs'] = equal_count
                df.loc[df.index[i], 'liquidity_high'] = max_high
            
            # Check for equal lows (support/liquidity below)
            recent_lows = df['low'].iloc[i - window:i]
            min_low = recent_lows.min()
            equal_count = sum(abs(recent_lows - min_low) / min_low < 0.002)
            
            if equal_count >= 2:
                df.loc[df.index[i], 'equal_lows'] = equal_count
                df.loc[df.index[i], 'liquidity_low'] = min_low
        
        # Distance to liquidity zones
        df['liquidity_high'] = df['liquidity_high'].ffill()
        df['liquidity_low'] = df['liquidity_low'].ffill()
        df['dist_to_liquidity_high'] = ((df['liquidity_high'] - df['close']) / df['close']) * 100
        df['dist_to_liquidity_low'] = ((df['close'] - df['liquidity_low']) / df['close']) * 100
        
        return df
    
    def _add_market_structure(self, df):
        """
        Market Structure: BOS (Break of Structure) and ChoCh (Change of Character)
        BOS = Breaking recent high/low in direction of trend (continuation)
        ChoCh = Breaking structure against trend (reversal warning)
        """
        window = self.swing_lookback
        
        df['swing_high'] = df['high'].rolling(window * 2 + 1, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(window * 2 + 1, center=True).min() == df['low']
        
        df['bos_bullish'] = 0
        df['bos_bearish'] = 0
        df['choch_bullish'] = 0
        df['choch_bearish'] = 0
        df['market_structure_trend'] = 0  # 1=bullish, -1=bearish, 0=neutral
        
        # Track recent swing points
        last_swing_high = None
        last_swing_low = None
        trend = 0  # 1=up, -1=down
        
        for i in range(window * 2, len(df)):
            # Update swing points
            if df['swing_high'].iloc[i]:
                if last_swing_high is not None and df['high'].iloc[i] > last_swing_high:
                    if trend == 1:
                        df.loc[df.index[i], 'bos_bullish'] = 1  # Continuation
                    else:
                        df.loc[df.index[i], 'choch_bullish'] = 1  # Reversal
                        trend = 1
                last_swing_high = df['high'].iloc[i]
            
            if df['swing_low'].iloc[i]:
                if last_swing_low is not None and df['low'].iloc[i] < last_swing_low:
                    if trend == -1:
                        df.loc[df.index[i], 'bos_bearish'] = 1  # Continuation
                    else:
                        df.loc[df.index[i], 'choch_bearish'] = 1  # Reversal
                        trend = -1
                last_swing_low = df['low'].iloc[i]
            
            df.loc[df.index[i], 'market_structure_trend'] = trend
        
        return df
    
    def _add_premium_discount_zones(self, df):
        """
        Premium/Discount Zones using Fibonacci levels
        Discount = 0-50% (buy zone)
        Equilibrium = 50%
        Premium = 50-100% (sell zone)
        """
        window = 24  # 24-hour range
        
        df['range_high'] = df['high'].rolling(window).max()
        df['range_low'] = df['low'].rolling(window).min()
        df['range_size'] = df['range_high'] - df['range_low']
        
        # Calculate position in range (0 = low, 1 = high)
        df['range_position'] = ((df['close'] - df['range_low']) / 
                                (df['range_high'] - df['range_low']))
        
        # Zones
        df['in_discount'] = (df['range_position'] < 0.5).astype(int)
        df['in_premium'] = (df['range_position'] > 0.5).astype(int)
        df['at_equilibrium'] = ((df['range_position'] >= 0.45) & 
                                (df['range_position'] <= 0.55)).astype(int)
        
        # Specific Fibonacci levels
        df['fib_0'] = df['range_low']
        df['fib_236'] = df['range_low'] + df['range_size'] * 0.236
        df['fib_382'] = df['range_low'] + df['range_size'] * 0.382
        df['fib_50'] = df['range_low'] + df['range_size'] * 0.5
        df['fib_618'] = df['range_low'] + df['range_size'] * 0.618
        df['fib_786'] = df['range_low'] + df['range_size'] * 0.786
        df['fib_100'] = df['range_high']
        
        return df
    
    def _add_liquidity_sweeps(self, df):
        """
        Liquidity Sweeps: Price wicks beyond key levels then reverses
        (Stop hunts)
        """
        df['liquidity_sweep_high'] = 0
        df['liquidity_sweep_low'] = 0
        
        window = self.swing_lookback
        
        for i in range(window, len(df)):
            # Check if high swept recent highs then closed below
            recent_highs = df['high'].iloc[i - window:i]
            max_recent = recent_highs.max()
            
            if (df['high'].iloc[i] > max_recent and 
                df['close'].iloc[i] < max_recent):
                df.loc[df.index[i], 'liquidity_sweep_high'] = 1
            
            # Check if low swept recent lows then closed above
            recent_lows = df['low'].iloc[i - window:i]
            min_recent = recent_lows.min()
            
            if (df['low'].iloc[i] < min_recent and 
                df['close'].iloc[i] > min_recent):
                df.loc[df.index[i], 'liquidity_sweep_low'] = 1
        
        return df
    
    def _add_inducement_zones(self, df):
        """
        Inducement: Fake breakouts that trap retail traders
        Price moves beyond a level then quickly reverses
        """
        df['inducement_long'] = 0
        df['inducement_short'] = 0
        
        # Look for false breakouts (break then reverse within 3 candles)
        for i in range(6, len(df)):
            # False bullish breakout (inducement for longs)
            if (df['close'].iloc[i - 3] > df['high'].iloc[i - 6:i - 3].max() and
                df['close'].iloc[i] < df['close'].iloc[i - 3]):
                df.loc[df.index[i], 'inducement_long'] = 1
            
            # False bearish breakdown (inducement for shorts)
            if (df['close'].iloc[i - 3] < df['low'].iloc[i - 6:i - 3].min() and
                df['close'].iloc[i] > df['close'].iloc[i - 3]):
                df.loc[df.index[i], 'inducement_short'] = 1
        
        return df
    
    def _add_smc_composite_signal(self, df):
        """
        Composite SMC signal combining all indicators
        Positive = Bullish SMC setup, Negative = Bearish SMC setup
        """
        df['smc_signal'] = 0.0
        
        for i in range(len(df)):
            signal = 0.0
            
            try:
                # Order Blocks (+/- 20 points)
                if df['bullish_ob'].iloc[i] == 1:
                    signal += 20
                if df['bearish_ob'].iloc[i] == 1:
                    signal -= 20
                
                # FVG (+/- 15 points)
                if df['bullish_fvg'].iloc[i] == 1:
                    signal += 15
                if df['bearish_fvg'].iloc[i] == 1:
                    signal -= 15
                
                # Market Structure (+/- 25 points)
                if df['bos_bullish'].iloc[i] == 1:
                    signal += 25
                if df['bos_bearish'].iloc[i] == 1:
                    signal -= 25
                if df['choch_bullish'].iloc[i] == 1:
                    signal += 20
                if df['choch_bearish'].iloc[i] == 1:
                    signal -= 20
                
                # Premium/Discount (+/- 10 points)
                if df['in_discount'].iloc[i] == 1:
                    signal += 10
                if df['in_premium'].iloc[i] == 1:
                    signal -= 10
                
                # Liquidity Sweeps (+/- 15 points)
                if df['liquidity_sweep_low'].iloc[i] == 1:
                    signal += 15  # Bullish reversal expected
                if df['liquidity_sweep_high'].iloc[i] == 1:
                    signal -= 15  # Bearish reversal expected
                
                # Inducement (strong reversal signal +/- 20)
                if df['inducement_long'].iloc[i] == 1:
                    signal -= 20  # Bearish (trap for longs)
                if df['inducement_short'].iloc[i] == 1:
                    signal += 20  # Bullish (trap for shorts)
                
            except:
                signal = 0.0
            
            df.loc[df.index[i], 'smc_signal'] = signal
        
        return df
    
    def get_latest_smc_indicators(self, df):
        """Get latest SMC indicator values for decision making"""
        latest = df.iloc[-1]
        
        return {
            # Order Blocks
            'bullish_ob': int(latest.get('bullish_ob', 0)),
            'bearish_ob': int(latest.get('bearish_ob', 0)),
            'ob_strength': float(latest.get('ob_strength', 0)),
            'dist_to_bull_ob': float(latest.get('dist_to_bull_ob', 0)),
            'dist_to_bear_ob': float(latest.get('dist_to_bear_ob', 0)),
            
            # Fair Value Gaps
            'bullish_fvg': int(latest.get('bullish_fvg', 0)),
            'bearish_fvg': int(latest.get('bearish_fvg', 0)),
            'price_in_fvg': int(latest.get('price_in_fvg', 0)),
            'fvg_size': float(latest.get('fvg_size', 0)),
            
            # Liquidity
            'equal_highs': int(latest.get('equal_highs', 0)),
            'equal_lows': int(latest.get('equal_lows', 0)),
            'dist_to_liquidity_high': float(latest.get('dist_to_liquidity_high', 0)),
            'dist_to_liquidity_low': float(latest.get('dist_to_liquidity_low', 0)),
            
            # Market Structure
            'bos_bullish': int(latest.get('bos_bullish', 0)),
            'bos_bearish': int(latest.get('bos_bearish', 0)),
            'choch_bullish': int(latest.get('choch_bullish', 0)),
            'choch_bearish': int(latest.get('choch_bearish', 0)),
            'market_structure_trend': int(latest.get('market_structure_trend', 0)),
            
            # Premium/Discount
            'range_position': float(latest.get('range_position', 0.5)),
            'in_discount': int(latest.get('in_discount', 0)),
            'in_premium': int(latest.get('in_premium', 0)),
            'at_equilibrium': int(latest.get('at_equilibrium', 0)),
            
            # Liquidity Sweeps
            'liquidity_sweep_high': int(latest.get('liquidity_sweep_high', 0)),
            'liquidity_sweep_low': int(latest.get('liquidity_sweep_low', 0)),
            
            # Inducement
            'inducement_long': int(latest.get('inducement_long', 0)),
            'inducement_short': int(latest.get('inducement_short', 0)),
            
            # Composite
            'smc_signal': float(latest.get('smc_signal', 0)),
        }


# ===== INTEGRATION WITH EXISTING FEATUREENGINEER CLASS =====

def integrate_smc_into_feature_engineer(feature_engineer_instance):
    """
    Helper function to integrate SMC into existing FeatureEngineer class
    
    Usage:
        fe = FeatureEngineer()
        integrate_smc_into_feature_engineer(fe)
        df = fe.add_all_features(df)  # Now includes SMC!
    """
    smc = SMCFeatureEngineer()
    
    # Store original add_all_features method
    original_add_all_features = feature_engineer_instance.add_all_features
    
    # Create new method that includes SMC
    def add_all_features_with_smc(df):
        # Run original features
        df = original_add_all_features(df)
        # Add SMC features
        df = smc.add_all_smc_features(df)
        return df
    
    # Replace method
    feature_engineer_instance.add_all_features = add_all_features_with_smc
    feature_engineer_instance.smc = smc
    
    logger.info("✅ SMC features integrated into FeatureEngineer!")
    
    return feature_engineer_instance


# ===== INTEGRATION WITH EXISTING FEATUREENGINEER CLASS =====

def integrate_smc_into_feature_engineer(feature_engineer_instance):
    """
    Helper function to integrate SMC into existing FeatureEngineer class
    
    Usage:
        fe = FeatureEngineer()
        integrate_smc_into_feature_engineer(fe)
        df = fe.add_all_features(df)  # Now includes SMC!
    """
    smc = SMCFeatureEngineer()
    
    # Store original add_all_features method
    original_add_all_features = feature_engineer_instance.add_all_features
    
    # Create new method that includes SMC
    def add_all_features_with_smc(df):
        # Run original features
        df = original_add_all_features(df)
        # Add SMC features
        df = smc.add_all_smc_features(df)
        return df
    
    # Replace method
    feature_engineer_instance.add_all_features = add_all_features_with_smc
    feature_engineer_instance.smc = smc
    
    logger.info("✅ SMC features integrated into FeatureEngineer!")
    
    return feature_engineer_instance