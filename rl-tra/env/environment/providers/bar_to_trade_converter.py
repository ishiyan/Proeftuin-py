from typing import List, Tuple
import numpy as np

from .entities import Trade, Candle, Kline
from .provider import Provider

class BarToTradeConverter(Provider):
    """
    Base class for providers which simulate stream of trades from candles or klines.
    
    Args:
        spread (float):
            Spread is the different between best bid and best ask prices.
            We don't have order book and we can't estimate spread from the
            stream of trades. That is why we need some predefined value to
            compute it.
            Actual spread is computed as `spread` multiplied by price.
            Default: 0.0005
    """

    def __init__(self, spread: float = 0.0005):
        super().__init__()
        if not isinstance(spread, float) or (spread < 0) or (spread > 1):
            raise ValueError(f'spread {spread} should be a float in [0, 1]')
        self.spread = spread
    
    def simulate_trades_from_candle(self, candle: Candle) -> List[Trade]:
        """
        Given a candle with fields:
        ```python
        (time_start, time_end, open, high, low, close, volume)
        ```
        produce a sequence of trades which simulate market trades.

        Since we don't have information how price moved during a candle,
        we have to make a guess.

        In order to simplify, we randomly choose one of two basic variants:
        `zigzag1` or `zigzag2`.
        
        ```text
        zigzag1 (open -> high -> low -> close):
             high
             /\\
        open   \\   close
                \\/
                low

        zigzag2 (open -> low -> high -> close):
                  high
                 /\\
        open \\  /  \\ close
              \\/
              low
        ```

        Args:
            candle (Candle):
                `(time_start, time_end, open, high, low, close, volume)`

        Returns:
            trades (List[Trade]):
                A list of trades, ordered in time.
        """
        open, high, low, close, volume = candle.open, candle.high, candle.low, candle.close, candle.volume
        if volume <= 0:
            return []
        
        # Generate a list of orders and their prices to cover the full range of a candle.
        step = self.spread * (high + low) / 2
        if np.random.random() < 0.5:
            prices, orders = self._zigzag_open_high_low_close(open, high, low, close, step)
        else:
            prices, orders = self._zigzag_open_low_high_close(open, high, low, close, step)
        
        # Generate amount for each order.
        amounts = np.ones_like(prices) * volume / len(prices)
        
        # Construct the resulting list of trades.
        result = []
        t = candle.time_start
        dt = (candle.time_end - candle.time_start) / len(prices)
        for order, price, amount in zip(orders, prices, amounts):
            result.append(Trade(
                datetime=t,
                operation=('B' if (order > 0) else 'S'),
                amount=amount,
                price=price
            ))
            t += dt
        return result
    
    def simulate_trades_from_kline(self, kline: Kline) -> List[Trade]:
        """
        Given a kline (an extended candle) with fields:
        ```python
        (time_start, time_end, open, high, low, close, volume, money, buy_volume, buy_money)
        ```
        produces a sequence of trades which simulate market trades.

        This function tries to keep overall volume weighted average price unchanged.

        A volume weighted average price (`VWAP`) is defined for a sequence of trades:
        ```text
        VWAP = (price1 * amount1 + price2 * amount2 + ... ) / (amount1 + amount2 + ...)
        ```

        Each trade has a buyer and a seller.
        So called "Buy" trade is a trade initiated by the buyer.

        The extra fields of a kline allow us to compute separately `VWAP` of the buy trades.
        ```python
        VWAP_buy = Buy_Money / Buy_Volume
        ```

        And `VWAP` for sell trades:
        ```python
        VWAP_sell = (Money - Buy_Money) / (Volume - Buy_Volume)
        ```

        This tries to preserve the `VWAP` of buy trades and the `VWAP` of sell trades.
        Thus, making overall `VWAP` of the kline also preserved.

        Args:
            kline (Kline):
                `(time_start, time_end, open, high, low, close, volume, money, buy_volume, buy_money)`

        Returns:
            trades (List[Trade]):
                A list of trades, ordered in time.
        """
        open, high, low, close = kline.open, kline.high, kline.low, kline.close
        volume, money, buy_volume, buy_money = \
            kline.volume, kline.money, kline.buy_volume, kline.buy_money
        sell_volume, sell_money = (volume - buy_volume), (money - buy_money)
        if volume <= 0:
            volume = 1.0 # return []
        
        # Generate a list of orders and their prices to cover the full range of a candle.
        step = self.spread * (high + low) / 2
        if np.random.random() < 0.5:
            prices, orders = self._zigzag_open_high_low_close(open, high, low, close, step)
        else:
            prices, orders = self._zigzag_open_low_high_close(open, high, low, close, step)
        
        # Generate amount for each order.
        amounts = np.zeros_like(prices)
        
        if buy_volume <= 0:
            orders[:] = 1
        elif sell_volume <= 0:
            orders[:] = -1
        
        if buy_volume > 0:
            buy_vwap = buy_money / buy_volume
            amounts[orders > 0] = buy_volume * self._target_coefficients(
                prices[orders > 0], target=buy_vwap)
        if sell_volume > 0:
            sell_vwap = sell_money / sell_volume
            amounts[orders < 0] = sell_volume * self._target_coefficients(
                prices[orders < 0], target=sell_vwap)
        
        # Construct the resulting list of trades.
        result = []
        t = kline.time_start
        dt = (kline.time_end - kline.time_start) / len(prices)
        for order, price, amount in zip(orders, prices, amounts):
            result.append(Trade(
                datetime=t,
                operation=('B' if (order > 0) else 'S'),
                amount=amount,
                price=price))
            t += dt
        return result
    
    @staticmethod
    def _zigzag_open_high_low_close(open, high, low, close, step) -> Tuple[np.ndarray, np.ndarray]:
        open_to_high = np.concatenate((np.arange(open, high, step), (high,)))
        high_to_low = np.concatenate((np.arange(high - step, low, -step), (low,)))
        low_to_close = np.concatenate((np.arange(low + step, close, step), (close,)))
        prices = np.concatenate((
            open_to_high,
            high_to_low,
            low_to_close))
        orders = np.concatenate((
            np.ones(len(open_to_high), dtype=np.int8),
            -1 * np.ones(len(high_to_low), dtype=np.int8),
            np.ones(len(low_to_close), dtype=np.int8)))
        return prices, orders
    
    @staticmethod
    def _zigzag_open_low_high_close(open, high, low, close, step) -> Tuple[np.ndarray, np.ndarray]:
        open_to_low = np.concatenate((np.arange(open, low, -step), (low,)))
        low_to_high = np.concatenate((np.arange(low + step, high, step), (high,)))
        high_to_close = np.concatenate((np.arange(high - step, close, -step), (close,)))
        prices = np.concatenate((
            open_to_low,
            low_to_high,
            high_to_close))
        orders = np.concatenate((
            -1 * np.ones(len(open_to_low), dtype=np.int8),
            np.ones(len(low_to_high), dtype=np.int8),
            -1 * np.ones(len(high_to_close), dtype=np.int8)))
        return prices, orders
    
    #import warnings
    @staticmethod
    def _target_coefficients(prices: np.ndarray, target: float) -> np.ndarray:
        '''
        Calculates a distribution of coefficients for a given
        set of prices based on a target price.
        '''
        # Creates a new NumPy array result with the same
        # shape and type as prices, filled with ones.
        result = np.ones_like(prices, dtype=np.float32)
        # Get the indices that would sort the prices array.
        indices = np.argsort(prices)
        # Sort the prices array.
        sorted_prices = prices[indices]
        # Find the index of the target price in the sorted prices array.
        # If the target price is not found, it returns the index
        # of the first element that is greater than the target price.
        target_index = max(1, np.searchsorted(sorted_prices, target))
        # Split the sorted prices array into two parts:
        # prices below the target price and prices above the target price.
        prices_below = sorted_prices[:target_index]
        prices_above = sorted_prices[target_index:]
        # Calculate the number of prices in each part.
        n_below, n_above = len(prices_below), len(prices_above)
        if (n_below > 0) and (n_above > 0):
            # Calculate the mean price in each part.
            mean_below, mean_above = prices_below.mean(), prices_above.mean()
            delta = mean_above - mean_below
            if delta != 0:
                # Calculate the coefficients for each part.
                k_below = (mean_above - target) / (mean_above - mean_below)
                k_above = (target - mean_below) / (mean_above - mean_below)
                # Assign the coefficients to the corresponding parts.
                result[indices[:target_index]] = k_below / n_below
                result[indices[target_index:]] = k_above / n_above
        elif n_below > 0: # n_above = 0
            # If there are no prices above the target price, calculate
            # the coefficient for the part below the target price.
            #warnings.warn(f'No prices above the target {target}: '
            #              f'n_above={n_above}, n_below={n_below}')
            # Calculate the mean price in the part below the target price.
            mean_below = prices_below.mean()
            # Calculate the coefficient for the part below the target price.
            k_below = target / mean_below
            # Assign the coefficient to the part below the target price.
            result[indices[:target_index]] = k_below / n_below
        elif n_above > 0: # n_below = 0
            # If there are no prices below the target price, calculate
            # the coefficient for the part above the target price.
            #warnings.warn(f'No prices below the target {target}: '
            #              f'n_above={n_above}, n_below={n_below}')
            # Calculate the mean price in the part above the target price.
            mean_above = prices_above.mean()
            # Calculate the coefficient for the part above the target price.
            k_above = target / mean_above
            # Assign the coefficient to the part above the target price.
            result[indices[target_index:]] = k_above / n_above
        else:
            # If there are no prices below or above the target price,
            # the result is unchanged.
            #warnings.warn(f'Target price {target} is outside of the given '
            #              f'price range: n_above={n_above}, n_below={n_below}')
            pass

        # Normalize the result array.
        result = result / result.sum()
        return result
