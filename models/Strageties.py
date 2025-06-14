import backtrader as bt
class SmaCross(bt.Strategy):
    def __init__(self):
        self.sma1 = bt.ind.SMA(period=10)
        self.sma2 = bt.ind.SMA(period=30)

    def next(self):
        if not self.position:
            if self.sma1[0] > self.sma2[0]:
                self.buy()
        elif self.sma1[0] < self.sma2[0]:
            self.sell()


# Technical indicators
def calculate_technical_indicators(df):
    # Moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['Upper_Band'] = df['MA_20'] + (2 * df['Close'].rolling(20).std())
    df['Lower_Band'] = df['MA_20'] - (2 * df['Close'].rolling(20).std())

    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_Change'] = df['Volume'].pct_change()

    return df

