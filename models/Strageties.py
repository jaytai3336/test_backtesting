import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from backtrader import Cerebro, TimeFrame
from backtrader.feeds import PandasData
import quantstats as qs


class QuantBacktester:
    def __init__(self, data, model=None):
        self.data = data
        self.model = model or RandomForestClassifier(n_estimators=100)
        self.results = None

    def preprocess_data(self):
        """Feature engineering for financial data"""
        df = self.data.copy()

        # Returns and target
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

        # Technical features
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

        # Volatility
        df['Volatility'] = df['Return'].rolling(21).std()

        self.processed_data = df.dropna()
        return self

    def _calculate_rsi(self, series, window):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def train_model(self, features=None):
        """Train predictive model"""
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        features = features or ['MA_10', 'MA_50', 'RSI', 'MACD', 'Volatility']
        X = self.processed_data[features]
        y = self.processed_data['Target']

        # Time-based split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(X_train, y_train)
        print(classification_report(y_test, self.model.predict(X_test)))
        return self

    def backtest(self, initial_cash=100000, commission=0.001):
        """Run backtest with Backtrader"""

        class SignalData(PandasData):
            lines = ('signal',)
            params = (('signal', -1),)

        # Prepare DataFrame with predictions
        df = self.processed_data.copy()
        df['Signal'] = self.model.predict(df[features])

        # Initialize Cerebro
        cerebro = Cerebro()
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)

        # Add strategy
        cerebro.addstrategy(self.Strategy)

        # Load data
        data = SignalData(dataname=df)
        cerebro.adddata(data)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        self.results = cerebro.run()
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

        return self

    def analyze_results(self):
        """Generate performance reports"""
        if not self.results:
            raise ValueError("Run backtest first")

        strat = self.results[0]

        # QuantStats report
        returns = pd.Series(strat.analyzers.returns.get_analysis())
        qs.reports.full(returns)

        # Plot
        cerebro.plot(style='candlestick')

    class Strategy(bt.Strategy):
        params = (('risk_per_trade', 0.01),)

        def __init__(self):
            self.signal = self.data.signal

        def next(self):
            if not self.position:
                if self.signal[0] == 1:  # Buy signal
                    size = (self.broker.getvalue() * self.p.risk_per_trade) / self.data.close[0]
                    self.buy(size=size)
            elif self.signal[0] == 0:  # Sell signal
                self.close()