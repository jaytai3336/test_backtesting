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