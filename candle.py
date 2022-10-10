# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 20:11:54 2022

@author: 82109
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 22:53:10 2022

@author: 82109
"""

import requests
import datetime
import pandas as pd
import time
import matplotlib.pyplot as plt
import mplfinance as fplt
import numpy as np

class Position:
    def __init__(self,symbol,cash):
        self.err = 0.00001
        self.symbol = symbol
        self.cash = cash
        self.qty = 0.0
        self.avg_price = 0.0
        self.trade_hist = []
        
    def setPosition(self,qty,avg_price):
        self.qty = qty
        self.avg_price = avg_price
        pass
    
    def getAvgPrice(self):
        return self.avg_price
    
    def getQty(self):
        return self.qty
    
    def addTrade(self,buy_sell,qty,price,t = time.time()):
        self.trade_hist.append([buy_sell,qty,price,t])
        
        if buy_sell == 'buy':
            if qty < 0:
                print('trade error')
                return 0
            
        elif buy_sell == 'sell':
            if qty > 0:
                print('trade error')
                return 0
            
        if buy_sell == 'buy':
            if self.qty >= 0:
                self.avg_price = (self.avg_price * self.qty + price * qty) / (self.qty + qty)
            else:
                if abs(self.qty + qty) < self.err:
                    self.avg_price = 0
                elif self.qty + qty > 0:
                    self.avg_price = price
                    
        elif buy_sell == 'sell':
            if self.qty <= 0:
                self.avg_price = (self.avg_price * self.qty + price * qty) / (self.qty + qty)
            else:
                if abs(self.qty + qty) < self.err:
                    self.avg_price = 0
                elif self.qty + qty < 0:
                    self.avg_price = price
                    
        self.qty += qty
        
        if abs(self.qty) < self.err:
            self.qty = 0
            self.avg_price = 0

class Candle:
    def __init__(self, df, symbol, interval):
        self._df = df.sort_index(ascending=False)
        self._symbol = symbol
        self._interval = interval
    
    def setCandle(self, df):
        self._df = df.sort_index(ascending=False)

    def getIndex(self, i):
        return self._df.index[i]
    
    def getDf(self):
        return self._df
    
    def getCandle(self, i):
        return self._df.iloc[i]

    def getOpen(self, i):
        return self.getCandle(i)['open']

    def getClose(self, i):
        return self.getCandle(i)['close']

    def getVol(self, i):
        return self.getCandle(i)['volume']

    def getTrades(self, i):
        return self.getCandle(i)['trades']
        
    def getVol(self, i):
        return self.getCandle(i)['volume']
                
    def candleSign(self, i):
        if self.getCandle(i)['open'] < self.getCandle(i)['close']:
            sign = 1
        elif self.getCandle(i)['open'] > self.getCandle(i)['close']:
            sign = -1
        else:
            sign = 0
        return sign
    
    def candleReturn(self, i):
        return (self._df.iloc[i]['close'] - self._df.iloc[i]['open'])/self._df.iloc[i]['open']
    
    def candleName(self, i):
        upper_shadow = self._df.iloc[i]['high'] - max(self._df.iloc[i]['close'],self._df.iloc[i]['open'])
        lower_shadow = self._df.iloc[i]['low'] - min(self._df.iloc[i]['close'],self._df.iloc[i]['open'])
        real_body = abs(self._df.iloc[i]['close'] - self._df.iloc[i]['open'])
        sign = self.candleSign(i)
        return 0




def get_candle(symbol, interval, cnt):
    COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 
               'tb_base_av', 'tb_quote_av', 'ignore']
    URL = 'https://fapi.binance.com/fapi/v1/klines'
    data = []
    if interval == '5m':
        m = 5
    
    end = int(time.time() * 1000)
    start = int(end - m * 60 * cnt * 1000)
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000,
        'startTime': start,
        'endTime': end
    }
    while start < end:
        #print(datetime.datetime.fromtimestamp(start // 1000))
        params['startTime'] = start
        result = requests.get(URL, params = params)
        js = result.json()
        if not js:
            break
        data.extend(js)  # result에 저장
        start = js[-1][0] + 60000  # 다음 step으로
    # 전처리
    if not data:  # 해당 기간에 데이터가 없는 경우
        print('해당 기간에 일치하는 데이터가 없습니다.')
        return -1
    df = pd.DataFrame(data)
    df.columns = COLUMNS
    df['open_time'] = df.apply(lambda x:datetime.datetime.fromtimestamp(x['open_time'] // 1000), axis=1)
    df = df.drop(columns = ['close_time', 'ignore'])
    df['symbol'] = symbol
    df.loc[:, 'open':'tb_quote_av'] = df.loc[:, 'open':'tb_quote_av'].astype(float)  # string to float
    df['trades'] = df['trades'].astype(int)
    df = df.set_index('open_time')
    return df

def get_data(start_date, end_date, symbol):
    COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 
               'tb_base_av', 'tb_quote_av', 'ignore']
    URL = 'https://fapi.binance.com/fapi/v1/klines'
    data = []
    
    start = int(time.mktime(datetime.datetime.strptime(start_date + ' 00:00', '%Y-%m-%d %H:%M').timetuple())) * 1000
    end = int(time.mktime(datetime.datetime.strptime(end_date +' 23:59', '%Y-%m-%d %H:%M').timetuple())) * 1000
    params = {
        'symbol': symbol,
        'interval': '5m',
        'limit': 1000,
        'startTime': start,
        'endTime': end
    }
    
    while start < end:
        #print(datetime.datetime.fromtimestamp(start // 1000))
        params['startTime'] = start
        result = requests.get(URL, params = params)
        js = result.json()
        if not js:
            break
        data.extend(js)  # result에 저장
        start = js[-1][0] + 60000  # 다음 step으로
    # 전처리
    if not data:  # 해당 기간에 데이터가 없는 경우
        print('해당 기간에 일치하는 데이터가 없습니다.')
        return -1
    df = pd.DataFrame(data)
    df.columns = COLUMNS
    df['open_time'] = df.apply(lambda x:datetime.datetime.fromtimestamp(x['open_time'] // 1000), axis=1)
    df = df.drop(columns = ['close_time', 'ignore'])
    df['symbol'] = symbol
    df.loc[:, 'open':'tb_quote_av'] = df.loc[:, 'open':'tb_quote_av'].astype(float)  # string to float
    df['trades'] = df['trades'].astype(int)
    return df

start_date = '2021-09-12'
end_date = '2022-10-10'
binance = get_data(start_date, end_date, 'BTCUSDT')
binance = binance.set_index('open_time')

#%% 백테스트 실행
def check_bundle_return(bundle_cnt, ret, *args):
    for i in range(bundle_cnt):
        i += 1
        if sum(args[0:i]) <= ret:
            return i
    return 0

def check_bundle_return_up(bundle_cnt, ret, *args):
    for i in range(bundle_cnt):
        i += 1
        if sum(args[0:i]) >= ret:
            return i
    return 0

factor = {'candle_cnt': 3,
          'bundle_cnt': 3,
          'return1': -0.002,
          'return2': -0.004,
          'return3': -0.002,
          'vol1': 10000,
          'vol2': -0.004,
          'vol3': -0.004,
          'profit_cut': 0.0060,
          'loss_cut': -0.0060,
          'pre_bummping': 0.010,
          'break_cnt': 3}

candle = Candle(binance,'BTCUSDT','5M')
buy_marker = []
sell_marker = []

size = binance.index.size
i = factor['candle_cnt']*factor['bundle_cnt']
trade_series = pd.Series(np.zeros(binance.index.size), index = binance.index)
buy_amt = 0
long_size = 0
lots = 0.1
win = 0
lose = 0
break_time = 0
while i < size:
    hit_cnt = 0
    pivot = size-i
    break_time -= 1
    if break_time <= 0:
        if candle.getVol(pivot+1) >= factor['vol1'] or candle.getVol(pivot+2) >= factor['vol1'] or  candle.getVol(pivot+3) >= factor['vol1'] or candle.getVol(pivot+4) >= factor['vol1'] or  candle.getVol(pivot+5) >= factor['vol1'] or candle.getVol(pivot+6) >= factor['vol1']:
            checking = check_bundle_return(factor['bundle_cnt'],factor['return1'],candle.candleReturn(pivot+1),candle.candleReturn(pivot+2),candle.candleReturn(pivot+3))
            if checking != 0:
                checking_2 = check_bundle_return(factor['bundle_cnt'],factor['return1'],candle.candleReturn(pivot+1+checking),candle.candleReturn(pivot+2+checking),candle.candleReturn(pivot+3+checking))
                if checking_2 != 0 :
                    checking_3 = check_bundle_return(factor['bundle_cnt'],factor['return1'],candle.candleReturn(pivot+1+checking+checking_2),candle.candleReturn(pivot+2+checking+checking_2),candle.candleReturn(pivot+3+checking+checking_2))
                    if checking_3 != 0:
                        index = pivot+checking+checking_2+checking_3
                        pre_bummping = check_bundle_return_up(8, factor['pre_bummping'], candle.candleReturn(index+1),candle.candleReturn(index+2),candle.candleReturn(index+3),candle.candleReturn(index+4),candle.candleReturn(index+5),candle.candleReturn(index+6),candle.candleReturn(index+7),candle.candleReturn(index+8))
                        if pre_bummping == 0:
                            buy_marker.append(candle.getIndex(pivot))
                            long_size += lots
                            buy_amt += lots * candle.getOpen(pivot)
                            trade_series[candle.getIndex(pivot)] = lots
                            i += 1
                            break_time = 6
                            continue
        checking = check_bundle_return(factor['bundle_cnt'],factor['return2'],candle.candleReturn(pivot+1),candle.candleReturn(pivot+2),candle.candleReturn(pivot+3))
        if checking != 0:
            checking_2 = check_bundle_return(factor['bundle_cnt'],factor['return2'],candle.candleReturn(pivot+1+checking),candle.candleReturn(pivot+2+checking),candle.candleReturn(pivot+3+checking))
            if checking_2 != 0 :
                index = pivot+checking+checking_2
                pre_bummping = check_bundle_return_up(8, factor['pre_bummping'], candle.candleReturn(index+1),candle.candleReturn(index+2),candle.candleReturn(index+3),candle.candleReturn(index+4),candle.candleReturn(index+5),candle.candleReturn(index+6),candle.candleReturn(index+7),candle.candleReturn(index+8))
                if pre_bummping == 0:
                    buy_marker.append(candle.getIndex(pivot))
                    long_size += lots
                    buy_amt += lots * candle.getOpen(pivot)
                    trade_series[candle.getIndex(pivot)] = lots
                    i += 1
                    break_time = 6
                    continue
    if long_size > 0:
        if (candle.getOpen(pivot) - buy_amt / long_size) / candle.getOpen(pivot) >= factor['profit_cut']:
            sell_marker.append(candle.getIndex(pivot))
            trade_series[candle.getIndex(pivot)] = -long_size
            win +=1
            buy_amt = 0
            long_size = 0
        elif (candle.getOpen(pivot) - buy_amt / long_size) / candle.getOpen(pivot) <= factor['loss_cut']:
            sell_marker.append(candle.getIndex(pivot))
            trade_series[candle.getIndex(pivot)] = -long_size
            lose +=1
            buy_amt = 0
            long_size = 0
    i += 1
#%% 백테스트 결

binance_buy = binance.filter(items = buy_marker, axis=0)['low']*0.999
binance_buy = binance_buy.reindex(binance.index)

binance_sell = binance.filter(items = sell_marker, axis=0)['high']*1.001
binance_sell = binance_sell.reindex(binance.index)

apd = [fplt.make_addplot(binance_buy, scatter=True, markersize=130, marker=r'$\Uparrow$', color='green'),
       fplt.make_addplot(binance_sell, scatter=True, markersize=130, marker=r'$\Downarrow$', color='red')
       ]

fplt.plot(
            binance,
            type='candle',
            style='charles',
            title='BTCUSDT',
            ylabel='Price',
            volume=True,
            ylabel_lower='valume',
            addplot=apd
            )

#%%
book = pd.DataFrame(np.zeros([binance.index.size,4]),index = binance.index,columns=['price_open','trade','eval_amt','cash'])
book.trade = trade_series
book.price_open = binance['open']
book.eval_amt = book.trade.cumsum() * book.price_open
book.cash = (book.trade * book.price_open * (-1)).cumsum()

(book.cash + book.eval_amt).plot()



#%%
factor = {'interval': '5m',
          'candle_cnt': 3,
          'bundle_cnt': 3,
          'return1': -0.002,
          'return2': -0.005,
          'return3': -0.002,
          'vol1': 10000,
          'vol2': -0.004,
          'vol3': -0.004,
          'profit_cut': 0.0060,
          'loss_cut': -0.0060,
          'short_loss_cut': -0.003,
          'pre_bummping': 0.015,
          'bumming_cnt':8,
          'break_cnt': 3,
          'buy_unit': 0.01}

class ThreeDownInLowStrategy():
    def __init__(self, symbol, factor, position):
        self.symbol = symbol
        self.position = position
        self.factor = factor
        self.candle = Candle(self.getCandleData(),self.symbol,self.factor['interval'])
        
        self.interval_m = 5
        self.buy_amt = 0
        self.buy_qty = 0
        self.winnig = 0
        self.losing = 0
        self.last_index = self.candle.getIndex(0)
        self.break_time = time.time()
    
    def setPosition(self, position):
        self.position = position
        pass
    
    def setCandle(self, candle_data = 0):
        if candle_data == 0:
            self.candle = Candle(self.getCandleData(),self.symbol,self.factor['interval'])
        else:
            self.candle = candle_data
        pass
    
    def getCandleData(self):
        cnt = self.factor['bundle_cnt']*self.factor['candle_cnt']+self.factor['bumming_cnt']+1
        return get_candle(self.symbol,self.factor['interval'],cnt)
        
    def buySignal(self):
        if self.last_index != self.candle.getIndex(1):
            print('------------------')
            self.last_index = self.candle.getIndex(1)
            if self.break_time <= time.time():
                if sum([1 for i in range(1,7) if self.candle.getVol(i)>=self.factor['vol1']]) >= 1:
                    checking = check_bundle_return(self.factor['bundle_cnt'],
                                                   self.factor['return1'],
                                                   self.candle.candleReturn(1),
                                                   self.candle.candleReturn(2),
                                                   self.candle.candleReturn(3))
                    if checking != 0:
                        print('a')
                        print(self.candle.candleReturn(1))
                        print(self.candle.candleReturn(2))
                        print(self.candle.candleReturn(3))
                        print(checking)
                        checking_2 = check_bundle_return(self.factor['bundle_cnt'],
                                                         self.factor['return1'],
                                                         self.candle.candleReturn(checking+1),
                                                         self.candle.candleReturn(checking+2),
                                                         self.candle.candleReturn(checking+3))
                        if checking_2 != 0 :
                            print('b')
                            print(self.candle.candleReturn(checking+1))
                            print(self.candle.candleReturn(checking+2))
                            print(self.candle.candleReturn(checking+3))
                            print(checking_2)
                            checking_3 = check_bundle_return(self.factor['bundle_cnt'],
                                                             self.factor['return1'],
                                                             self.candle.candleReturn(checking+checking_2+1),
                                                             self.candle.candleReturn(checking+checking_2+2),
                                                             self.candle.candleReturn(checking+checking_2+3))
                            if checking_3 != 0:
                                print('bb')
                                print(self.candle.candleReturn(checking+checking_2+1))
                                print(self.candle.candleReturn(checking+checking_2+2))
                                print(self.candle.candleReturn(checking+checking_2+3))
                                print(checking_3)
                                index = checking+checking_2+checking_3
                                pre_bummping = check_bundle_return_up(self.factor['bumming_cnt'], 
                                                                      self.factor['pre_bummping'], 
                                                                      self.candle.candleReturn(index+1),
                                                                      self.candle.candleReturn(index+2),
                                                                      self.candle.candleReturn(index+3),
                                                                      self.candle.candleReturn(index+4),
                                                                      self.candle.candleReturn(index+5),
                                                                      self.candle.candleReturn(index+6),
                                                                      self.candle.candleReturn(index+7),
                                                                      self.candle.candleReturn(index+8))
                                print(pre_bummping,'bummping',self.candle.candleReturn(index+1)+self.candle.candleReturn(index+2)+self.candle.candleReturn(index+3)+self.candle.candleReturn(index+4)+self.candle.candleReturn(index+5)+self.candle.candleReturn(index+6)+self.candle.candleReturn(index+7)+self.candle.candleReturn(index+8))
                                if pre_bummping == 0:
                                    print('c')
                                    print(self.candle.getDf())
                                    self.break_time = time.time() + self.interval_m * self.factor['break_cnt'] * 60
                                    return {'act':'buy','price':self.candle.getClose(0),'qrt':self.factor['buy_unit'],'time':time.time()}
                                    
            
            if self.break_time <= time.time():
                if sum([1 for i in range(1,6) if self.candle.getVol(i)>=self.factor['vol1']]) >= 1:                    
                    checking = check_bundle_return(self.factor['bundle_cnt'],
                                                   self.factor['return2'],
                                                   self.candle.candleReturn(1),
                                                   self.candle.candleReturn(2),
                                                   self.candle.candleReturn(3))
                    if checking != 0:
                        print('d')
                        print(self.candle.candleReturn(1))
                        print(self.candle.candleReturn(2))
                        print(self.candle.candleReturn(3))
                        print(checking)
                        checking_2 = check_bundle_return(self.factor['bundle_cnt'],
                                                         self.factor['return2'],
                                                         self.candle.candleReturn(checking+1),
                                                         self.candle.candleReturn(checking+2),
                                                         self.candle.candleReturn(checking+3))
                        if checking_2 != 0 :
                            print('e')
                            print(self.candle.candleReturn(checking+1))
                            print(self.candle.candleReturn(checking+2))
                            print(self.candle.candleReturn(checking+3))
                            print(checking)
                            index = pivot+checking+checking_2
                            pre_bummping = check_bundle_return_up(self.factor['bumming_cnt'], 
                                                                  self.factor['pre_bummping'], 
                                                                  self.candle.candleReturn(index+1),
                                                                  self.candle.candleReturn(index+2),
                                                                  self.candle.candleReturn(index+3),
                                                                  self.candle.candleReturn(index+4),
                                                                  self.candle.candleReturn(index+5),
                                                                  self.candle.candleReturn(index+6),
                                                                  self.candle.candleReturn(index+7),
                                                                  self.candle.candleReturn(index+8))
                            print(pre_bummping,'bummping',self.candle.candleReturn(index+1)+self.candle.candleReturn(index+2)+self.candle.candleReturn(index+3)+self.candle.candleReturn(index+4)+self.candle.candleReturn(index+5)+self.candle.candleReturn(index+6)+self.candle.candleReturn(index+7)+self.candle.candleReturn(index+8))
                            if pre_bummping == 0:
                                print('f')
                                print(self.candle.getDf())
                                self.break_time = time.time() + self.interval_m * self.factor['break_cnt'] * 60
                                return {'act':'buy','price':self.candle.getClose(0),'qrt':self.factor['buy_unit'],'time':time.time()}
                                
        
        
        return {'act':'none', 'qrt':0, 'time':time.time()}
    
    def takeProfitSignal(self):
        if self.position.getQty() != 0:
            print('take')
            if (self.candle.getClose(0) - self.position.getAvgPrice()) / self.position.getAvgPrice() >= self.factor['profit_cut']:
                return {'act':'sell','price':self.candle.getClose(0),'qrt':-self.position.getQty(),'time':time.time()}
            else:
                return {'act':'none', 'qrt':0, 'time':time.time()}
        return {'act':'none', 'qrt':0, 'time':time.time()}
    
    def cutLossSignal(self):
        if self.position.getQty() >= self.factor['buy_unit']*2:
            print('cut')
            if (self.candle.getClose(0) - self.position.getAvgPrice()) / self.position.getAvgPrice() <= self.factor['loss_cut']:
                return {'act':'sell','price':self.candle.getClose(0), 'qrt':-self.position.getQty(), 'time':time.time()}
            else:
                return {'act':'none', 'qrt':0, 'time':time.time()}
        return {'act':'none', 'qrt':0, 'time':time.time()}
        
    def shortCutLossSignal(self):
        print('shrtcut')
        if self.position.getQty() != 0:
            if (self.candle.getClose(0) - self.position.getAvgPrice()) / self.position.getAvgPrice() <= self.factor['short_loss_cut']:
                return {'act':'sell','price':self.candle.getClose(0),'qrt':-self.factor['buy_unit'],'time':time.time()}
            else:
                return {'act':'none', 'qrt':0, 'time':time.time()}
        return {'act':'none', 'qrt':0, 'time':time.time()}
            
#%%
from binance.client import Client
client = Client(API_KEY, API_SECRET)

symbol = 'BTCUSDT'

position = Position(symbol,0)
run = ThreeDownInLowStrategy(symbol,factor,position)
while True:
    run.setCandle()
    signal = run.buySignal()
    if signal['act'] == 'buy':
        print('f')
        print(signal)
        client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=signal['qrt'])
        position.addTrade('buy',signal['qrt'],signal['price'])
    elif signal['act'] == 'sell':
        print(signal)
        client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=signal['qrt'])
        position.addTrade('sell',signal['qrt'],signal['price'])
    else:
        signal_loss = run.cutLossSignal()
        signal_profit = run.takeProfitSignal()
        signal_short = run.shortCutLossSignal()
        
        if signal_loss['act'] != 'none':
            client.futures_create_order(symbol=symbol, side=signal_loss['act'].upper(), type='MARKET', quantity=signal_loss['qrt'])
            position.addTrade(signal_loss['act'],signal['qrt'],signal['price'])
            print(1)
        elif signal_profit['act'] != 'none':
            client.futures_create_order(symbol=symbol, side=signal_profit['act'].upper(), type='MARKET', quantity=signal_profit['qrt'])
            position.addTrade(signal_profit['act'],signal_profit['qrt'],signal_profit['price'])
            print(1)
        elif signal_short['act'] != 'none':
            client.futures_create_order(symbol=symbol, side=signal_short['act'].upper(), type='MARKET', quantity=signal_short['qrt'])
            position.addTrade(signal_short['act'],signal_short['qrt'],signal_short['price'])
            print(1)
    run.setPosition(position) 
    time.sleep(2)
    
position.addTrade('buy',0.2,500)
position.getQty()
position.getAvgPrice()
