# 导入函数库
import jqdata
import datetime
import pandas as pd
import numpy as np
from jqdata import finance
from jqfactor import get_factor_values

# 初始化函数，设定基准等等
def initialize(context):
  # 设定沪深300作为基准
  set_benchmark('000300.XSHG')
  # 开启动态复权模式(真实价格)
  set_option('use_real_price', True)
  set_params(context)
  # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
  set_order_cost(
    OrderCost(
      close_tax = 0.001,
      open_commission = 0.0003,
      close_commission = 0.0003,
      min_commission = 5
    ),
    type = 'stock'
  )
  # 运行时间
  run_daily(
    before_open,
    time = '08:00',
    reference_security = '000300.XSHG'
  )
  run_daily(
    market_open,
    time = 'open',
    reference_security = '000300.XSHG'
  )

# 全局变量设置
def set_params(context):
  g.stock_list = []  # 股票池
  g.maxnum = 5  # 最大持仓数
  g.lower = -2  # 下限
  g.upper = 1   # 上限
  g.stocks_num = 1
  g.zscore_window = 60  # zscore窗口
  g.ma_window = 20  # 均线窗口
  log.set_level('order', 'error')
  g.year = 1
  g.stocks = []
  g.buy_mouth = pd.DataFrame(columns = ['date'])

# 获取当天买卖股票
def get_buy_sell(context):
  #stock_list = get_index_stocks('000016.XSHG')[:10]
  yesterday = context.current_dt - datetime.timedelta(1)  # 昨天
  count = g.zscore_window + g.ma_window - 1  # 2个窗口数和
  
  data = get_current_data()  # 当前时间数据
  buy, sell = [], []
  if len(context.portfolio.positions) < g.maxnum:
    log.info('buy_stocklist',len(list(g.buy_mouth.index)))
  
  hold = context.portfolio.positions.keys()    
  for code in list(g.buy_mouth.index):
    if data[code].paused:  # 跳过停牌股
      continue
    single_df = get_price(code, end_date = yesterday, fields = 'close', count = count)
    single_df['ma'] = pd.Series.rolling(
      single_df.close,
      window = g.ma_window
    ).mean()  # 均线
    single_df.dropna(inplace = True)
    single_df['sub'] = single_df.close - single_df.ma  # 对差值进行回归
    
    zscore_mean = single_df['sub'].mean();
    zscore_std = single_df['sub'].std()  # 均值和标准差
    try:
      zscore_value = (single_df['sub'][-1] - zscore_mean) / zscore_std  # zscore值
    except:
      zscore_value = 0
    record(zscore = zscore_value)
    record(lower = g.lower)
    record(upper = g.upper)
    
    if zscore_value <= g.lower and code not in hold:  # 买入
      buy.append(code)
    if zscore_value >= g.upper and code in hold:  # 卖出
      sell.append(code)
  
  #买入超过30天卖出
  for st in hold :
    if context.current_dt >= context.portfolio.positions[st].init_time + datetime.timedelta(days = 30):
      sell.append(st)
    if context.portfolio.positions[st].price > context.portfolio.positions[st].avg_cost * 1.3:
      sell.append(st)
    if context.portfolio.positions[st].price < context.portfolio.positions[st].avg_cost * 0.9:
      sell.append(st)
  return buy, sell

## 开盘前运行函数
def before_open(context):
  # year1 =context.previous_date.year-1#去年
  if context.previous_date.year != g.year:###每年选一次股票，选出营收，每股收益5年正
    stocks = list(
      get_all_securities(types = ['stock'], date = context.previous_date).index
    )
    stocks = filter_stock(stocks,context)
    # stocks = get_factor_value(stocks,['earnings_growth','sales_growth'],end_date = context.previous_date,count =1)
    stocks = get_factor_filter_list(context,stocks,'earnings_growth')
    g.stocks = get_factor_filter_list(context,stocks,'sales_growth')
    g.year = context.previous_date.year
    print('one year',len(g.stocks))
  get_report(context)
  g.buy, g.sell = get_buy_sell(context)

#305001	业绩大幅上升 305002	业绩预增 305004	预计扭亏 305009	大幅减亏
def get_report(context):
  stock = list(
    finance.run_query(
      query(finance.STK_FIN_FORCAST.code)
        .filter(finance.STK_FIN_FORCAST.type_id.in_([305001,305002,305004,305009]
    ),
    finance.STK_FIN_FORCAST.pub_date == context.previous_date,
    finance.STK_FIN_FORCAST.code.in_(g.stocks)))['code']
  )
  g.buy_mouth['date'] = g.buy_mouth['date'] + 1
  # print('before',g.buy_mouth)
  g.buy_mouth = g.buy_mouth[g.buy_mouth['date'] <= 30]#一个月内业绩预告，超过就删除
  buy_mouth = pd.DataFrame(columns = ['date'])
  for st in stock:
    price = get_current_data()[st].day_open
    buy_mouth.loc[st] = 1
  # print(buy_mouth)
  g.buy_mouth = g.buy_mouth.append(buy_mouth)
  # print('after',g.buy_mouth)

def get_factor_filter_list(context, stock_list, jqfactor):
  yesterday = context.previous_date
  score_list = get_factor_values(
    stock_list,
    jqfactor,
    end_date = yesterday,
    count = 1
  )[jqfactor].iloc[0].tolist()
  df = pd.DataFrame(columns=['code', 'score'])
  df['code'] = stock_list
  df['score'] = score_list
  df = df.dropna()
  df = df[df['score'] > 0]
  # df.sort_values(by='score', ascending=sort, inplace=True)
  filter_list = list(df.code)
  return filter_list

## 开盘时运行函数
def market_open(context):
  # 先卖
  for code in g.sell:
    order_target(code, 0)
  # 再买
  n = len(g.buy)
  hold = len(context.portfolio.positions)
  if n + hold <= g.maxnum and n > 0 and hold < g.maxnum:
    cash_per_stock = context.portfolio.available_cash / n
  elif hold < g.maxnum:
    cash_per_stock = context.portfolio.available_cash / (g.maxnum - hold)
  else:
    cash_per_stock = 0
  for code in g.buy:
      
    # 未达到最大持仓数
    if hold < g.maxnum:
      # 个股资金
      if cash_per_stock > 3000 and cash_per_stock / get_current_data()[code].last_price > 100:
        order_target_value(code, cash_per_stock)
      else:
        break
  print('buy: %d  sell: %d  hold: %d' % (len(g.buy), len(g.sell), len(context.portfolio.positions)))
  if len(g.buy) > 0:
    log.info('「买入股票」：' + str(g.buy))
  if len(g.sell) > 0:
    log.info('「卖入股票」：' + str(g.sell))

def filter_stock(stock_pool,context):
  stocks = []
  for st in stock_pool:
    if not st.startswith('300') and not st.startswith('688'):
      stn = get_security_info(st, context.current_dt)
      if not 'ST' in stn.display_name and not '*' in stn.display_name and not '*' in stn.display_name:
        stocks.append(st)
  return stocks

## 获取个股流通市值数据
def get_circulating_market_cap(stock_list, context):
  q = query(
    valuation.code,
    valuation.circulating_market_cap
  ).filter(valuation.code.in_(stock_list))
  market_cap = get_fundamentals(q, context.previous_date)
  market_cap.set_index('code', inplace = True)
  return market_cap
  
    
