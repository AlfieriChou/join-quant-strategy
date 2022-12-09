# 导入函数库
from jqdata import *


## 初始化函数，设定要操作的股票、基准等等
def initialize(context):
  # 设定沪深300作为基准
  set_benchmark('000300.XSHG')
  # True为开启动态复权模式，使用真实价格交易
  set_option('use_real_price', True)
  # 打开防未来函数
  set_option("avoid_future_data", True)
  # 设定成交量比例
  set_option('order_volume_ratio', 1)
  # 股票类交易手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
  set_order_cost(OrderCost(
    open_tax = 0,
    close_tax = 0.001,
    open_commission = 0.0003,
    close_commission = 0.0003,
    close_today_commission = 0,
    min_commission = 5
  ), type = 'stock')
  # 持仓数量
  g.stocknum = 10
  # 交易日计时器
  g.days = 5
  # 调仓频率
  g.refresh_rate = 20
  # 运行函数
  run_daily(trade, 'every_bar')
  g.previous_date = context.previous_date

## 选出小市值股票
def check_stocks(context):
  # 设定查询条件
  # 上海市场:XSHG/深圳市场:XSHE/香港市场:XHKG
  CY = '399006.XSHE'
  HS_300 = '000300.XSHG'
  # HS_300 = '000016.XSHG'

  a = 0
  b = 0

  # 获取股票的收盘价
  ah_a = attribute_history(
    CY,
    g.refresh_rate,
    unit = '1d',
    fields = ['open', 'close', 'high', 'low', 'volume', 'money'],
    skip_paused = True,
    df = True,
    fq = 'pre'
  )
  # log.info("CY:" + str(ah_a))
  a = (ah_a["close"][-1] - ah_a["open"][0]) / ah_a["open"][0]

  ah_b = attribute_history(
    HS_300,
    g.refresh_rate,
    unit = '1d',
    fields = ['open', 'close', 'high', 'low', 'volume', 'money'],
    skip_paused = True,
    df = True,
    fq = 'pre'
  )
  # log.info("HS_300:" + str(ah_b))
  b = (ah_b["close"][-1] - ah_b["open"][0]) / ah_b["open"][0]

  log.info("a: {}, b: {}".format(a, b))

  if a <= 0 and b <= 0:
    return []
  elif a >= b:
    buylist = get_small_stocks(context)
    return buylist[:g.stocknum]
  elif a <= b:
    buylist = get_big_stocks(context)
    return buylist[:g.stocknum]
  else:
      return []


## 选出小市值股票
def get_small_stocks(context):
  yesterday = str(context.previous_date)
  # 设定查询条件，按市值升序
  q = query(
    valuation.code,
    valuation.market_cap
  ).filter(
    valuation.market_cap.between(20, 30)
  ).order_by(
    valuation.market_cap.asc()
  )

  # 选出低市值的股票，构成buylist
  df = get_fundamentals(q)
  low_liability_list = list(df.code)
  
  q = query(
    balance.code,
    balance.total_assets, #总资产
    balance.bill_receivable, #应收票据
    balance.account_receivable, #应收账款
    balance.other_receivable, #其他应收款
    balance.good_will, #商誉
    balance.intangible_assets, #无形资产
    balance.inventories, #存货
    balance.constru_in_process, #在建工程
  ).filter(
    balance.code.in_(low_liability_list)
  )
  df = get_fundamentals(q)
  df = df.fillna(0)
  df['bad_assets'] = df.sum(1) - df['total_assets']
  df['ratio'] = df['bad_assets'] / df['total_assets']
  df = df.sort_values(by='ratio')
  proper_receivable_list = list(df.code)[int(0.2*len(list(df.code))):int(0.8*len(list(df.code)))]
  
  df = get_history_fundamentals(
    proper_receivable_list,
    fields = [indicator.code, indicator.roe],
    watch_date = yesterday,
    count = 5,
    interval = '1q'
  )
  df = df.groupby('code').apply(lambda x:x.reset_index()).roe.unstack()
  df['past_average'] = 0.1*df.iloc[:,0] + 0.2*df.iloc[:,1] + 0.3*df.iloc[:,2] + 0.4*df.iloc[:,3]
  df['now_average'] = 0.1*df.iloc[:,1] + 0.2*df.iloc[:,2] + 0.3*df.iloc[:,3] + 0.4*df.iloc[:,4]
  df['delta_average'] = df['now_average'] - df['past_average']
  df.dropna(inplace = True)
  df.sort_values(by = 'delta_average', ascending = False, inplace = True)
  roe_list = list(df.index)[:int(0.1*len(list(df.index)))]
  
  q = query(
    valuation.code,
    valuation.pb_ratio
  ).filter(
    balance.code.in_(roe_list)
  ).order_by(
    valuation.pb_ratio.asc()
  )
  df = get_fundamentals(q)
  df = df[df['pb_ratio']>0]
  pb_list = list(df.code)
  
  buylist = pb_list
  # 过滤停牌股票
  buylist = filter_paused_stock(buylist)
  buylist = filter_new_stock(buylist)
  buylist = filter_kcb_stock(buylist)
  buylist = filter_bjb_stock(buylist)
  buylist = filter_st_stock(buylist)
  return buylist[:g.stocknum]


## 选出大市值股票
def get_big_stocks(context):
  yesterday = str(context.previous_date)
  # 设定查询条件，按市值降序
  q = query(
    valuation.code,
    valuation.market_cap
  ).filter(
    valuation.market_cap.between(1000, 10000)
  ).order_by(
    valuation.market_cap.desc()
  )

  # 选出低市值的股票，构成buylist
  df = get_fundamentals(q)
  low_liability_list = list(df.code)
  
  q = query(
    balance.code,
    balance.total_assets, #总资产
    balance.bill_receivable, #应收票据
    balance.account_receivable, #应收账款
    balance.other_receivable, #其他应收款
    balance.good_will, #商誉
    balance.intangible_assets, #无形资产
    balance.inventories, #存货
    balance.constru_in_process, #在建工程
  ).filter(
    balance.code.in_(low_liability_list)
  )
  df = get_fundamentals(q)
  df = df.fillna(0)
  df['bad_assets'] = df.sum(1) - df['total_assets']
  df['ratio'] = df['bad_assets'] / df['total_assets']
  df = df.sort_values(by='ratio')
  proper_receivable_list = list(df.code)[int(0.2*len(list(df.code))):int(0.8*len(list(df.code)))]
  
  df = get_history_fundamentals(proper_receivable_list, fields=[indicator.code, indicator.roe], watch_date=yesterday, count=5, interval='1q')
  df = df.groupby('code').apply(lambda x:x.reset_index()).roe.unstack()
  df['past_average'] = 0.1*df.iloc[:,0] + 0.2*df.iloc[:,1] + 0.3*df.iloc[:,2] + 0.4*df.iloc[:,3]
  df['now_average'] = 0.1*df.iloc[:,1] + 0.2*df.iloc[:,2] + 0.3*df.iloc[:,3] + 0.4*df.iloc[:,4]
  df['delta_average'] = df['now_average'] - df['past_average']
  df.dropna(inplace = True)
  df.sort_values(by = 'delta_average', ascending = False, inplace = True)
  roe_list = list(df.index)[:int(0.1*len(list(df.index)))]
  
  q = query(valuation.code, valuation.pb_ratio).filter(balance.code.in_(roe_list)).order_by(valuation.pb_ratio.asc())
  df = get_fundamentals(q)
  df = df[df['pb_ratio']>0]
  pb_list = list(df.code)
  
  buylist = pb_list
  # 过滤停牌股票
  buylist = filter_paused_stock(buylist)
  buylist = filter_new_stock(buylist)
  buylist = filter_kcb_stock(buylist)
  buylist = filter_bjb_stock(buylist)
  buylist = filter_st_stock(buylist)
  return buylist[:g.stocknum]


## 交易函数
def trade(context):
  if g.days % g.refresh_rate == 0:

    ## 获取持仓列表
    sell_list = list(context.portfolio.positions.keys())
    # 如果有持仓，则卖出
    if len(sell_list) > 0:
      for stock in sell_list:
        order_target_value(stock, 0)

    ## 分配资金
    if len(context.portfolio.positions) < g.stocknum:
      Num = g.stocknum - len(context.portfolio.positions)
      Cash = context.portfolio.cash / Num
    else:
      Cash = 0

    ## 选股
    stock_list = check_stocks(context)

    ## 买入股票
    for stock in stock_list:
      if len(context.portfolio.positions.keys()) < g.stocknum:
        order_value(stock, Cash)

    # 天计数加一
    g.days = 1
  else:
    g.days += 1


# 过滤停牌股票
def filter_paused_stock(stock_list):
  current_data = get_current_data()
  return [stock for stock in stock_list if not current_data[stock].paused]

#过滤科创板
def filter_kcb_stock(stock_list):
  return [stock for stock in stock_list  if stock[0:3] != '688']

#过滤北交所
def filter_bjb_stock(stock_list):
  return [stock for stock in stock_list  if stock[0:1] != '8']

#过滤次新股
def filter_new_stock(stock_list):
  yesterday = g.previous_date
  return [stock for stock in stock_list if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=250)]

#过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
  current_data = get_current_data()
  return [stock for stock in stock_list
    if not current_data[stock].is_st
    and 'ST' not in current_data[stock].name
    and '*' not in current_data[stock].name
    and '退' not in current_data[stock].name]
