#导入函数库
from jqdata import *
from jqfactor import get_factor_values
import numpy as np
import pandas as pd

#初始化函数 
def initialize(context):
    #设定股票池
    set_benchmark('000300.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option('avoid_future_data', True)
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    #选股参数
    g.stock_num = 3 #持仓数
    g.percentile = 0.25 #选股百分比
    g.buy_mouth = pd.DataFrame(columns = ['date'])
    # 设置交易时间，每天运行
    run_daily(my_trade, time = '9:30', reference_security = '000300.XSHG')
    run_daily(print_trade_info, time = '15:30', reference_security = '000300.XSHG')

#2-1 选股模块
def get_factor_filter_list(context, stock_list, jqfactor, p1, p2):
  yesterday = context.previous_date
  score_list = get_factor_values(
    stock_list,
    jqfactor,
    end_date = yesterday,
    count = 1
  )[jqfactor].iloc[0].tolist()
  df = pd.DataFrame(columns = ['code', 'score'])
  df['code'] = stock_list
  df['score'] = score_list
  df.dropna(inplace = True)
  df.sort_values(by = 'score', ascending = False, inplace = True)
  filter_list = list(df.code)[int(p1*len(stock_list)):int(p2*len(stock_list))]
  return filter_list

# 2-8 过滤业绩预期 305001-业绩大幅上升 305002-业绩预增 305004-预计扭亏 305009-大幅减亏
def filter_by_report(context, stock_list):
  stock = list(
    finance.run_query(
      query(finance.STK_FIN_FORCAST.code)
        .filter(finance.STK_FIN_FORCAST.type_id.in_([305001, 305002, 305004, 305009]
    ),
    finance.STK_FIN_FORCAST.pub_date == context.previous_date,
    finance.STK_FIN_FORCAST.code.in_(stock_list)))['code']
  )
  df = pd.DataFrame(columns = ['code'])
  df['code'] = stock
  df.dropna(inplace = True)
  filter_list = list(df.code)
  return filter_list

#1-2 选股模块
def get_stock_list(context):
  initial_list = get_all_securities().index.tolist()
  initial_list = filter_kcbj_stock(initial_list)
  initial_list = filter_st_stock(initial_list)
  initial_list = filter_new_stock(context, initial_list, 250)
  earnings_growth_list = get_factor_filter_list(context, initial_list, 'earnings_growth', 0, 1)
  sales_growth_list = get_factor_filter_list(context, earnings_growth_list, 'sales_growth', 0, 1)
  q = query(valuation.code).filter(valuation.code.in_(sales_growth_list)).order_by(valuation.circulating_market_cap.desc())
  final_list = list(get_fundamentals(q).code)
  report_list = filter_by_report(context, final_list)
  log.info('「预处理股票」：' + str(report_list))
  return report_list

#2-1 过滤停牌股票
def filter_paused_stock(stock_list):
  current_data = get_current_data()
  return [stock for stock in stock_list if not current_data[stock].paused]

#2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
  current_data = get_current_data()
  return [stock for stock in stock_list
    if not current_data[stock].is_st
    and 'ST' not in current_data[stock].name
    and '*' not in current_data[stock].name
    and '退' not in current_data[stock].name]

#2-3 获取最近N个交易日内有涨停的股票
def get_recent_limit_up_stock(context, stock_list, recent_days):
  stat_date = context.previous_date
  new_list = []
  for stock in stock_list:
    df = get_price(
      stock,
      end_date = stat_date,
      frequency = 'daily',
      fields = ['close', 'high_limit'],
      count = recent_days,
      panel = False,
      fill_paused = False
    )
    df = df[df['close'] == df['high_limit']]
    if len(df) > 0:
      new_list.append(stock)
  return new_list

#2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
  last_prices = history(1, unit = '1m', field = 'close', security_list = stock_list)
  current_data = get_current_data()
  return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
    or last_prices[stock][-1] <  current_data[stock].high_limit]

#2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
  last_prices = history(1, unit = '1m', field = 'close', security_list = stock_list)
  current_data = get_current_data()
  return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
    or last_prices[stock][-1] > current_data[stock].low_limit]

#2-6 过滤科创北交股票
def filter_kcbj_stock(stock_list):
  for stock in stock_list[:]:
    if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68' or stock[:3] == '300':
      stock_list.remove(stock)
  return stock_list

#2-7 过滤次新股
def filter_new_stock(context, stock_list, d):
  yesterday = context.previous_date
  return [stock for stock in stock_list if not yesterday - get_security_info(stock).start_date <  datetime.timedelta(days = d)]

#3-1 交易模块-自定义下单
#报单成功返回报单(不代表一定会成交),否则返回None,应用于
def order_target_value_(security, value):
  if value == 0:
    log.debug("Selling out %s" % (security))
  else:
    log.debug("Order %s to value %f" % (security, value))
  # 如果股票停牌，创建报单会失败，order_target_value 返回None
  # 如果股票涨跌停，创建报单会成功，order_target_value 返回Order，但是报单会取消
  # 部成部撤的报单，聚宽状态是已撤，此时成交量>0，可通过成交量判断是否有成交
  return order_target_value(security, value)

#3-2 交易模块-开仓
#买入指定价值的证券,报单成功并成交(包括全部成交或部分成交,此时成交量大于0)返回True,报单失败或者报单成功但被取消(此时成交量等于0),返回False
def open_position(security, value):
  order = order_target_value_(security, value)
  if order != None and order.filled > 0:
    return True
  return False

#3-3 交易模块-平仓
#卖出指定持仓,报单成功并全部成交返回True，报单失败或者报单成功但被取消(此时成交量等于0),或者报单非全部成交,返回False
def close_position(position):
  security = position.security
  order = order_target_value_(security, 0)  # 可能会因停牌失败
  if order != None:
    if order.status == OrderStatus.held and order.filled == order.amount:
      return True
  return False

#3-4 交易模块-调仓
#当择时信号为买入时开始调仓，输入过滤模块处理后的股票列表，执行交易模块中的开平仓操作
def adjust_position(context, buy_stocks):
  for stock in context.portfolio.positions:
    if stock not in buy_stocks:
      log.info("[%s]已不在应买入列表中" % (stock))
      position = context.portfolio.positions[stock]
      close_position(position)
    else:
      log.info("[%s]已经持有无需重复买入" % (stock))
  # 根据股票数量分仓
  # 此处只根据可用金额平均分配购买，不能保证每个仓位平均分配
  position_count = len(context.portfolio.positions)
  if g.stock_num > position_count:
    value = context.portfolio.cash / (g.stock_num - position_count)
    for stock in buy_stocks:
      if context.portfolio.positions[stock].total_amount == 0:
        if open_position(stock, value):
          if len(context.portfolio.positions) == g.stock_num:
            break

#3-5 交易模块-择时交易
#结合择时模块综合信号进行交易
def my_trade(context):
  #获取选股列表并过滤掉:st,st*,退市,涨停,跌停,停牌
  check_out_list = get_stock_list(context)
  check_out_list = filter_st_stock(check_out_list)
  # check_out_list = filter_limitup_stock(context, check_out_list)
  # check_out_list = filter_limitdown_stock(context, check_out_list)
  check_out_list = filter_paused_stock(check_out_list)
  check_out_list = check_out_list[:g.stock_num]
  print('今日自选股:{}'.format(check_out_list))
  adjust_position(context, check_out_list)

#4-1 复盘模块-打印
#打印每日持仓信息
def print_trade_info(context):
  #打印当天成交记录
  trades = get_trades()
  for _trade in trades.values():
    print('成交记录：'+str(_trade))
  #打印账户信息
  for position in list(context.portfolio.positions.values()):
    securities=position.security
    cost=position.avg_cost
    price=position.price
    ret=100*(price/cost-1)
    value=position.value
    amount=position.total_amount    
    print('代码:{}'.format(securities))
    print('成本价:{}'.format(format(cost,'.2f')))
    print('现价:{}'.format(price))
    print('收益率:{}%'.format(format(ret,'.2f')))
    print('持仓(股):{}'.format(amount))
    print('市值:{}'.format(format(value,'.2f')))
    print('———————————————————————————————————')
  print('———————————————————————————————————————分割线————————————————————————————————————————')