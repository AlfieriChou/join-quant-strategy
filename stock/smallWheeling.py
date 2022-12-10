#导入函数库
from jqdata import *
from jqfactor import get_factor_values
import numpy as np
import pandas as pd

#初始化函数 
def initialize(context):
  # 设定沪深300作为基准
  set_benchmark('000300.XSHG')
  # 用真实价格交易
  set_option('use_real_price', True)
  # 打开防未来函数
  set_option("avoid_future_data", True)
  # 将滑点设置为0
  set_slippage(FixedSlippage(0))
  # 设置交易成本万分之三
  set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0003, close_commission=0.0003, close_today_commission=0, min_commission=5),
                  type='fund')
  # 过滤order中低于error级别的日志
  log.set_level('order', 'error')

  g.rsrs_score = []
  #动量轮动参数
  g.momentum_day = 30 #最新动量参考最近momentum_day的
    #初始化全局变量
  g.stock_num = 30
  g.limit_up_list = []
  g.hold_list = []
  g.weights = [1.0, 1.0, 1.6, 0.8, 2.0]
  g.high_limit_list = []
  #rsrs择时参数
  g.ref_stock = '000300.XSHG' #用ref_stock做择时计算的基础数据
  g.N = 18 # 计算最新斜率slope，拟合度r2参考最近N天
  g.M = 600 # 计算最新标准分zscore，rsrs_score参考最近M天
  g.score_threshold = 0.7 # rsrs标准分指标阈值
  g.slope_series = initial_slope_series()[:-1] # 除去回测第一天的slope，避免运行时重复加入
  #ma择时参数
  g.mean_day = 20 #计算mean_day的ma
  g.mean_diff_day = 3 #计算mean_diff_day前的g.mean_diff_day的ma
  g.industry_control = True
  g.industry_filter_list = [
    '钢铁I', '煤炭I', '石油石化I', '采掘I', #重资产
    '银行I', '非银金融I', '金融服务I', #高负债
    '交运设备I', '交通运输I', '传媒I', '环保I' #盈利差
  ]
  # 设置交易时间，每天运行
  run_daily(prepare_stock_list, time='9:05', reference_security='000300.XSHG')
  run_weekly(weekly_adjustment, weekday=1, time='9:30', reference_security='000300.XSHG')
  # run_daily(my_trade, time='9:30', reference_security='000300.XSHG')
  run_daily(print_trade_info, time='15:00', reference_security='000300.XSHG')
  run_daily(check_limit_up, time='14:00', reference_security='000300.XSHG') #检查持仓中的涨停股是否需要卖出

    
#1-1 计算线性回归统计值
#对输入的自变量每日最低价x(series)和因变量每日最高价y(series)建立OLS回归模型,返回元组(截距,斜率,拟合度)
def get_ols(x, y):
  slope, intercept = np.polyfit(x, y, 1)
  r2 = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
  return (intercept, slope, r2)

#1-2 设定初始斜率序列
#通过前M日最高最低价的线性回归计算初始的斜率,返回斜率的列表
def initial_slope_series():
  data = attribute_history(g.ref_stock, g.N + g.M, '1d', ['high', 'low'])
  return [get_ols(data.low[i:i+g.N], data.high[i:i+g.N])[1] for i in range(g.M)]

#1-3 计算标准分
#通过斜率列表计算并返回截至回测结束日的最新标准分
def get_zscore(slope_series):
  mean = np.mean(slope_series)
  std = np.std(slope_series)
  return (slope_series[-1] - mean) / std

#1-4 计算综合信号
#获得rsrs与MA信号，信号至少一个为True时返回调仓信号，同为False时返回卖出信号
def get_timing_signal(stock):
  #计算MA信号
  close_data = attribute_history(g.ref_stock, g.mean_day + g.mean_diff_day, '1d', ['close'])
  today_MA = close_data.close[g.mean_diff_day:].mean() 
  before_MA = close_data.close[:-g.mean_diff_day].mean()
  print('MA差值={}'.format(format(today_MA-before_MA,'.2f')))
  #计算rsrs信号
  high_low_data = attribute_history(g.ref_stock, g.N, '1d', ['high', 'low'])
  intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
  g.slope_series.append(slope)
  g.rsrs_score.append(get_zscore(g.slope_series[-g.M:]) * r2)
  if len(g.rsrs_score)>10:
    g.rsrs_score.remove(g.rsrs_score[0])
  print('rsrs_score={}'.format(format(g.rsrs_score[-1],'.2f')))
  #综合判断所有信号
  if g.rsrs_score[-1] > g.score_threshold or today_MA > before_MA:
    return "BUY"
  elif g.rsrs_score[-1] < -g.score_threshold and today_MA < before_MA:
    return "SELL"



#2-1 根据动量判断市场风格
#基于指数年化收益和判定系数打分,并按照分数从大到小排名
def get_index_signal(context):
  close_data = attribute_history(g.ref_stock, 11, '1d', ['close'])
  today_MA = close_data.close[-1]
  before_MA = close_data.close[0:-1].mean()
  high_low_data = attribute_history(g.ref_stock, g.N, '1d', ['high', 'low'])
  intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
  g.slope_series.append(slope)
  
  g.rsrs_score.append(get_zscore(g.slope_series[-g.M:]) * r2)
  if len(g.rsrs_score)>3:
    g.rsrs_score.remove(g.rsrs_score[0])
  rsrs_mean = mean(g.rsrs_score)
  if (rsrs_mean>0.4 and  today_MA > before_MA*(1+0.03)) or len(g.rsrs_score)<4:
    best_index = 1 
  else:
    best_index = 0
  record(score = best_index)
  return best_index

#2-2 聚宽因子选股
#输入股票列表，要查询的聚宽因子，排序方式，选股比例，返回选股后的列表
def get_factor_filter_list(context, stock_list, jqfactor, sort, p1, p2):
  yesterday = context.previous_date
  score_list = get_factor_values(stock_list, jqfactor, end_date=yesterday, count=1)[jqfactor].iloc[0].tolist()
  df = pd.DataFrame(columns=['code','score'])
  df['code'] = stock_list
  df['score'] = score_list
  df = df.dropna()
  df = df[df['score']>0]
  df.sort_values(by='score', ascending=sort, inplace=True)
  filter_list = list(df.code)[int(p1*len(stock_list)):int(p2*len(stock_list))]
  return filter_list


#1-2 选股模块
def get_stock_list0(context):#740策略
    
  # 获取前N个单位时间当时的收盘价
  def get_close(stock, n, unit):
    return attribute_history(stock, n, unit, 'close')['close'][0]
  
  # 获取现价相对N个单位前价格的涨幅
  def get_return(stock, n, unit):
    price_before = attribute_history(stock, n, unit, 'close')['close'][0]
    price_now = get_close(stock, 1, '1m')
    if not isnan(price_now) and not isnan(price_before) and price_before != 0:
      return price_now / price_before
    else:
      return 100
    
  # 获得初始列表
  yesterday = context.previous_date
  initial_list = get_all_securities('stock', yesterday).index.tolist()
  initial_list = filter_kcbj_stock(initial_list)
  initial_list = filter_new_stock(context, initial_list)
  initial_list = filter_st_stock(initial_list)
  q = query(
    valuation.code,
    valuation.market_cap,
    valuation.circulating_market_cap
  ).filter(
    valuation.code.in_(initial_list),
    valuation.pb_ratio > 0,
    indicator.inc_return > 0,
    indicator.inc_total_revenue_year_on_year > 0,
    indicator.inc_net_profit_year_on_year > 0
  ).order_by(
    valuation.market_cap.asc()).limit(100)
  df = get_fundamentals(q, date=yesterday)
  df.index = df.code
  initial_list = list(df.index)
    
  #获取原始值
  MC, CMC, PN, TV, RE = [], [], [], [], []
  for stock in initial_list:
    #总市值
    mc = df.loc[stock]['market_cap']
    MC.append(mc)
    #流通市值
    cmc = df.loc[stock]['circulating_market_cap']
    CMC.append(cmc)
    #当前价格
    pricenow = get_close(stock, 1, '1m')
    PN.append(pricenow)
    #5日累计成交量
    total_volume_n = attribute_history(stock, 1200, '1m', 'volume')['volume'].sum()
    TV.append(total_volume_n)
    #60日涨幅
    m_days_return = get_return(stock, 60, '1d') 
    RE.append(m_days_return)
  #合并数据
  df = pd.DataFrame(
    index = initial_list,
    columns = ['market_cap', 'circulating_market_cap', 'price_now', 'total_volume_n', 'm_days_return']
  )
  df['market_cap'] = MC
  df['circulating_market_cap'] = CMC
  df['price_now'] = PN
  df['total_volume_n'] = TV
  df['m_days_return'] = RE
  df = df.dropna()
  min0, min1, min2, min3, min4 = min(MC), min(CMC), min(PN), min(TV), min(RE)
  #计算合成因子
  temp_list = []
  for i in range(len(list(df.index))):
    score = g.weights[0] * math.log(min0 / df.iloc[i,0]) + g.weights[1] * math.log(min1 / df.iloc[i,1]) + g.weights[2] * math.log(min2 / df.iloc[i,2]) + g.weights[3] * math.log(min3 / df.iloc[i,3]) + g.weights[4] * math.log(min4 / df.iloc[i,4])
    temp_list.append(score)
  df['score'] = temp_list
  
  #排序并返回最终选股列表
  df = df.sort_values(by='score', ascending=False)
  final_list = list(df.index)[0:30]
  return final_list

#1-2 选股模块
def get_stock_list1(context):#wywyzhe
  yesterday = str(context.previous_date)    
  initial_list = get_all_securities().index.tolist()
  initial_list = filter_new_stock(context,initial_list)
  initial_list = filter_kcb_stock(context, initial_list)
  initial_list = filter_st_stock(initial_list)
  initial_list = filter_paused_stock(initial_list)
  #SG 5年营业收入增长率
  sg_list = get_factor_filter_list(context, initial_list, 'sales_growth', False, 0, 0.1)
  q = query(
    valuation.code,
    valuation.circulating_market_cap,
    indicator.eps
  ).filter(
    valuation.code.in_(sg_list)
  ).order_by(
    valuation.circulating_market_cap.asc()
  )
  df = get_fundamentals(q, date = yesterday)
  df = df[df['eps'] > 0]
  sg_list = list(df.code)
  #MS
  factor_values = get_factor_values(initial_list, [
    'operating_revenue_growth_rate', #营业收入增长率
    'total_profit_growth_rate', #利润总额增长率
    'net_profit_growth_rate', #净利润增长率
    'earnings_growth', #5年盈利增长率
  ], end_date = yesterday, count = 1)
  df = pd.DataFrame(index = initial_list, columns = factor_values.keys())
  df['operating_revenue_growth_rate'] = list(factor_values['operating_revenue_growth_rate'].T.iloc[:,0])
  df['total_profit_growth_rate'] = list(factor_values['total_profit_growth_rate'].T.iloc[:,0])
  df['net_profit_growth_rate'] = list(factor_values['net_profit_growth_rate'].T.iloc[:,0])
  df['earnings_growth'] = list(factor_values['earnings_growth'].T.iloc[:,0])
  df['total_score'] = 0.1*df['operating_revenue_growth_rate'] + 0.35*df['total_profit_growth_rate'] + 0.15*df['net_profit_growth_rate'] + 0.4*df['earnings_growth']
  df = df.sort_values(by=['total_score'], ascending=False)
  complex_growth_list = list(df.index)[:int(0.1*len(list(df.index)))]
  q = query(valuation.code,valuation.circulating_market_cap,indicator.eps).filter(valuation.code.in_(complex_growth_list)).order_by(valuation.circulating_market_cap.asc())
  df = get_fundamentals(q)
  df = df[df['eps']>0]
  ms_list = list(df.code)
  #PEG
  peg_list = get_factor_filter_list(context, initial_list, 'PEG', True, 0, 0.2)
  turnover_list = get_factor_filter_list(context, peg_list, 'turnover_volatility', True, 0, 0.5)
  q = query(valuation.code,valuation.circulating_market_cap,indicator.eps).filter(valuation.code.in_(turnover_list)).order_by(valuation.circulating_market_cap.asc())
  df = get_fundamentals(q, date=yesterday)
  peg_list = list(df.code)
    #PB过滤
  q = query(valuation.code, valuation.pb_ratio, indicator.eps).filter(valuation.code.in_(initial_list)).order_by(valuation.pb_ratio.asc())
  df = get_fundamentals(q)
  df = df[df['eps']>0]
  df = df[df['pb_ratio']>0]
  pb_list = list(df.code)[:int(0.5*len(df.code))]
  #ROEC过滤
  #因为get_history_fundamentals有返回数据限制最多5000行，需要把pb_list拆分后查询再组合
  interval = 1000 #count=5时，一组最多1000个，组数向下取整
  pb_len = len(pb_list)
  if pb_len <= interval:
    df = get_history_fundamentals(pb_list, fields=[indicator.code, indicator.roe], watch_date=yesterday, count=5, interval='1q')
  else:
    df_num = pb_len // interval
    df = get_history_fundamentals(pb_list[:interval], fields=[indicator.code, indicator.roe], watch_date=yesterday, count=5, interval='1q')
    for i in range(df_num):
      dfi = get_history_fundamentals(pb_list[interval*(i+1):min(pb_len,interval*(i+2))], fields=[indicator.code, indicator.roe], watch_date=yesterday, count=5, interval='1q')
      df = df.append(dfi)
  df = df.groupby('code').apply(lambda x:x.reset_index()).roe.unstack()
  df['increase'] = 4*df.iloc[:,4] - df.iloc[:,0] - df.iloc[:,1] - df.iloc[:,2] - df.iloc[:,3]
  df.dropna(inplace=True)
  df.sort_values(by='increase',ascending=False, inplace=True)
  temp_list = list(df.index)
  temp_len = len(temp_list)
  roe_list = temp_list[:int(0.1*temp_len)]
  #行业过滤
  if g.industry_control == True:
    industry_df = get_stock_industry(roe_list, yesterday)
    ROE_list = filter_industry(industry_df, g.industry_filter_list)
  else:
    ROE_list = roe_list
  #市值排序
  q = query(valuation.code,valuation.circulating_market_cap).filter(valuation.code.in_(ROE_list)).order_by(valuation.circulating_market_cap.asc())
  df = get_fundamentals(q)
  ROEC_list = list(df.code)
  
  final_list = [sg_list, ms_list, peg_list,ROEC_list]
  sg_list = final_list[0][:5]
  ms_list = final_list[1][:5]
  peg_list = final_list[2][:5]
  ROEC_list = final_list[3][:10]
  union_list = list(set(sg_list).union(set(ms_list)).union(set(peg_list)).union(set(ROEC_list)))
  return union_list


#1-6 调整昨日涨停股票
def check_limit_up(context):
  now_time = context.current_dt
  high_low_data = attribute_history(g.ref_stock, g.N, '1d', ['high', 'low'])
  intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
  g.slope_series.append(slope)
  g.rsrs_score.append(get_zscore(g.slope_series[-g.M:]) * r2)
  if len(g.rsrs_score)>6:
    g.rsrs_score.remove(g.rsrs_score[0])
  if g.high_limit_list != []:
    #对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
    for stock in g.high_limit_list:
      current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close','high_limit'], skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
      if current_data.iloc[0,0] < current_data.iloc[0,1]:
        log.info("[%s]涨停打开，卖出" % (stock))
        position = context.portfolio.positions[stock]
        close_position(position)
      else:
        log.info("[%s]涨停，继续持有" % (stock))

#1-1 准备股票池
def prepare_stock_list(context):
  #获取已持有列表
  g.hold_list= []
  for position in list(context.portfolio.positions.values()):
    stock = position.security
    g.hold_list.append(stock)
  #获取昨日涨停列表
  if g.hold_list != []:
    df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close','high_limit'], count=1, panel=False, fill_paused=False)
    df = df[df['close'] == df['high_limit']]
    g.high_limit_list = list(df.code)
  else:
    g.high_limit_list = []  

#1-4 整体调整持仓
def weekly_adjustment(context):
  #获取应买入列表
  #获取选股列表并过滤掉:st,st*,退市,涨停,跌停,停牌
  index_signal = get_index_signal(context)
  if index_signal == 1:
    stock_list = get_stock_list1(context)
      
  elif index_signal == 0:
    stock_list = get_stock_list0(context)
  stock_list = filter_limitup_stock(context, stock_list)
  stock_list = filter_limitdown_stock(context, stock_list)
  target_list = filter_paused_stock(stock_list)
  
  record(target_len = len(target_list))
  # hold_list = context.portfolio.position.keys()
  #调仓卖出
  for stock in g.hold_list:
    if (stock not in target_list) and (stock not in g.high_limit_list):
      log.info("卖出[%s]" % (stock))
      position = context.portfolio.positions[stock]
      close_position(position)
    else:
      log.info("已持有[%s]" % (stock))
  #调仓买入
  position_count = len(context.portfolio.positions)
  target_num = len(target_list)
  if target_num > position_count:
    value = context.portfolio.cash / (target_num - position_count)
    for stock in target_list:
      if context.portfolio.positions[stock].total_amount == 0:
        if open_position(stock, value):
          if len(context.portfolio.positions) == target_num:
            break

#3-1 过滤停牌股票
#输入选股列表，返回剔除停牌股票后的列表
def filter_paused_stock(stock_list):
	current_data = get_current_data()
	return [stock for stock in stock_list if not current_data[stock].paused]

#2-0 行业过滤函数
def get_stock_industry(securities, watch_date, level='sw_l1', method='industry_name'): 
  industry_dict = get_industry(securities, watch_date)
  industry_ser = pd.Series({k: v.get(level, {method: np.nan})[method] for k, v in industry_dict.items()})
  industry_df = industry_ser.to_frame('industry')
  return industry_df

def filter_industry(industry_df, select_industry, level='sw_l1', method='industry_name'):
  filter_df = industry_df.query('industry != @select_industry')
  filter_list = filter_df.index.tolist()
  return filter_list

#3-2 过滤ST及其他具有退市标签的股票
#输入选股列表，返回剔除ST及其他具有退市标签股票后的列表
def filter_st_stock(stock_list):
	current_data = get_current_data()
	return [stock for stock in stock_list
    if not current_data[stock].is_st
    and 'ST' not in current_data[stock].name
    and '*' not in current_data[stock].name
    and '退' not in current_data[stock].name]

#2-6 过滤科创北交股票
def filter_kcbj_stock(stock_list):
  for stock in stock_list[:]:
    if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68' or stock[:2] == '30':
      stock_list.remove(stock)
  return stock_list

#3-3 过滤涨停的股票
#输入选股列表，返回剔除未持有且已涨停股票后的列表
def filter_limitup_stock(context, stock_list):
	last_prices = history(1, unit='1m', field='close', security_list=stock_list)
	current_data = get_current_data()
	# 已存在于持仓的股票即使涨停也不过滤，避免此股票再次可买，但因被过滤而导致选择别的股票
	return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
		or last_prices[stock][-1] < current_data[stock].high_limit]

#3-4 过滤跌停的股票
#输入股票列表，返回剔除已跌停股票后的列表
def filter_limitdown_stock(context, stock_list):
	last_prices = history(1, unit='1m', field='close', security_list=stock_list)
	current_data = get_current_data()
	return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
		or last_prices[stock][-1] > current_data[stock].low_limit]

#3-5 过滤科创板
#输入股票列表，返回剔除科创板后的列表
def filter_kcb_stock(context, stock_list):
  return [stock for stock in stock_list  if stock[0:3] != '688']

#3-6 过滤次新股
#输入股票列表，返回剔除上市日期不足250日股票后的列表
def filter_new_stock(context,stock_list):
  yesterday = context.previous_date
  return [stock for stock in stock_list if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=250)]

#4-1 自定义下单
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

#4-2 开仓
#买入指定价值的证券,报单成功并成交(包括全部成交或部分成交,此时成交量大于0)返回True,报单失败或者报单成功但被取消(此时成交量等于0),返回False
def open_position(security, value):
	order = order_target_value_(security, value)
	if order != None and order.filled > 0:
		return True
	return False

#4-3 平仓
#卖出指定持仓,报单成功并全部成交返回True，报单失败或者报单成功但被取消(此时成交量等于0),或者报单非全部成交,返回False
def close_position(position):
	security = position.security
	order = order_target_value_(security, 0)  # 可能会因停牌失败
	if order != None:
		if order.status == OrderStatus.held and order.filled == order.amount:
			return True
	return False

#4-4 调仓
#当择时信号为买入时开始调仓，输入过滤模块处理后的股票列表，执行交易模块中的开平仓操作
def adjust_position(context, buy_stocks, stock_num):
	for stock in context.portfolio.positions:
		if stock not in buy_stocks:
			log.info("[%s]已不在应买入列表中" % (stock))
			position = context.portfolio.positions[stock]
			close_position(position)
		else:
			log.info("[%s]已经持有无需重复买入" % (stock))
	# 根据股票数量分仓
	# 此处只根据可用金额平均分配购买，不能保证每个仓位平均分配302
	
	position_count = len(context.portfolio.positions)
	if stock_num > position_count:
		value = context.portfolio.cash / (stock_num - position_count)
		for stock in buy_stocks:
			if context.portfolio.positions[stock].total_amount == 0:
				if open_position(stock, value):
					if len(context.portfolio.positions) == stock_num:
						break

#5-1 复盘模块-打印
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
    print('————————————————————————————————————')
  print('———————————————————————————————————————分割线————————————————————————————————————————')
