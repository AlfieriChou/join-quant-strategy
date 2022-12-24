# 标题：价值投资策略_基于线性模型

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 
import jqdata

# 定义一个初始化函数
def initialize(context):
  # 包括相关参数设置的函数
  set_params()
  # 设置回测环境的函数
  set_backtest()
  # 设置每日运营交易
  run_daily(trade, 'every_bar')

# 定义参数设置的函数
def set_params():
  # 定义初始日期为0
  g.days = 0
  # 每5天调仓一次
  g.refresh_rate = 5
  # 最大持股数
  g.stocknum = 3

# 定义回测的函数
def set_backtest():
  # 对比基准
  set_benchmark('000001.XSHG')
  # 开启动态复权模式(真实价格)
  set_option('use_real_price', True)
  # 设置日志记录订单和报错
  log.set_level('order', 'error')

# 设置交易函数
def trade(context):
  # 5天跑一次
  if g.days % 5 == 0:
    t = g.days
    # print(t)
    # 股票池
    stocks = get_index_stocks('000300.XSHG', date = None)
    #stocks
    # 获取股票代码和对应的因子数据
    q = query(
      valuation.code
      # 市值
      ,valuation.market_cap
      # 净资产 = 总-负债
      ,balance.total_assets-balance.total_liability
      # 资产负债率 取倒数
      ,balance.total_assets/balance.total_liability
      # 净利润
      ,income.net_profit
      # 年度收入增长
      ,indicator.inc_revenue_year_on_year
      # 研发费用
      ,balance.development_expenditure
    ).filter(
      valuation.code.in_(stocks)
    )
    # 数据表格化
    df = get_fundamentals(q, date = None)
    # 加表头
    df.columns = ['code', 'market_cap', 'na', '1/DA_ratio', 'net_profit', 'growth', 'RD']
    # 模型训练
    df.index = df['code'].values
    # print(df.index)
    # 删除首行         
    df = df.drop('code', axis = 1)
    df = df.fillna(0)
    X = df.drop('market_cap', axis = 1)
    y = df['market_cap']
    # 0填充表中的空值
    X =  X.fillna(0)
    y = y.fillna(0)
    # 线性拟合
    reg = LinearRegression().fit(X, y)
    #　模型预测值输入预测表内
    predict = pd.DataFrame(
      reg.predict(X),
      index = y.index,
      columns = ['predict_mcap']
    )
    #predict.head()
    predict['mcap'] = df['market_cap']
    diff = predict['mcap'] - predict['predict_mcap']
    diff = pd.DataFrame(diff, index = y.index, columns = ['diff'])
    diff = diff.sort_values(by = 'diff', ascending = True)

    stockset = list(diff.index[:10])
    # 执行交易    
    sell_list = list(context.portfolio.positions.keys())
    #print(sell_list)
    # 卖
    for stock in sell_list:
      if stock not in stockset[:g.stocknum]:
        stock_sell = stock
        # 清仓
        log.info('「卖出股票」：' + str(stock_sell))
        order_target_value(stock_sell, 0)
            
    if len(context.portfolio.positions) < g.stocknum:
      num = g.stocknum - len(context.portfolio.positions)
      cash = context.portfolio.cash / num 
    else:
      cash = 0
      num = 0 
            
    # 买
    for stock in stockset[:g.stocknum]:
      #均仓购入
      if stock in sell_list:
        pass
      else:
        stock_buy = stock
        log.info('「买入股票」：' + str(stock_buy))
        order_target_value(stock_buy, cash)
        num = num - 1
        if num == 0:
          break 
    # 天数+1
    g.days += 1
  else:
    g.days = g.days + 1
