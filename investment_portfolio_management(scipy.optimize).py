#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle
import datetime
import warnings
warnings.filterwarnings('ignore')

with open('invest_portfolio.pkl','rb') as pf:
    daily_profit = pickle.load(pf)
columns = daily_profit.columns
date_list = list(daily_profit.index)
start_date = date_list[0]
end_date = date_list[-1]

#获取日期列表
def get_tradeday_list(start,end,frequency=None):
    '''
    input:
    start:str or datetime,起始时间，与count二选一
    end:str or datetime，终止时间
    frequency:
        str: day,month,quarter,halfyear,默认为day
        int:间隔天数
    count:int,与start二选一，默认使用start
    '''
    df = daily_profit.iloc[:,:2]
    if frequency == None or frequency =='day':
        days = df.index
    else:
        df['year-month'] = [str(i)[0:7] for i in df.index]
        if frequency == 'month':
            days = df.drop_duplicates('year-month').index
        elif frequency == 'quarter':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='04') | (df['month']=='07') | (df['month']=='10') ]
            days = df.drop_duplicates('year-month').index
        elif frequency =='halfyear':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='06')]
            days = df.drop_duplicates('year-month').index
    if isinstance(days[0],str):
        return list(days)
    else:
        trade_days = [datetime.datetime.strftime(i,'%Y-%m-%d') for i in days]
        return trade_days
date_list_str = get_tradeday_list(start_date,end_date,frequency='day')
daily_profit.index = date_list_str
trade_days = get_tradeday_list(start_date,end_date,frequency='quarter')
trade_days_month = get_tradeday_list(start_date,end_date,frequency='month')
daily_profit = daily_profit.loc[:,columns]


def portfolio_optimizer_function(daily_profit, date, target, count=250, bounds=(0.0, 1), default_port_weight_range=[0.99, 1.0], ftol=1e-9, rf=0.04):
    '''
    input:
    daily_profit:dataframe,index为时间，columns为投资组合的标的代码，values为每日收益率
    date:优化发生的日期，也就是调仓时间
    target: 优化目标函数，只能选择一个，共五个可选 'mean_weights','mean_vol','min_var','risk_parity','max_sharpe'
    count:向前计算的数据长度，默认250，过去一年的数据
    bounds: 边界函数，用以对组合中单标的权重进行限制，可设置一个或多个相同/不同类别的函数。如果不填，默认为 Bound(0., 1.)；\
    如果有多个 bound，则每只股票取对应的边界值
    default_port_weight_range: 长度为2的列表，默认的组合权重之和的范围，默认值为 [0.0, 1.0]。如果限制函数(constraints) 中没有 WeightConstraint 或 WeightEqualConstraint 限制，则会添加 WeightConstraint(low=default_port_weight_range[0], high=default_port_weight_range[1]) 到 constraints列表中。
    ftol: 默认为 1e-9，优化函数触发结束的函数值。当求解结果精度不够时可以适当降低 ftol 的值，当求解时间过长时可以适当提高 ftol 值
    rf:计算夏普比率时用到的无风险利率，默认0.04
    '''
    date_list = list(daily_profit.index)
    start_date = date_list[0]
    end_date = date_list[-1]
    date_list_str = get_tradeday_list(start_date, end_date, frequency='day')
    end_date_index = date_list_str.index(date)
    if end_date_index < 250:
        print('data not enouph to current date')
        return None
    else:
        start_date_index = end_date_index - 250
    pct_temp = daily_profit.iloc[start_date_index:end_date_index]
    cov_mat = pct_temp.cov()
    omega = np.matrix(cov_mat.values)

    # 初始值
    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    # 每个变量的取值范围
    if isinstance(bounds, tuple):
        bnds = tuple(bounds for x in x0)
    elif isinstance(bounds, list):
        bnds = bounds
    else:
        print('error:please input correct bounds')
        return None

    cons = ({'type': 'ineq', 'fun': lambda x: sum(x) - default_port_weight_range[0]}, \
            {'type': 'ineq', 'fun': lambda x: -sum(x) + default_port_weight_range[1]})
    options = {'disp': False, 'maxiter': 1000, 'ftol': ftol}

    if target == 'mean_weights':
        weights = np.array([1.0 / pct_temp.shape[1]] * pct_temp.shape[1])
        return weights
    elif target == 'mean_vol':
        wts = 1 / pct_temp.std()
        weights = wts / wts.sum()
        return weights.values
    elif target == 'min_var':
        # 最小方差
        def min_var(x):
            risk = np.dot(x.T, np.dot(cov_mat, x))
            return risk

        weights = minimize(min_var, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)['x']
        return weights
    elif target == 'risk_parity':
        # 风险平价
        def risk_parity(x):
            tmp = (omega * np.matrix(x).T).A1
            z = np.sqrt(np.matrix(x) * omega * np.matrix(x).T).A1[0]
            risk = x * tmp / z
            delta_risk = [sum((i - risk) ** 2) for i in risk]
            return sum(delta_risk)

        weights = minimize(risk_parity, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)['x']
        return weights
    elif target == 'max_sharpe':
        # 最大夏普
        def max_sharpe(x):
            r_b = rf
            pct_mean = pct_temp.mean()
            p_r = (1 + np.dot(x, pct_mean)) ** 250 - 1
            p_sigma = np.sqrt(np.dot(x.T, np.dot(cov_mat, x)) * 250)
            p_sharpe = (p_r - r_b) / p_sigma
            return -p_sharpe

        weights = minimize(max_sharpe, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)['x']
        return weights


weights = portfolio_optimizer_function(daily_profit, '2018-02-05', target='mean_weights')


def cal_portfolio_profit(daily_profit, trade_days, target):
    '''
    daily_profit:dataframe,index为时间，columns为投资组合的标的代码，values为每日收益率
    target: 优化目标函数，只能选择一个，共五个可选 'mean_weights','mean_vol','min_var','risk_parity','max_sharpe'
    trade_days:list,交易时间列表
    '''
    l = []
    for i in range(len(trade_days) - 1):
        weights = portfolio_optimizer_function(daily_profit, trade_days[i], target)
        profit_cut = (daily_profit.loc[trade_days[i]:trade_days[i + 1]] * weights)
        profit_sum = profit_cut.cumsum(axis=1).iloc[:, -1]
        profit = (profit_sum + 1).cumprod().values[-1] - 1
        l.append(profit)
    df = pd.DataFrame(l, columns=[target])
    df[target + '_cumprod'] = (df[target] + 1).cumprod()
    return df

# profit = cal_portfolio_profit(daily_profit,trade_days_month[-30:],'mean_weights')

weights_name = [ 'mean_weights','mean_vol','min_var','risk_parity','max_sharpe']
l = []
for name in weights_name:
    profit = cal_portfolio_profit(daily_profit,trade_days_month[-30:],name)
    profit_cumprod = profit[name+'_cumprod']
    l.append(profit_cumprod)
profit_df = pd.concat(l,axis=1)
profit_df.plot(figsize=(15,7))
plt.show()