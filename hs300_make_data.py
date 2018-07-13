# -*- coding: utf-8 -*-

import pandas as pd
import tushare as ts
import numpy as np
import copy
import time
import os
import json as js

def getOpts():
    opts = {}
    opts['datadir'] = './Data/HS300_total/'
    opts['trade_index'] = './data_trade_index.xls'
    opts['data_type'] = '/daily_data'
    opts['datadest_excel'] = opts['datadir'] + '/data_excel'
    opts['datadest_record'] = opts['datadir'] + opts['data_type'] + '/data_record_20'
    # opts['start_time'] = '1990-12-19'
    opts['start_time'] = '2005-01-01'
    opts['end_time'] = '2018-06-12'

    # trade file
    if os.path.exists(opts['trade_index']):
        Date_index = pd.read_excel(opts['trade_index'], sheet_name='sheet1')
    else:
        raise Exception('No trade day index file')
    opts['Date_index'] = Date_index

    return opts
# opts['datadir'] = 'K:/ZJU/Projects/DLstock/code/Data/hs300/data_excel/'
# opts['datadest_excel'] = 'K:/ZJU/Projects/Dlstock/code/Data/hs300/week_data/data_record_20/'
# hs_300 = pd.read_excel(opts['datadir'] + 'hs300index.xls', 'sheet1', na_values='NA')

# opts['datadir'] = 'M:/Dlstock/data/HS300_total/'
# opts['datadest_excel'] = opts['datadir'] + '/data_excel/'

# 旧版本
def from_D_data_get_20d_data(opts):
    Date_index = opts['Date_index']
    docs_num = 25
    date_length = 20
    for i in range(docs_num):
        sheet1 = pd.read_excel(opts['datadest_excel'] + '/hs300_{}_D.xls'.format(i+1), 'sheet1')
        new_sheet1 = sheet1.loc[:,['code','sheet']]
        writer = pd.ExcelWriter(opts['datadest_excel'] + '/hs300_{}_20D_noflag.xls'.format(i+1))
        new_sheet1.to_excel(writer, sheet_name = 'sheet1')
        for j in range(len(sheet1)):
            stock_code = '{:0>6}'.format(sheet1.iloc[j]['code'])
            stock_index = sheet1.iloc[j]['sheet']
            stock_data = pd.read_excel(opts['datadest_excel'] + '/hs300_{}_D.xls'.format(i+1), str(stock_index))
            # stock_data_20d = pd.DataFrame(columns=['date', 'open', 'close', 'high', 'low', 'volume', 'code', 'flag_no_lost_data'])
            stock_data_20d = pd.DataFrame(columns=['date', 'open', 'close', 'high', 'low', 'volume', 'code'])
            data_length = len(stock_data) - date_length + 1
            # new_data_series = []
            for k in range(data_length):
                date_start = stock_data.iloc[k]['date']
                date_end = stock_data.iloc[k + date_length - 1]['date']
                date = date_start + ',' + date_end
                open = stock_data.iloc[k]['open']
                close = stock_data.iloc[k + date_length - 1]['close']
                high = np.max(np.array(stock_data.loc[k:k+date_length-1, 'high']))
                low = np.min(np.array(stock_data.loc[k:k+date_length-1, 'low']))
                volume = np.sum(np.array(stock_data.loc[k:k+date_length-1, 'volume']))
                code = stock_code

                # if np.where(Date_index == date_start)[0] + 19 == np.where(Date_index == date_end)[0]:
                #     flag_no_lost_data = 1
                # else:
                #     flag_no_lost_data = 0
                # new_data_series = [date, open, close, high, low, volume, code, flag_no_lost_data]
                # stock_data_20d.loc[k] = new_data_series

                if np.where(Date_index == date_start)[0] + 19 == np.where(Date_index == date_end)[0]:
                    new_data_series = [date, open, close, high, low, volume, code]
                    stock_data_20d.loc[k] = new_data_series
            stock_data_20d.to_excel(writer, sheet_name = str(j + 1))
            print('{}/{} done of doc {}'.format(j+1, len(sheet1), i+1))
        writer.save()
        writer.close()

def from_D_data_get_20d_data_with_date_index(opts):
    Date_index = opts['Date_index']
    Date_index = np.array(Date_index['calendarDate'])
    docs_num = 25
    date_length = 20
    for i in range(docs_num):
        sheet1 = pd.read_excel(opts['datadest_excel'] + '/hs300_{}_D.xls'.format(i+1), 'sheet1')
        new_sheet1 = sheet1.loc[:,['code','sheet']]
        for j in range(len(sheet1)):
            new_sheet1.loc[j, 'code'] = '{:0>6}'.format(sheet1.loc[j, 'code'])
        writer = pd.ExcelWriter(opts['datadest_excel'] + '/hs300_{}_20D_date_flag.xls'.format(i+1))
        new_sheet1.to_excel(writer, sheet_name = 'sheet1')
        for j in range(len(sheet1)):
            stock_code = '{:0>6}'.format(sheet1.iloc[j]['code'])
            stock_index = sheet1.iloc[j]['sheet']
            stock_data = pd.read_excel(opts['datadest_excel'] + '/hs300_{}_D.xls'.format(i+1), str(stock_index))
            # stock_data_20d = pd.DataFrame(columns=['date', 'open', 'close', 'high', 'low', 'volume', 'code', 'no_invalid_data', '20D_periods'])

            data_length = len(stock_data) - date_length + 1
            stock_date_index = np.array(stock_data['date'])
            the_first_date = stock_date_index[0]
            the_last_date = stock_date_index[-1]
            the_first_date_index = np.where(Date_index == the_first_date)[0][0]
            the_end_date_index = np.where(Date_index == the_last_date)[0][0] - date_length + 1
            stock_data_20d = pd.DataFrame(index=[k for k in range(the_end_date_index - the_first_date_index + 1)],
                columns=['date', 'open', 'close', 'high', 'low', 'volume', 'code', 'no_invalid_data', '20D_periods'])
            # stock_data_20d.index = [k for k in range(the_end_date_index - the_first_date_index + 1)]
            month_period = {}
            valid_data_count = 0
            for k in range(20):
                month_period[k] = the_first_date_index + k
            # new_data_series = []
            for key in month_period.keys():
                # start_date_index = month_period[key]
                current_index = month_period[key]
                while current_index <= the_end_date_index:
                    date_start = Date_index[current_index]
                    date_end = Date_index[current_index + 19]
                    valid_data_list = np.arange(len(stock_data))[np.array(stock_date_index >= date_start).astype(np.bool) & np.array(stock_date_index <= date_end).astype(np.bool)]
                    valid_data_len = len(valid_data_list)
                    if valid_data_len:
                        valid_data_count += 1
                        date_name = date_start + ',' + date_end
                        if valid_data_len > 20:
                            assert Exception('Error in original data')
                        elif valid_data_len == 20:
                            flag_no_invalid_data = 1
                        else:
                            flag_no_invalid_data = 0
                        valid_data_list.sort()
                        open_price = stock_data.iloc[valid_data_list[0]]['open']
                        close_price = stock_data.iloc[valid_data_list[-1]]['close']
                        high = np.max(np.array(stock_data['high'])[valid_data_list])
                        low = np.min(np.array(stock_data['low'])[valid_data_list])
                        volume = np.sum(np.array(stock_data['volume'])[valid_data_list])
                        code = stock_code
                        stock_data_20d.iloc[current_index - the_first_date_index] = [date_name, open_price, close_price, high, low, volume, code, flag_no_invalid_data, key]
                    else:
                        stock_data_20d.iloc[current_index - the_first_date_index] = ['0', 0, 0, 0, 0, 0, '0', 0, 0]
                    current_index += date_length

            code_data_list = np.array(stock_data_20d['code'])
            invalid_index_list = np.array(stock_data_20d.index)[code_data_list == '0']
                    
            stock_data_20d.drop(invalid_index_list, axis=0, inplace = True)
            stock_data_20d.index = [m for m in range(len(stock_data_20d))]
            assert len(stock_data_20d) == valid_data_count
            stock_data_20d.to_excel(writer, sheet_name = str(j + 1))
            print('{}/{} done of doc {}'.format(j+1, len(sheet1), i+1))
        writer.save()
        writer.close()



def data_for_every_stock(hs300, data_type, opts):
    stock_list = hs300
    for i in range(len(stock_list)):
        stock_name = '{:0>6}'.format(stock_list[i])
        stock_dir = os.path.join(opts['datadest_record'], stock_name)
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)
        stock_data = ts.get_k_data(stock_name, start=opts['start_time'], end=opts['end_time'], ktype=data_type, autype='qfq')
        # time.sleep(0.5)
        stock_data.to_excel(stock_dir + '/data.xls', sheet_name='sheet1')
        # stock_data.to_csv(stock_dir + '/data.csv')
        print('{}/{} done'.format(i, len(stock_list)))



def make_raw_data(hs_300, data_type, opts):
    subset = 30
    hs_300_length = len(hs_300)
    subset_num = [hs_300_length // subset if hs_300_length % subset == 0 else hs_300_length // subset + 1][0]

    if not os.path.exists(opts['datadest_excel']):
        os.makedirs(opts['datadest_excel'])

    for i in range(subset_num):
        hs_code = hs_300[i*subset:np.min((hs_300_length, (i+1)*subset))]
        hs_len = len(hs_code)
        hs_index = [j for j in range(hs_len)]
        hs300 = pd.DataFrame(columns=['code', 'sheet', 'amount'])

        hs300['sheet'] = [j+1 for j in range(hs_len)]
        hs300['code'] = ['{:0>6}'.format(hs_code[j]) for j in range(hs_len)]
        writer = pd.ExcelWriter(opts['datadest_excel'] + 'hs300_{}_{}.xls'.format(i + 1, data_type))
        # for i in range(hs_len):
        #     hs300['sheet'][i] = i + 1
        # hs300.to_excel(writer, '0')

        # 获取沪深300股票十年间的数据
        st_dict = {}
        st_num = []
        for index in range(hs_len):
            st_code = '{:0>6}'.format(hs_code[index])
            st_data = ts.get_k_data(st_code, start=opts['start_time'], end=opts['end_time'], ktype=data_type, autype='qfq')
            st_dict[index] = st_data
            st_num.append(len(st_data))
            # time.sleep(0.5)
            print('{}/{} done'.format(i*subset + index, hs_300_length))

        # 表格0中增加amount列
        # for j in range(hs_len):
        #     hs300['amount'][j] = st_num[j]
        hs300['amount'] = [st_num[j] for j in range(hs_len)]
        hs300.to_excel(writer, sheet_name='sheet1')

        for index in range(hs_len):
            st_data = st_dict[index]
            st_data.to_excel(writer, sheet_name=str(index + 1))
        print('Load down')
        writer.save()
        writer.close()



# for i in range(subset_num):
#     hs300 = copy.copy(hs_300[j*subset:(j+1)*subset])
#     hs_index = [i for i in range(len(hs300))]
#     hs300.index = hs_index
#     hs_code = hs300['code']
#     hs_name = hs300['name']
#     hs_len = len(hs_code)
#     # 重新建立表格数据，表格0中增加sheet列
#     writer = pd.ExcelWriter(opts['datadir'] + 'hs300data_{}_{}.xls'.format(i+1, data_type))
#     sh_index = pd.DataFrame([j+1 for j in range(hs_len)], columns=['sheet'])
#     hs300['sheet'] = sh_index
#     # for i in range(hs_len):
#     #     hs300['sheet'][i] = i + 1
#     # hs300.to_excel(writer, '0')
#
#     # 获取沪深300股票十年间的数据
#     st_dict = {}
#     st_num = []
#     for index in range(hs_len):
#         st_code = '{:0>6}'.format(hs_code[index])
#         start_time = '2008-01-01'
#         end_time = '2018-01-01'
#         st_data = ts.get_k_data(st_code, start=start_time, end=end_time, ktype=data_type, autype='qfq')
#         st_dict[index] = st_data
#         st_num.append(len(st_data))
#         # time.sleep(0.5)
#         print(st_code, 'done')
#
#     # 表格0中增加amount列
#     st_amount = pd.DataFrame([j+1 for j in range(hs_len)], columns=['amount'])
#     hs300['amount'] = st_amount
#     for j in range(hs_len):
#         hs300['amount'][j] = st_num[j]
#     hs300.to_excel(writer, sheet_name='sheet1')
#
#     for index in range(hs_len):
#         st_data = st_dict[index]
#         st_data.to_excel(writer, sheet_name=str(index+1))
#     print('Load down')
#     writer.save()
#     writer.close()

if __name__ == '__main__':
    opts = getOpts()

    # from_D_data_get_20d_data(opts)
    from_D_data_get_20d_data_with_date_index(opts)

    with open(opts['datadir'] + 'hs300_code_total.json', 'r') as f:
        hs_300 = js.load(f)
    hs_300 = np.array(hs_300)

    data_type = 'D' # 日K
    # data_type = 'M'  # 月K

    # make_raw_data(hs_300, data_type, opts)

    data_for_every_stock(hs_300, data_type, opts)

