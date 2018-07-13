# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import time
import re
from sklearn.metrics.pairwise import cosine_similarity as cosine
import pandas as pd
from igraph import Graph as IGraph
import igraph as ig
import louvain
import scipy.io as scio




def getOpts(opts):
    opts['randomSeed'] = 2
    opts['inputImageSize'] = 224
    opts['trainEpoches'] = 100
    opts['alldata'] = 572638  # all data 57397
    opts['databatch'] = 10
    opts['trainWeightDecay'] = 5e-04  # 5e-04
    opts['stddev'] = 0.1
    opts['database_candle'] = './data/data_candlestick'
    opts['database_mat'] = './data/data_mat'
    opts['choose_num'] = 2
    opts['tradeday'] = '2017-06-01'
    opts['path2save'] = './outputs/'
    opts['output'] =  opts['path2save']+ opts['tradeday'] + '/'



    opts['metafile'] = './model/model_epoch.ckpt.meta'
    opts['modelname'] = './model/model_epoch.ckpt'

    return opts


def plot_images(image1, image2, num):

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(2, num)
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):

        if i < num :
            # Plot image.
            ax.imshow(image1[i,:,:,:])
            xlabel = 'Input{}'.format(i)
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.imshow(image2[i-num,:,:,:])
            xlabel = 'Reconstruction{}'.format(i-num)
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
    # plt.show()

def GetRandBatch(datalist, batch, opts):
    # VGG_MEAN = [103.939, 116.779, 123.68]
    choose = [datalist[i] for i in batch]
    lens = len(batch)
    imagebatch = np.zeros([lens, opts['inputImageSize'], opts['inputImageSize'], 3], dtype=np.float32)
    for i in range(lens):
        img = cv2.imread(choose[i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[:, 55:540, :]
        img = cv2.resize(img, (opts['inputImageSize'], opts['inputImageSize'])) / 255
        imagebatch[i, :, :, :] = img
    bgrbatch = imagebatch
    return bgrbatch

def TradeDay(opts):
    tradedays = ts.trade_cal()
    tradedays = tradedays[tradedays.isOpen == 1]
    tradedays.index = [i for i in range(len(tradedays))]
    start_index = tradedays[tradedays.calendarDate == opts['tradeday']].index
    tradeday_end = tradedays.iloc[start_index + 19].calendarDate.values.tolist()[0]
    tradeday_test_start = tradedays.iloc[start_index + 20].calendarDate.values.tolist()[0]

    return tradeday_end, tradeday_test_start

def GetStocks(opts):
    if os.path.exists(opts['output']+'index_stocks_for_{}.txt'.format(opts['tradeday'])):
        with open(opts['output']+'index_stocks_for_{}.txt'.format(opts['tradeday']), 'r') as f:
            stocks_lists = f.read().split('\n')
        stocks_lists = np.array(stocks_lists)
        null_index = np.where(stocks_lists == '')
        stocks_lists = np.delete(stocks_lists, null_index)
        return stocks_lists.tolist(), []
    stocks_lists = []
    stocks_mat_lists = []
    stocks_filespath = os.listdir(opts['database_candle'])
    stocks_matpath = os.listdir(opts['database_mat'])
    stocks_mat_collect = []
    # # 先取100支股票
    # stocks_filespath = stocks_filespath[0:50]
    stocks_num = len(stocks_filespath)
    start_day = opts['tradeday']
    end_day, _ = TradeDay(opts)
    for i in range(stocks_num):
        # st_name = os.path.join(opts['database_candle'],stocks_filespath[i])
        # st_mat_name = os.path.join(opts['database_mat'], stocks_matpath[i])
        st_name = opts['database_candle'] + '/' + stocks_filespath[i]
        st_mat_name = opts['database_mat'] + '/' + stocks_matpath[i]
        st_list = os.listdir(st_name)
        st_mat_list = os.listdir(st_mat_name)
        st_index = ''
        st_choose = ''
        for j in range(len(st_list)):
            index = st_list[j].find(start_day)
            if index == 0 and st_list[j].find(end_day) != -1:
               st_index = st_list[j]
               st_mat_index = st_mat_list[j]
               stocks_mat_collect.append(stocks_matpath[i])
               break
        if st_index == '':
            print(st_name)
            continue # 直接舍弃无数据的股票
        # assert st_index != ''
        st_choose = os.path.join(st_name, st_index)
        st_mat_choose = os.path.join(st_mat_name, st_mat_index)
        stocks_lists.append(st_choose)
        stocks_mat_lists.append(st_mat_choose)
    # assert len(stocks_lists) == stocks_num
    print(len(stocks_lists))
    with open(opts['output']+'index_stocks_for_{}.txt'.format(opts['tradeday']), 'w') as f:
        for i in range(len(stocks_lists)):
            f.write(stocks_lists[i] + '\n')
    with open(opts['output']+'index_stocks_mat_for_{}.txt'.format(opts['tradeday']), 'w') as f:
        for i in range(len(stocks_mat_lists)):
            f.write(stocks_mat_lists[i] + '\n')

    sharpe_ratio_df = pd.DataFrame(columns=['codes', 'sharpes', 'return_20days'])
    sharpe_ratio_df['codes'] = stocks_mat_collect
    sharpes = []
    return_during_20days = []
    for i in range(len(stocks_mat_lists)):
        data = scio.loadmat(stocks_mat_lists[i])
        data = np.array(data['Prices'])
        data = (data[:,3] - data[:,0]) / data[:,0]
        # sharpe_ratio
        data_mean = np.mean(data)
        data_std = np.std(data)
        sharpe_ratio = data_mean / data_std
        sharpes.append(sharpe_ratio)
        # 20 day return
        data = data.tolist()
        data_str = ','.join(str(data[j]) for j in range(len(data)))
        return_during_20days.append(data_str)
    sharpe_ratio_df['sharpes'] = sharpes
    sharpe_ratio_df['return_20days'] = return_during_20days
    sharpe_ratio_df.to_excel(opts['output']+'sharpe_ratios_for_{}.xls'.format(opts['tradeday']), sheet_name='sheet1')

    return stocks_lists, stocks_mat_lists

def get_feature_from_vgg(datalist, opts):
    datalen = len(datalist)
    dataindex = [i for i in range(datalen)]
    # batch_num = datalen // opts['databatch']
    datafeatures = np.ones(shape=[datalen, 512])
    imagecollection = np.zeros(shape=[datalen, 224, 224, 3])
    re_imgcollection = np.zeros(shape=[datalen, 224, 224, 3])

    # 加载模型
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if opts['metafile']:
        saver = tf.train.import_meta_graph(opts['metafile'])
        saver.restore(sess, opts['modelname'])
    else:
        raise Exception('No history model file found!')
    graph = tf.get_default_graph()


    inputImage = graph.get_tensor_by_name('inputImage:0')
    reconstruction = graph.get_tensor_by_name('reconstruction:0')
    ave_pooling = graph.get_tensor_by_name('new_encoder_net/ave_pooling:0')
    # lossOp = graph.get_tensor_by_name('Loss/lossOp:0')
    # global_step = graph.get_tensor_by_name('global_step:0')

    # 获取数据
    sampleNum = 0
    while sampleNum < datalen:
        batch = np.min([opts['databatch'], datalen - sampleNum])
        batch_index = dataindex[sampleNum : sampleNum + batch]
        imagebatch = GetRandBatch(datalist, batch_index, opts)
        # imgweight = np.ones(np.shape(imagebatch))
        feature, outputs = sess.run([ave_pooling,reconstruction], feed_dict={inputImage: imagebatch})
        feature = sess.run(ave_pooling, feed_dict={inputImage: imagebatch})
        datafeatures[sampleNum : sampleNum + batch, :] = feature
        imagecollection[sampleNum : sampleNum + batch, :,:,:] = imagebatch
        re_imgcollection[sampleNum : sampleNum + batch, :, :, :] = outputs
        sampleNum += batch

    # name_index = time.strftime('%H_%M_%S', time.localtime(time.time()))
    plt.figure(1)
    plot_images(imagecollection, re_imgcollection, 5)
    plt.savefig(opts['output']+'rebuild_for_{}'.format(opts['tradeday']))
    plt.close()
    # 数据处理
    # np.savetxt一般把array数组保存为txt
    np.savetxt(opts['output']+'features_for_{}.txt'.format(opts['tradeday']), datafeatures, delimiter=',')
    return datafeatures

def get_excel_graph_model(datafeatures, opts):
    # datafeatures = np.loadtxt(opts['output']+'features_for_{}.txt'.format(opts['tradeday']), delimiter = ',')
    stock_num = np.shape(datafeatures)[0]
    with open(opts['output']+'index_stocks_for_{}.txt'.format(opts['tradeday']), 'r') as f:
        stock_index = f.readlines()
    stock_name = []
    for i in range(len(stock_index)):
        find_code = False
        sl = re.split(r'[\/\\]+', stock_index[i])
        for j in range(len(sl)):
            if re.match(r'\d{6}', sl[j]):
                stock_name.append(sl[j])
                find_code = True
        assert find_code
    assert len(stock_name) == stock_num

    cos_similarity = cosine(datafeatures)
    # np.savetxt('./output/cosine_similarity_for_{}.txt'.format(opts['tradeday']), cos_similarity, delimiter=',')
    feature_ungraph = pd.DataFrame(columns=['vertex1', 'vertex2', 'weights'])
    vertex1 = []
    vertex2 = []
    weights = []
    for i in list(range(stock_num))[0:-1]:
        for j in list(range(stock_num))[i+1:]:
            vertex1.append(stock_name[i])
            vertex2.append(stock_name[j])
            weights.append(cos_similarity[i, j])
    feature_ungraph['vertex1'] = vertex1
    feature_ungraph['vertex2'] = vertex2
    feature_ungraph['weights'] = weights
    feature_ungraph.to_excel(opts['output']+'feature_graph_for_{}.xls'.format(opts['tradeday']), sheet_name='sheet1')

    return feature_ungraph

def modularity_analysis(feature_graph, opts):

    # feature_ungraph = pd.read_excel(opts['output']+'feature_graph_for_{}.xls'.format(opts['tradeday']),
    #                                 sheet_name='sheet1')
    # 网络生成
    length = len(feature_graph)
    edges = []
    edge_weights = []
    for i in range(length):
        # tuple(节点1, 节点2, 权值)
        edges.append(tuple(['{:0>6}'.format(int(feature_graph.loc[i][0])),
                            '{:0>6}'.format(int(feature_graph.loc[i][1])),
                            feature_graph.loc[i][2]]))
        edge_weights.append(feature_graph.loc[i][2])

    graph = IGraph.TupleList(edges=edges, directed=False, weights=True)

    modularity_graph = louvain.find_partition(graph, louvain.ModularityVertexPartition, weights=edge_weights)

    # vertex_count = graph.vcount()
    graph.vs['label'] = graph.vs['name']

    mode_num = len(modularity_graph)
    modularity_index = np.zeros(len(graph.vs['name']),)
    for i in range(mode_num):
        modularity_index[modularity_graph[i]] = i
    modularity_index = list(map(int, modularity_index))
    graph.vs['modularity'] = modularity_index
    mode_list = []
    for i in range(mode_num):
        vertexes = graph.vs.select(modularity=i)
        one_mode_list = [vertexes[j]['name'] for j in range(len(vertexes))]
        mode_list.append(one_mode_list)
    with open(opts['output'] + 'modularity_for_{}.txt'.format(opts['tradeday']), 'w') as f:
        for i in range(mode_num):
            f.write(','.join(mode_list[i]) + '\n')

    # 作图
    color_dict = {0:'red', 1:'green', 2:'blue'}
    graph.vs['color'] = [color_dict[index] for index in graph.vs['modularity']]
    ig.plot(graph)

    return graph, modularity_graph

def choose_stocks_from_modularity(opts):
    assert os.path.exists(opts['output'] + 'modularity_for_{}.txt'.format(opts['tradeday']))
    with open(opts['output'] + 'modularity_for_{}.txt'.format(opts['tradeday']), 'r') as f:
        modularity = f.read()
        modularity = modularity.strip('\n').split('\n')

    if not os.path.exists(opts['output'] + 'sharpe_ratios_for_{}.xls'.format(opts['tradeday'])):
        _, _ = GetStocks(opts)

    stocks_sharpe_ratio = pd.read_excel(opts['output'] + 'sharpe_ratios_for_{}.xls'.format(opts['tradeday']),
                                        sheet_name='sheet1')

    stocks_dict = {}
    for i in range(len(modularity)):
        stock_df = pd.DataFrame(columns=['codes', 'sharpes'])
        stock_list = modularity[i].split(',')
        stock_sharpes = np.array([])
        for j in range(len(stock_list)):
            sharpe = stocks_sharpe_ratio[stocks_sharpe_ratio.codes == int(stock_list[j])].sharpes.values
            stock_sharpes = np.append(stock_sharpes, sharpe)
        stock_df['codes'] = stock_list
        stock_df['sharpes'] = stock_sharpes.tolist()
        stocks_dict[str(i)] = stock_df

    choose_dict = {}
    for i in range(len(modularity)):
        stock_sharpes = stocks_dict[str(i)].sort_values(by= 'sharpes', ascending=False)
        choose_dict[str(i)] = list(stock_sharpes[0:opts['choose_num']].codes.values)

    return choose_dict

def check_modularity_effect(opts):
    assert os.path.exists(opts['output'] + 'modularity_for_{}.txt'.format(opts['tradeday']))
    with open(opts['output'] + 'modularity_for_{}.txt'.format(opts['tradeday']), 'r') as f:
        modularity = f.read()
        modularity = modularity.strip('\n').split('\n')

    if not os.path.exists(opts['output'] + 'sharpe_ratios_for_{}.xls'.format(opts['tradeday'])):
        _, _ = GetStocks(opts)

    stocks_sharpe_ratio = pd.read_excel(opts['output'] + 'sharpe_ratios_for_{}.xls'.format(opts['tradeday']),
                                        sheet_name='sheet1')

    modularity_list = []
    for i in range(len(modularity)):
        stock_list = modularity[i].split(',')
        stocks_list = []
        for j in range(len(stock_list)):
            return_value = stocks_sharpe_ratio[stocks_sharpe_ratio.codes == int(stock_list[j])].return_20days.values
            stocks_list.append(return_value)
        modularity_list.append(stocks_list)

    for i in range(len(modularity_list)):
        modularity_stock = modularity_list[i]
        for j in range(len(modularity_stock)):
            x_data = [k for k in range(21)]
            y_data = modularity_stock[j].tolist()[0].split(',')
            y_data = [float(y_data[k]) for k in range(len(y_data))]
            y_data = np.array([0] + y_data) + 1
            y_data = np.cumprod(y_data)
            plt.plot(x_data, y_data)
            plt.title('20days-return-for-the-{}th-modularity'.format(i))
            plt.xlabel('days')
            plt.ylabel('cumulative-return')
            plt.savefig(opts['output'] + '{}th_return_for_{}'.format(i, opts['tradeday']))
        plt.close()

def check_effect_next_10days(opts):
    assert os.path.exists(opts['output'] + 'modularity_for_{}.txt'.format(opts['tradeday']))
    with open(opts['output'] + 'modularity_for_{}.txt'.format(opts['tradeday']), 'r') as f:
        modularity = f.read()
        modularity = modularity.strip('\n').split('\n')

    print('训练数据开始于：{}'.format(opts['tradeday']))
    _, test_start_day = TradeDay(opts)
    # test next 10 days
    opts['tradeday'] = test_start_day
    if not os.path.exists(opts['output'] + 'sharpe_ratios_for_{}.xls'.format(opts['tradeday'])):
        _, _ = GetStocks(opts)
    stocks_sharpe_ratio = pd.read_excel(opts['output'] + 'sharpe_ratios_for_{}.xls'.format(opts['tradeday']),
                                        sheet_name='sheet1')
    print('测试数据开始于：{}'.format(opts['tradeday']))

    modularity_list = []
    for i in range(len(modularity)):
        stock_list = modularity[i].split(',')
        stocks_list = []
        for j in range(len(stock_list)):
            return_value = stocks_sharpe_ratio[stocks_sharpe_ratio.codes == int(stock_list[j])].return_20days.values
            if return_value:
                stocks_list.append(return_value)
            else:
                print('no complete data for {}'.format(stock_list[j]))
        modularity_list.append(stocks_list)

    for i in range(len(modularity_list)):
        modularity_stock = modularity_list[i]
        for j in range(len(modularity_stock)):
            x_data = [k for k in range(11)]
            y_data = modularity_stock[j].tolist()[0].split(',')
            y_data = [float(y_data[k]) for k in range(len(y_data))]
            y_data = y_data[0:10]
            y_data = np.array([0] + y_data) + 1
            y_data = np.cumprod(y_data)
            plt.plot(x_data, y_data)
            plt.title('20days-return-for-the-{}th-modularity'.format(i))
            plt.xlabel('days')
            plt.ylabel('cumulative-return')
            plt.savefig(opts['output'] + 'test_{}th_return_for_{}'.format(i, opts['tradeday']))
        plt.close()


def main(_):
    opts = {}
    opts = getOpts(opts)
    if  not os.path.exists(opts['output']):
        os.makedirs(opts['output'])
    # t1 = time.time()
    datalist, _ = GetStocks(opts)
    # print('获取数据用时：{}s'.format(time.time()-t1))

    # 卷积自编码器生成特征
    # datafeatures = get_feature_from_vgg(datalist, opts)
    # datafeatures = np.loadtxt(opts['output'] + 'features_for_{}.txt'.format(opts['tradeday']), delimiter=',')
    # feature_graph = get_excel_graph_model(datafeatures, opts)
    feature_graph = pd.read_excel(opts['output']+'feature_graph_for_{}.xls'.format(opts['tradeday']), sheet_name='sheet1')

    # 模块化聚类分析
    graph, modularity =modularity_analysis(feature_graph, opts)
    # # #
    choose_stocks = choose_stocks_from_modularity(opts)
    print(choose_stocks)
    # #
    # 测试
    # check_modularity_effect(opts)

    # check_effect_next_10days(opts)

if __name__ == '__main__':
    tf.app.run()