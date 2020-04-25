#!coding:utf-8

# ******************预处理阶段1：生成词典*****************
import os
import pickle as pkl
import time

from gensim import corpora, models
from pandas import np
from scipy.sparse import csr_matrix
from sklearn import metrics
import xgboost as xgb
from FileLoader.LoadF import loadFiles
from FileLoader.SegDocs import seg_doc


# 产生存储词典的对象
def GeneDict(path_doc_root, path_dictionary):
    # 数据预处理阶段1：生成词典并去掉低频率项，如果词典不存在则重新生成。反之跳过该阶段
    print('=== 未检测到有词典存在，开始遍历生成词典 ===')
    dictionary = corpora.Dictionary()
    files = loadFiles(path_doc_root)
    for i, msg in enumerate(files):
        #       if i % n == 0:
        catg = msg[0]
        content = seg_doc(msg[1])  # 对文本内容分词处理
        dictionary.add_documents([content])
    # 去掉词典中出现次数过少的词
    small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5]
    dictionary.filter_tokens(small_freq_ids)
    # 重新产生连续的编号
    dictionary.compactify()
    dictionary.save(path_dictionary)
    print('=== 词典已经生成 ===')


# ******************预处理阶段2：生成TFIDF*****************


def GeneTFIDF(path_doc_root, path_dictionary, path_tmp_tfidf):
    print('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
    dictionary = corpora.Dictionary.load(path_dictionary)
    os.makedirs(path_tmp_tfidf)
    files = loadFiles(path_doc_root)
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    corpus_tfidf = {}
    # 开始向量化处理
    for i, msg in enumerate(files):
        catg = msg[0]
        word_list = seg_doc(msg[1])
        file_bow = dictionary.doc2bow(word_list)
        file_tfidf = tfidf_model[file_bow]
        tmp = corpus_tfidf.get(catg,[])
        tmp.append(file_tfidf)
        if tmp.__len__() == 1:
            corpus_tfidf[catg] = tmp
    # 将tfidf中间结果储存起来，即文本类别列表
    catgs = list(corpus_tfidf.keys())
    for catg in catgs:
        corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg), corpus_tfidf.get(catg),
                                   id2word=dictionary)
        print('catg {c} has been transformed into tfidf vector'.format(c=catg))
    print('=== tfidf向量已经生成 ===')


# ******************预处理阶段3：TFIDF向量模型生成LSI主题模型*****************
'''
num_topics： 设置保存权重最大的前N个数据特征，默认300
path_tmp_lsimodel： lsi模型保存路径 
'''


def GeneLSI(path_dictionary, path_tmp_tfidf, path_tmp_lsimodel, path_tmp_lsi, num_topics=300):
    print('=== 未检测到有lsi文件夹存在，开始生成lsi向量 ===')
    dictionary = corpora.Dictionary.load(path_dictionary)
    # 从对应文件夹中读取所有类别
    catg_list = []
    for file in os.listdir(path_tmp_tfidf):
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)

    # 从磁盘中读取corpus
    corpus_tfidf = {}
    for catg in catg_list:
        path = '{f}{s}{c}.mm'.format(f=path_tmp_tfidf, s=os.sep, c=catg)
        corpus = corpora.MmCorpus(path)
        corpus_tfidf[catg] = corpus
    print('tfidf文档读取完毕，开始转化成lsi向量 ...')

    # 生成lsi model
    corpus_tfidf_total = []
    for catg in list(corpus_tfidf.keys()):
        tmp = corpus_tfidf.get(catg)
        corpus_tfidf_total += tmp
    lsi_model = models.LsiModel(corpus=corpus_tfidf_total, id2word=dictionary, num_topics=num_topics)

    # 将lsi模型存储到磁盘上
    lsi_file = open(path_tmp_lsimodel, 'wb')
    pkl.dump(lsi_model, lsi_file)
    lsi_file.close()
    del corpus_tfidf_total  # lsi model已经生成，释放变量空间
    print('--- lsi模型已经生成 ---')

    # 生成corpus of lsi, 并逐步去掉 corpus of tfidf
    corpus_lsi = {}
    for catg in list(corpus_tfidf.keys()):
        corpu = [lsi_model[doc] for doc in corpus_tfidf.get(catg)]
        corpus_lsi[catg] = corpu
        corpus_tfidf.pop(catg)
        corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_lsi, s=os.sep, c=catg), corpu, id2word=dictionary)
    print('=== lsi向量已经生成 ===')


#######################生成分类器阶段######################################
'''
path_tmp_lsi 各类主题mm保存的父目录
path_tmp_predictor： 分类器模型保存路径
train_test_ratio: 训练集与测试集划分比例，默认70%训练集和30%测试集
'''


def GeneClassifier(path_tmp_lsi, path_tmp_predictor, train_test_ratio=0.7):
    print('=== 未检测到分类器存在，开始进行分类过程 ===')
    print('--- 未检测到lsi文档，开始从磁盘中读取 ---')
    catg_list = []
    for file in os.listdir(path_tmp_lsi):
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)

    # 从磁盘中读取corpus
    corpus_lsi = {}
    for catg in catg_list:
        path = '{f}{s}{c}.mm'.format(f=path_tmp_lsi, s=os.sep, c=catg)
        corpus = corpora.MmCorpus(path)
        corpus_lsi[catg] = corpus
    print('--- lsi文档读取完毕，开始进行分类 ---')
    # 类别标签、文档数、 语料主题
    tag_list, doc_num_list, corpus_lsi_total = [], [], []
    for count, catg in enumerate(catg_list):
        tmp = corpus_lsi[catg]
        tag_list += [count] * tmp.__len__()
        doc_num_list.append(tmp.__len__())
        corpus_lsi_total += tmp
        corpus_lsi.pop(catg)
    print("文档类别数目:", len(doc_num_list))
    # 将gensim中的mm表示转化成numpy矩阵表示
    print("LSI语料总大小:", len(corpus_lsi_total))

    data, rows, cols = [], [], []
    line_count = 0
    for line in corpus_lsi_total:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    lsi_matrix = csr_matrix((data, (rows, cols))).toarray()
    print("LSI矩阵规模:", lsi_matrix.shape)
    print("数据样本数目:", line_count)
    # 生成训练集和测试集
    rarray = np.random.random(size=line_count)
    train_set, train_tag, test_set, test_tag = [], [], [], []
    for i in range(line_count):
        if rarray[i] < train_test_ratio:
            train_set.append(lsi_matrix[i, :])
            train_tag.append(tag_list[i])
        else:
            test_set.append(lsi_matrix[i, :])
            test_tag.append(tag_list[i])
    # 生成分类器
    predictor = xgboost_multi_classify(train_set, test_set, train_tag, test_tag)
    x = open(path_tmp_predictor, 'wb')
    pkl.dump(predictor, x)
    x.close()


'''
x_train 训练集样本
x_test  训练集标签
y_train 测试集样本
y_test  测试集标签
train_test_ratio： 训练集与测试集划分比例
'''


# *********************xgboost训练分类模型************************
def xgboost_multi_classify(train_set, test_set, train_tag, test_tag):
    # 统计信息
    print("训练集大小:", len(train_tag), " 测试集大小:", len(test_tag))
    train_info = {k: train_tag.count(k) for k in train_tag}
    print("训练集类别对应的样本数:", train_info)
    test_info = {k: test_tag.count(k) for k in test_tag}
    print("测试集类别对应的样本数", test_info)

    # XGBoost
    data_train = xgb.DMatrix(train_set, label=train_tag)
    data_test = xgb.DMatrix(test_set, label=test_tag)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 5,  # 类别数，与 multisoftmax 并用
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合
        'eta': 0.3,  # 如同学习率
        'eval_metric': 'merror',
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'subsample': 0.9,  # 随机采样训练样本
    }  # 参数
    xgb_model = xgb.train(param, data_train, num_boost_round=250, evals=watch_list)  # num_boost_round控制迭代次数
    y_hat = xgb_model.predict(data_test)
    validateModel(test_tag, y_hat)
    return xgb_model

    # 显示重要特征
    plot_importance(xgb_model[20])
    pyplot.show()


# *********************xgboost分类结果验证************************
'''
y_true 文本对应的正确类别
y_pred 分类器预测的类别
'''


def validateModel(y_true, y_pred):
    classify_report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    score = metrics.accuracy_score(y_true, y_pred)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))


def GeneModel():
    path_doc_root = r"C:\Users\15561\PycharmProjects\pres\CEC-Corpus-master\CEC-Corpus-master\CEC"  # 根目录 即存放按类分类好的文本数据集
    path_tmp = r"C:\Users\15561\PycharmProjects\pres\CEC_model"  # 存放中间结果的位置
    # 数据预处理阶段1：生成词典并去掉低率项，如果词典不存在则重新生成。反之跳过该阶段
    path_dictionary = os.path.join(path_tmp, 'CECNews.dict')  # 词典路径
    if os.path.exists(path_dictionary):
        print('=== 检测到词典已生成，跳过数据预处理阶段1 ===')
    else:
        os.makedirs(path_tmp)  # 创建中间结果保存路径
        GeneDict(path_doc_root, path_dictionary)

    # 数据预处理阶段2：开始将文档转化成tfidf
    path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_CEC')  # tfidf存储路径
    if os.path.exists(path_tmp_tfidf):
        print('=== 检测到tfidf向量已经生成，跳过数据预处理阶段2 ===')
    else:
        GeneTFIDF(path_doc_root, path_dictionary, path_tmp_tfidf)

    # 数据预处理阶段3：TFIDF向量模型生成LSI主题模型
    num_topics = 300  # 特征维度
    train_test_ratio = 0.7  # 训练集和测试集比率
    param = str(num_topics) + '_' + str(train_test_ratio)
    # 存放中间结果的位置
    path_tmp = os.path.join(path_tmp, param)
    path_tmp_lsi = os.path.join(path_tmp, 'lsi_corpus_' + param)
    # lsi模型保存路径
    path_tmp_lsimodel = os.path.join(path_tmp, 'lsi_model_' + param + '.pkl')
    if os.path.exists(path_tmp_lsi):
        print('=== 检测到LSI主题模型已经生成，跳过数据预处理阶段3 ===')
    else:
        os.makedirs(path_tmp_lsi)
        GeneLSI(path_dictionary, path_tmp_tfidf, path_tmp_lsimodel, path_tmp_lsi, num_topics)

    # 生成分类器阶段：xgboost训练分类模型
    path_tmp_predictor = os.path.join(path_tmp, 'predictor_' + param + '.pkl')
    print("\n特征维度:", num_topics, "\n训练集和测试集划分比率:", train_test_ratio)
    if os.path.exists(path_tmp_predictor):
        print('=== 检测到分类器已经生成，跳过生成分类器阶段 ===')
    else:
        GeneClassifier(path_tmp_lsi, path_tmp_predictor, train_test_ratio)

    return path_dictionary, path_tmp_lsi, path_tmp_lsimodel, path_tmp_predictor


########################主函数#############################################

if __name__ == '__main__':
    print('Start....{t}'.format(t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    print('\n\n\n', "*" * 15, '分类器训练过程', "*" * 15)
    # 模型训练过程
    path_dictionary, path_tmp_lsi, path_tmp_lsimodel, path_tmp_predictor = GeneModel()
    print("*" * 15, '分类器训练结束', "*" * 15, '\n\n\n\n\n')

    """print("*" * 15, '分类开始', "*" * 15)
    demo_doc = " "
    demo_doc2 = " "
    print("未知文本类别内容为：\n\n", demo_doc2, '\n')
    TestClassifier(demo_doc2, path_dictionary, path_tmp_lsimodel, path_tmp_predictor, path_tmp_lsi)
    print("*" * 15, '分类结束', "*" * 15, '\n\n\n')"""

    print('End......{t}'.format(t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
