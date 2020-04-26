# -*- coding: utf-8 -*-
# @Time: 2020/4/24 9:53
import warnings
warnings.filterwarnings(action='ignore')

from warnings import showwarning
from pandas import DataFrame
import seaborn as sns
import sys
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
import joblib

# user defined module
from userDefinedError import NonSupportError, dataTypeError

def cosine_similarity(vector_a, vector_b):
    """
    Cosine similarity
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    cosSim = 0.5 + 0.5 * cos  # normalization
    return cosSim

class missing_value_processing():
    def _initial_check(self, data, cols):
        if type(data) != DataFrame or type(cols) not in [str, list, set]:
            raise NonSupportError

    def _fill_value(self, col, value, int_type):
        if int_type:
            self.data[col] = self.data[col].fillna(int(value))
        else:
            self.data[col] = self.data[col].fillna(value)
        return self.data

    def __init__(self, data: DataFrame, cols:'str or list or set'=''):
        '''
        :param data: DataFrame datas
        :param cols: The columns which need to fill missing value.
        '''
        self._initial_check(data, cols)
        self.data = data
        self.cols = list(cols)

    def drop(self, axis=0):
        self.data = self.data.dropna(axis=axis)
        return self.data

    def fill_by_statistic(self, method="mean", int_type=False):
        '''
        fill by statistic
        :param method: mean or median or mode
        :param int_type: if True, convert value to integer
        :return:
        '''
        if method == "mean":
            for col in self.cols:
                fill_value = self.data[col].mean()
                self._fill_value(col, fill_value, int_type)
        elif method == "median":
            for col in self.cols:
                fill_value = self.data[col].median()
                self._fill_value(col, fill_value, int_type)
        elif method == "mode":
            for col in self.cols:
                fill_value = self.data[col].mode()
                if len(fill_value) > 1:
                    fill_value = fill_value[0]  # 有多个众数时，取第一个
                self._fill_value(col, fill_value, int_type)
        else:
            raise NonSupportError
        return self.data

    def hot_card_filling(self, method="COA", user_defined_cols=''):
        '''
        :param method: COA means 'coefficient of association' - 相关系数矩阵(皮尔逊相关系数)
                       COS means 'Cosine similarity' - 余弦相似度
        :return:
        '''
        if method == "COA":
            '''
            常用策略：找到最相关指标，根据此指标排序，对缺失指标采用ffill策略填充。(可考虑用前后均值优化等)
            注意：当排序后的前几个值为空时，ffill的填充结果仍为空！
            '''
            corr = self.data.corr()
            for col in self.cols:
                simil_value = corr[col].sort_values().values[-3]
                if simil_value < 0.5:
                    showwarning("The coefficient of association is less than 0.5. Other strategies are suggested!",
                                category=UserWarning,
                                filename=str(sys._getframe().f_code.co_filename),
                                lineno=str(sys._getframe().f_lineno))
                simil_col = corr[col].sort_values().index[-2]
                self.data = self.data.sort_values(by=simil_col)
                self.data[col] = self.data[col].ffill()
        elif method == "COS":
            if user_defined_cols == "":
                useful_cols = [col for col in self.data.columns.tolist() if self.data[col].dtype != object]
            else:
                for c in user_defined_cols:
                    if self.data[c].dtype == object:
                        raise dataTypeError
                useful_cols = user_defined_cols
            for col in self.cols:
                cal_cols = list(set(useful_cols) - set([col]))
                df_nan = self.data[self.data[col].isnull()]
                for i in range(len(df_nan)):
                    vector_a = df_nan.iloc[i][useful_cols].values.tolist()
                    index_a = df_nan.index.values[i]
                    cos_list = []
                    for j in range(len(self.data)):
                        index_b = self.data.index.values[j]
                        if index_b == index_a or math.isnan(self.data.iloc[j][col]):
                            continue
                        else:
                            vector_b = self.data.iloc[j][useful_cols].values.tolist()
                            useful_values = [(vector_a[n], vector_b[n]) for n in range(len(vector_a))
                                             if not math.isnan(vector_a[n]) and not math.isnan(vector_b[n])]
                            if len(useful_values) < 3:
                                continue
                            else:
                                cos = cosine_similarity([v[0] for v in useful_values], [v[1] for v in useful_values])
                                cos_list.append((cos, index_b))
                    cos_list = sorted(cos_list, key=lambda x: x[0], reverse=True)
                    for order in range(len(cos_list)):
                        try:
                            index_ = cos_list[order][1]
                            fill_v = self.data[self.data.index.values == index_][col].values[0]
                            self.data.loc[index_a, col] = fill_v
                        except:
                            print("col->{} index->{} 没有搜索到相似样本。".format(col, index_a))
        else:
            raise NonSupportError

    def kmeans_filling(self, n_clusters=5, int_type=False, user_defined_cols=''):
        # 待添加模型保存和预测阶段处理过程
        if user_defined_cols == "":
            useful_cols = [col for col in self.data.columns.tolist() if self.data[col].dtype != object]
        else:
            for c in user_defined_cols:
                if self.data[c].dtype == object:
                    raise dataTypeError
            useful_cols = user_defined_cols
        for col in self.cols:
            cal_cols = list(set(useful_cols) - set(self.cols))
            df_nan = self.data[self.data[col].isnull()]
            # data check
            df_dropna = self.data[~self.data[col].isnull()][useful_cols + [col]].dropna()
            if len(df_dropna) < n_clusters * 1:  # 视数据情况，建议倍数尽可能大，目前按最低要求1倍
                raise BaseException("用于计算的特征列缺失较严重，有效样本不足！")
            if len(df_nan[cal_cols].dropna()) < len(df_nan):
                raise BaseException("数据缺失，无法预测！请预先进行缺失值填充或重新选择有效特征！")
            else:
                km = KMeans(n_clusters=n_clusters)
                km.fit(df_dropna[cal_cols])
                df_dropna["cluster"] = km.predict(df_dropna[cal_cols])
                df_nan["cluster"] = km.predict(df_nan[cal_cols])
                for i in range(len(df_nan)):
                    index_ = df_nan.index.values[i]
                    fill_v = np.average(df_dropna[df_dropna["cluster"] == df_nan.iloc[i]["cluster"]][col].values)
                    if int_type:
                        fill_v = int(fill_v)
                    else:
                        fill_v = round(fill_v, 4)
                    self.data.loc[index_, col] = fill_v

        return self.data






if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from itertools import product

    df = pd.DataFrame()
    df["a"] = np.random.randint(0,20,8)
    df["b"] = [1,2,3,5,np.NaN,6,np.NaN,9]
    df["c"] = [np.NaN,np.NaN,1.8,5,np.NaN,3.6,np.NaN,np.NaN]
    df["d"] = range(8)
    df["e"] = [1,2,2,3,5,6,7,8]
    print("df:\n", df)

    # dropna
    if 0:
        m1 = missing_value_processing(df.copy())
        df_m1 = m1.drop(axis=0)
        print("dropna:\n", df_m1)

    # fill_by_statistic
    if 0:
        for method, int_type in product(["mean", "median", "mode"], [True, False]):
            print("method:{}, int_type:{} ----> ".format(method, int_type))
            m2 = missing_value_processing(df.copy(),["b", "c"])
            df_m2 = m2.fill_by_statistic(method=method, int_type=int_type)
            print("\t", df_m2)

    # hot card
    if 0:
        m3 = missing_value_processing(df.copy(), ["b", "c"])
        df_m3 = m3.hot_card_filling(method="COS")
        print("df_m3:", df_m3)

    # k-means
    if 1:
        m4 = missing_value_processing(df.copy(), ["b", "c"])
        df_m4 = m4.kmeans_filling(n_clusters=3, user_defined_cols=["a", "d", "e"])
        print("df_m4:", df_m4)