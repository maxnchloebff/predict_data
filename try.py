#coding:utf-8
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DCT基作为稀疏基，重建算法为OMP算法 ，图像按列进行处理
# 参考文献: 任晓馨. 压缩感知贪婪匹配追踪类重建算法研究[D].
#北京交通大学, 2012.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 导入所需的第三方库文件
import  numpy as np
from PIL import Image
import math
import re
from enum import Enum
import  mysql.connector as connector
import string
import time
import datetime
class reconstruct():
    def __init__(self,sensordata,valid_size,original_size,none_index):
        """
        sensor_data is a numpy array
        :param sensor_data:
        """
        self.sensordata = sensordata # of valid_size
        self.original_size = original_size
        self.valid_size = valid_size
        self.none_index = none_index
        # 生成高斯随机测量矩阵
        sampleRate = 0.7  # 采样率
        Phi = np.random.randn(int(self.valid_size * sampleRate), self.valid_size)
        # 生成稀疏基DCT矩阵
        self.mat_dct_1d = np.zeros((original_size, original_size))
        self.re_Phi = Phi
        for loc in self.none_index:
            self.re_Phi = np.insert(self.re_Phi, loc, axis=1, values=np.zeros((int(self.valid_size * sampleRate))))
        v = range(original_size)
        for k in range(original_size):
            dct_1d = np.cos(np.dot(v, k * math.pi / original_size))
            if k > 0:
                dct_1d = dct_1d - np.mean(dct_1d)
            self.mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)

        sampled_data = np.dot(Phi, sensordata)
        Theta_1d = np.dot(self.re_Phi, self.mat_dct_1d)  # 测量矩阵乘上基矩阵
        result = self.cs_omp(sampled_data=sampled_data, D = Theta_1d )
        self.result = np.dot(self.mat_dct_1d,result)

    def cs_omp(self, sampled_data, D):
        L=math.floor(3*(sampled_data.shape[0])/4)
        residual = sampled_data  # 初始化残差
        result = np.zeros((self.original_size))
        index = np.zeros((L), dtype=int)
        for i in range(L):
            index[i] = -1
        for j in range(L):  # 迭代次数
            product = np.fabs(np.dot(D.T, residual))
            pos = np.argmax(product)  # 最大投影系数对应的位置
            index[j] = pos
            list = []
            for value in index:
                if value >= 0:
                    list.append(value)
            my = np.linalg.pinv(D[:, list])  # 最小二乘,看参考文献1
            a = np.dot(my, sampled_data)  # 最小二乘,看参考文献1
            residual = sampled_data - np.dot(D[:, list], a)
            result[list] = a
        return result


if __name__ == "__main__":
    mysql = connector.connect(user='root', password='00011122q', buffered=True, host='127.0.0.1')
    cursor = mysql.cursor()
    cursor.execute("use bigdata")
    time = datetime.datetime(year=2018, month=7, day=1, hour=0, minute=0, second=0 )
    time_delta = datetime.timedelta(hours=1)
    while True:
        if not time < datetime.datetime(year=2018, month=7, day=31, hour=23, minute=59, second=59 ):
            break
        print(str(time))
        cursor.execute("select var1 from table_1 where var1 is null and ts <'{}' and ts >'{}'".format(str(time+time_delta),str(time)) )
        result_invalid = cursor.fetchall()
        len_invalid = len(result_invalid)
        if len_invalid ==0:
            time = time + time_delta
            continue
        else:
            cursor.execute("select var1 from table_1 where ts <'{}' and ts >'{}'".format(str(time+time_delta),str(time))  )
            time = time + time_delta
            tem = cursor.fetchone()
            result_valid = []
            none_index = []
            result_all = []
            i = 0
            while tem is not None:
                if tem[0] == None:
                    none_index.append(i)
                    result_all.append(tem[0])
                else:
                    result_valid.append(tem[0])
                    result_all.append(tem[0])
                tem = cursor.fetchone()
                i = i+1
            result_valid = np.array(result_valid)
            result_all = np.array(result_all)
            len_valid = len(result_valid)
            len_all = len(result_valid)+len(none_index)
        # print(len(result))

            recon = reconstruct(sensordata=result_valid, original_size=len_all,none_index=none_index,valid_size=len_valid)
            # image2 = Image.fromarray(recon.result[:,np.newaxis])
            # image2.show()








# select_nth_column = "select var{} from table_1 where var{} is not null"
# s
#
# # print(select_fine)
# # print(select_all_fine)
# cursor.execute(use_database)
# cursor.execute(select_nth_column.format(1,1))

# predict_module(values_array)