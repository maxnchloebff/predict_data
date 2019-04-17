import numpy as np
import math
import random
class CoSaMP():
    def __init__(self,sensordata,valid_size,original_size,none_index):
        self.sensor_data = sensordata
        self.valid_size = valid_size
        self.original_size = original_size
        self.none_index = none_index
        # 生成高斯随机测量矩阵
        self.sampleRate = 0.5  # 采样率
        self.Phi = np.random.randn(int(self.valid_size * self.sampleRate),
                              self.valid_size)
        # 生成稀疏基DCT矩阵
        self.mat_dct_1d = np.zeros((original_size, original_size))
        v = range(original_size)
        for k in range(0, original_size):
            dct_1d = np.cos(np.dot(v, k * math.pi / original_size))
            if k > 0:
                dct_1d = dct_1d - np.mean(dct_1d)
            self.mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)
        self.re_Phi = self.Phi
        for loc in self.none_index:
            self.re_Phi = np.insert(self.re_Phi,
                                    loc, axis=1,
                                    values=np.zeros((int(self.valid_size * self.sampleRate))))
        # 随机测量
        sampled_data = np.dot(self.Phi, sensordata)
        Theta_1d = np.dot(self.re_Phi, self.mat_dct_1d)  # 测量矩阵乘上基矩阵
        result = self.cs_CoSaMP(sampled_data=sampled_data, D=Theta_1d)
        self.result = np.dot(self.mat_dct_1d, result)

    def cs_CoSaMP(self, sampled_data,D):
        S=math.floor(sampled_data.shape[0]/4)  #稀疏度
        residual=sampled_data  #初始化残差
        pos_last=np.array([],dtype=np.int64)
        result=np.zeros((self.original_size))
        for j in range(int(S)):  #迭代次数
            product=np.fabs(np.dot(D.T,residual))
            pos_temp=np.argsort(product)
            pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
            pos_temp=pos_temp[0:int(2*S)]#对应步骤3
            pos=np.union1d(pos_temp,pos_last)

            result_temp=np.zeros((self.original_size))
            result_temp[pos] =np.dot(np.linalg.pinv(D[:,pos]),sampled_data)

            pos_temp=np.argsort(np.fabs(result_temp))
            pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
            result[pos_temp[:int(S)]]=result_temp[pos_temp[:int(S)]]
            pos_last=pos_temp
            residual=sampled_data-np.dot(D,result)
        return  result

class OMP():
    def __init__(self,sensordata,valid_size,original_size,none_index):
        self.sensordata = sensordata # of valid_size
        self.original_size = original_size
        self.valid_size = valid_size
        self.none_index = none_index
        # 生成高斯随机测量矩阵
        sampleRate = 0.5  # 采样率
        self.Phi = np.random.randn(int(self.valid_size * sampleRate),
                              self.valid_size)
        # 生成稀疏基DCT矩阵
        self.mat_dct_1d = np.zeros((original_size, original_size))
        self.re_Phi = self.Phi
        for loc in self.none_index:
            self.re_Phi = np.insert(self.re_Phi,
                                    loc, axis=1,
                                    values=np.zeros((int(self.valid_size * sampleRate))))
        v = range(original_size)
        for k in range(original_size):
            dct_1d = np.cos(np.dot(v, k * math.pi / original_size))
            if k > 0:
                dct_1d = dct_1d - np.mean(dct_1d)
            self.mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)
        sampled_data = np.dot(self.Phi, sensordata)
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

class Mean():

    def __init__(self,sensordata,valid_size,original_size,none_index):
        self.sensor_data = sensordata
        self.valid_size = valid_size
        self.original_size = original_size
        self.none_index = none_index
        self.segs = 1
        num = none_index[0]
        for i in none_index:
            if i == num:
                num = num + 1
                continue
            else:
                self.segs += 1
                num = i + 1
        self.result = self.mean()

    def mean(self):
        # sample 10 values from the original sensor data
        slice = range(len(self.sensor_data))
        sampled_index = random.sample(slice,10)
        mean = 0
        for index in sampled_index:
            mean += self.sensor_data[index]
        mean /= 10
        # insert every mean into sensor_data
        result = self.sensor_data
        for index in self.none_index:
            np.insert(result,index,values=mean)
        return result






