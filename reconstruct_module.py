import numpy as np
import math
import random
from Enum_module import algorithm_mode


class Reconstruct():
    def __init__(self,sensordata,valid_size,original_size,none_index,using_method,sample_rate=0.7):
        self.sensor_data = sensordata
        self.valid_size = valid_size
        self.original_size = original_size
        self.none_index = none_index
        self.using_method = using_method
        # measure matrix
        self.sampleRate = sample_rate
        self.Phi = np.random.randn(int(self.valid_size * self.sampleRate),
                              self.valid_size)
        # generate DCT matrix
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
        # random measure
        sampled_data = np.dot(self.Phi, sensordata)
        Theta_1d = np.dot(self.re_Phi, self.mat_dct_1d)  # measure matrix dot the base matrix
        if self.using_method == algorithm_mode.COSAMP:
            result = self.cs_CoSaMP(sampled_data=sampled_data, D=Theta_1d)
        elif self.using_method == algorithm_mode.OMP:
            result = self.cs_omp(sampled_data,Theta_1d)
        elif self.using_method == algorithm_mode.IRLS:
            result = self.cs_IRLS(sampled_data, Theta_1d)
        elif self.using_method == algorithm_mode.IHT:
            result = self.cs_IHT(sampled_data,Theta_1d)
        elif self.using_method == algorithm_mode.SP:
            result = self.cs_sp(sampled_data, Theta_1d)
        else:
            result = self.mean()
        self.result = np.dot(self.mat_dct_1d, result)

    def cs_CoSaMP(self, sampled_data,D):
        S=math.floor(3*sampled_data.shape[0]/4)  #sparse
        residual=sampled_data  #initialize the residual
        pos_last=np.array([],dtype=np.int64)
        result=np.zeros((self.original_size))
        for j in range(int(S)):  #iteration times
            product=np.fabs(np.dot(D.T,residual))
            pos_temp=np.argsort(product)
            pos_temp=pos_temp[::-1]#reverse the pos_tem
            pos_temp=pos_temp[0:int(2*S)]#select the first 2*s better location
            pos=np.union1d(pos_temp,pos_last)

            result_temp=np.zeros((self.original_size))
            result_temp[pos] =np.dot(np.linalg.pinv(D[:,pos]),sampled_data)

            pos_temp=np.argsort(np.fabs(result_temp))
            pos_temp=pos_temp[::-1]#reverse, then select the first s best
            result[pos_temp[:int(S)]]=result_temp[pos_temp[:int(S)]]
            pos_last=pos_temp
            residual=sampled_data-np.dot(D,result)
        return  result

    def cs_omp(self, sampled_data, D):
        L=math.floor(3*(sampled_data.shape[0])/4) # sparse
        # initialize the residual
        residual = sampled_data
        result = np.zeros((self.original_size))
        index = np.zeros((L), dtype=int)
        for i in range(L):
            index[i] = -1
        for j in range(L):  # iteration times
            product = np.fabs(np.dot(D.T, residual))
            pos = np.argmax(product)  # the best location
            index[j] = pos
            list = []
            for value in index:
                if value >= 0:
                    list.append(value)
            my = np.linalg.pinv(D[:, list])  # OLS
            a = np.dot(my, sampled_data)
            residual = sampled_data - np.dot(D[:, list], a)
            result[list] = a
        return result

    def cs_IRLS(self, sampled_data, D):
        L = math.floor((sampled_data.shape[0]) / 4)
        hat_x_tp = np.dot(D.T, sampled_data)
        epsilong = 1
        p = 1  # solution for l-norm p
        times = 1
        while (epsilong > 10e-9) and (times < L):  # 迭代次数
            weight = (hat_x_tp ** 2 + epsilong) ** (p / 2 - 1)
            Q_Mat = np.diag(1 / weight)
            # hat_x=Q_Mat*T_Mat'*inv(T_Mat*Q_Mat*T_Mat')*y
            temp = np.dot(np.dot(D, Q_Mat), D.T)
            temp = np.dot(np.dot(Q_Mat, D.T), np.linalg.inv(temp))
            hat_x = np.dot(temp, sampled_data)
            if (np.linalg.norm(hat_x - hat_x_tp, 2) < np.sqrt(epsilong) / 100):
                epsilong = epsilong / 10
            hat_x_tp = hat_x
            times = times + 1
        return hat_x_tp

    def cs_IHT(self,sampled_data, D):
        K = math.floor(sampled_data.shape[0] / 4)  # 稀疏度
        result_temp = np.zeros((self.original_size))  # 初始化重建信号
        u = 0.5  # 影响因子
        result = result_temp
        for j in range(int(K)):  # 迭代次数
            x_increase = np.dot(D.T, (sampled_data - np.dot(D, result_temp)))  # x=D*(y-D*y0)
            result = result_temp + np.dot(x_increase, u)  # x(t+1)=x(t)+u*D*(y-D*y0)
            temp = np.fabs(result)
            pos = temp.argsort()
            pos = pos[::-1]  # 反向，得到前面K个大的位置
            result[pos[int(K):]] = 0
            result_temp = result
        return result

    def cs_sp(self,sampled_data, D):
        K = int(math.floor(sampled_data.shape[0] / 3) )
        pos_last = np.array([], dtype=np.int64)
        result = np.zeros((self.original_size))

        product = np.fabs(np.dot(D.T, sampled_data))
        pos_temp = product.argsort()
        pos_temp = pos_temp[::-1]  # 反向，得到前面L个大的位置
        pos_current = pos_temp[0:K]  # 初始化索引集 对应初始化步骤1
        residual_current = sampled_data - np.dot(D[:, pos_current], np.dot(np.linalg.pinv(D[:, pos_current]), sampled_data))  # 初始化残差 对应初始化步骤2

        while True:  # 迭代次数
            product = np.fabs(np.dot(D.T, residual_current))
            pos_temp = np.argsort(product)
            pos_temp = pos_temp[::-1]  # 反向，得到前面L个大的位置
            pos = np.union1d(pos_current, pos_temp[0:K])  # 对应步骤1
            pos_temp = np.argsort(np.fabs(np.dot(np.linalg.pinv(D[:, pos]), sampled_data)))  # 对应步骤2
            pos_temp = pos_temp[::-1]
            pos_last = pos_temp[0:K]  # 对应步骤3
            residual_last = sampled_data - np.dot(D[:, pos_last], np.dot(np.linalg.pinv(D[:, pos_last]), sampled_data))  # 更新残差 #对应步骤4
            if np.linalg.norm(residual_last) >= np.linalg.norm(residual_current):  # 对应步骤5
                pos_last = pos_current
                break
            residual_current = residual_last
            pos_current = pos_last
        result[pos_last[0:K]] = np.dot(np.linalg.pinv(D[:, pos_last[0:K]]), sampled_data)  # 对应输出步骤
        return result

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
            result = np.insert(result,index,values=mean)
        return result

