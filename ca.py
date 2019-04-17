#coding:utf-8

#导入集成库
import math
import re
from enum import Enum
import  mysql.connector as connector
import string
import time
debug_mode = Enum('debug_module','load_data get_column_data process_column_data')
mode = debug_mode.process_column_data

# 导入所需的第三方库文件
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包

# Phi=np.random.randn(256,256)
# u, s, vh = np.linalg.svd(Phi)
# Phi = u[:256*sampleRate,] #将测量矩阵正交化

#CoSaMP算法类
class predict_module():
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
#重建

if __name__ == "__main__":

    if mode == debug_mode.load_data:
        # 首先预处理数据，由于数据中存在，，这样表示空值的数据，我们的操作是将这样的，，
        # 变为,\N, 这样有利于mysql load数据时的操作，不然空值会被自然设置为0
        f = open('/home/kai/Downloads/data/001/201807.csv')
        # 这里的路径是你csv存储的路径，绝对路径！
        f2 = open('/home/kai/Downloads/data/001/201807~.csv','w')
        for line in f:
            f2.write(re.sub(r',,',r',\N,',line))
        f2.close()
        f.close()
        #  下面是最标准的连接mysql的方法，user指的是用户名，
        #  之后是密码，buffered要职位true，host是localhost，故而是127.0.0.1
        mysql = connector.connect(user='root', password='00011122q',
                                  buffered=True, host='127.0.0.1')
        #  cursor 可理解为光标，在数据库中移动的光标，
        #  用于处理mysql命令，并且在cursor.feychall()中返回所查询的值
        cursor = mysql.cursor()
        # 创建bigdata数据库
        cursor.execute("create database if not exists bigdata")
        # 声明使用bigdata
        cursor.execute("use bigdata")
        # 打开这个模式，以便可以将外部文件数据load进如mysql
        cursor.execute("set global local_infile = 'ON' ")
        # 如果存在的话，删除table001
        cursor.execute("drop table if exists table_001 ")
        # 创建一个新的table001，将默认值设置为null
        cursor.execute(
            "create table if not exists table_001 (ts datetime not null,id int not null,var1 double default null,var2 double default null,var3 double default null,var4 double default null,var5 double default null,var6 double default null, var7 double default null,var8 double default null,var9 double default null,var10 double default null,var11 double default null,var12 double default null,var13 double default null,var14 double default null,var15 double default null,var16 int default null,var17 double default null,var18 double default null,var19 double default null,var20 int default null,var21 double default null,var22 double default null,var23 double default null,var24 double default null,var25 double default null,var26 double default null,var27 double default null,var28 double default null,var29 double default null,var30 double default null,var31 double default null,var32 double default null,var33 double default null,var34 double default null,var35 double default null,var36 double default null,var37 double default null,var38 double default null,var39 double default null,var40 double default null,var41 double default null,var42 double default null,var43 double default null,var44 double default null,var45 double default null,var46 double default null,var47 int default null,var48 double default null,var49 double default null,var50 double default null,var51 double default null,var52 double default null,var53 bool default null,var54 double default null,var55 double default null,var56 double default null,var57 double default null,var58 double default null,var59 double default null,var60 double default null,var61 double default null,var62 double default null,var63 double default null,var64 double default null,var65 double default null,var66 bool default null,var67 double default null,var68 double default null) ")
        # 将外部csv文件load进入table_001中
        cursor.execute("load data local infile '/home/kai/Downloads/data/001/201807~.csv' "
                       "into table table_001 fields terminated by ',' "
                       "OPTIONALLY ENCLOSED BY '\"' "
                       "lines terminated by '\n' ignore 1 lines ")
        # 接下来挑选其中有null值的row，下面的select_null字符串代表了mysql中最为常见的选择命令，
        # {}中是待填入的参数（1-68），下面这句话的意思就是从table001中选出var{}值为null的行，
        # 并返回他们这一行的ts值，返回的是一个list
        select_null = "select ts from table_001 where var{} is null "
        null_rows = []
        for i in range(1, 69):
            cursor.execute(select_null.format(i))
            tem = cursor.fetchall()  # 这是一个list，里面存放的是每一个var值为null的行的ts的值
            print('{} {}'.format(i, tem.__len__()))  # 打印每一个tem中有多少行存在null值

        cursor.close()
        mysql.close()

    if mode == debug_mode.get_column_data:
        mysql = connector.connect(user='root', password='00011122q', host='127.0.0.1', buffered=True)
        cursor = mysql.cursor()
        use_database = "use bigdata"
        select_nth_column = "select var{} from table_1 where var{} is not none "

        # print(select_fine)
        # print(select_all_fine)
        cursor.execute(use_database)
        for i in range(68):

            cursor.execute(select_nth_column.format(i+1))
            values = []
            one = cursor.fetchone()
            while one is not None:
                values.append(one[0])
                one = cursor.fetchone()
            values_array = np.array(values)
            print(values_array.shape)

    if mode == debug_mode.process_column_data:

        mysql = connector.connect(user='root', password='00011122q', host='127.0.0.1', buffered=True)
        cursor = mysql.cursor()
        use_database = "use bigdata"
        select_nth_column = "select var{} from table_1  "

        # print(select_fine)
        # print(select_all_fine)
        cursor.execute(use_database)
        cursor.execute(select_nth_column.format(1))
        values = []
        one = cursor.fetchone()
        while one is not None:
            values.append(one[0])
            one = cursor.fetchone()
        values_array = np.array(values)
        predict_module(values_array)

