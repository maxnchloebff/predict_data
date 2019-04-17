#coding:utf-8
import  numpy as np
import math
from PIL import Image



#OMP算法函数
def cs_omp(y,D):
    L=math.floor(3*(y.shape[0])/4)
    residual=y  #初始化残差
    index=np.zeros((L),dtype=int)
    for i in range(L):
        index[i]= -1
    result=np.zeros((256))

    for j in range(L):  #迭代次数
        product=np.fabs(np.dot(D.T,residual))
        pos=np.argmax(product)  #最大投影系数对应的位置
        index[j]=pos
        list = []
        for value in index:
            if value >=0:
                list.append(value)
        my=np.linalg.pinv(D[:,list]) #最小二乘,看参考文献1
        a=np.dot(my,y) #最小二乘,看参考文献1
        residual=y-np.dot(D[:,list],a)
        result[list]=a
    return  result

#读取图像，并变成numpy类型的 array
im = np.array(Image.open('lena256.bmp')) #图片大小256*256

#生成高斯随机测量矩阵
sampleRate=0.7  #采样率
Phi=np.random.randn(int(256*sampleRate),256)

#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((256,256))
v=range(256)
for k in range(0,256):
    dct_1d=np.cos(np.dot(v,k*math.pi/256))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

#随机测量
img_cs_1d=np.dot(Phi,im)

#重建
sparse_rec_1d=np.zeros((256,256))   # 初始化稀疏系数矩阵
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
for i in range(256):
    print('正在重建第',i,'列。')
    column_rec=cs_omp(img_cs_1d[:,i],Theta_1d) #利用OMP算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

#显示重建后的图片
image2=Image.fromarray(img_rec)
image2.show()