#coding:utf-8
import  numpy as np
from PIL import Image
import math
import re
from enum import Enum
import  mysql.connector as connector
import string
import time
import datetime
import csv
import matplotlib.pyplot as plt
from reconstruct_module import *

USE_OMP = False
USE_COSAMP = True



if __name__ == "__main__":
    f =  open('result.csv', 'w', newline='')
    writer = csv.writer(f)
    mysql = connector.connect(user='root', password='00011122q', buffered=True, host='127.0.0.1')
    cursor = mysql.cursor()
    cursor.execute("use bigdata")
    time = datetime.datetime(year=2018, month=7, day=1, hour=0, minute=0, second=0 )
    time_delta = datetime.timedelta(hours=1)
    cursor.execute("select ts from table_1 ")
    final_result = []
    tem_time = cursor.fetchone()
    while tem_time is not None:
        final_result.append(tem_time)
        tem_time = cursor.fetchone()
    final_result = np.array(final_result)
    # for every var go into a loop
    for num in range(0,68):
        column = np.array([])
        while True:
            # judge the range of time
            if time >= datetime.datetime(year=2018, month=8, day=1, hour=0, minute=0, second=0 ):
                time = datetime.datetime(year=2018, month=7, day=1, hour=0, minute=0, second=0 )
                break
            print("Constructing var{}".format(num+1)+" now is" +str(time))
            # select the null var and calculate the num of none value
            cursor.execute("select var{} from table_1 where var{} is null and ts <'{}' and ts >='{}'".format(num+1,num+1,str(time+time_delta),str(time)) )
            result_invalid = cursor.fetchall()
            len_invalid = len(result_invalid)
            # select all the value of the value
            cursor.execute(
                "select var{} from table_1 where ts <'{}' and ts >='{}'".format(num + 1, str(time + time_delta),str(time)))
            # if there is no none value then go into the next time period
            if len_invalid ==0:
                time = time + time_delta
                tem = cursor.fetchone()
                partial_column = []
                while tem is not None:
                    partial_column.append(tem[0])
                    tem = cursor.fetchone()
                partial_column = np.array(partial_column)
                column = np.append(column,partial_column)
                continue
            # if there is none value in this time period
            else:
                time = time + time_delta
                tem = cursor.fetchone()
                result_valid = [] # deposit value that is not none
                none_index = []  # deposit none index in this list
                result_all = []  # deposit all values including none
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
                plt.plot(result_all)
                len_valid = len(result_valid)
                len_all = len(result_valid)+len(none_index)
            # print(len(result))
                if USE_OMP:
                    recon = OMP(sensordata=result_valid, original_size=len_all,none_index=none_index,valid_size=len_valid)
                elif USE_COSAMP:
                    recon = CoSaMP(sensordata=result_valid, original_size=len_all,none_index=none_index,valid_size=len_valid)
                else:
                    recon = Mean(sensordata=result_valid, original_size=len_all,none_index=none_index,valid_size=len_valid)
                column = np.append(column, recon.result)
                plt.plot(recon.result)
                plt.show()
                print("Reconstruct successful")
                # image2 = Image.fromarray(recon.result[:,np.newaxis])
                # image2.show()
        final_result = np.c_[final_result,column]
    writer.writerows(final_result)


