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

USE_OMP = True
USE_COSAMP = False
SAMPLE_RATE = 0.9
IF_PLOT = True



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
                tem = cursor.fetchone() # the tem unit of the sensordata
                sensordata_valid = [] # deposit value that is not none
                none_index = []  # deposit none index in this list
                sensordata_all = []  # deposit all values including none
                i = 0
                while tem is not None:
                    # when encounter the none value, append the none index into list
                    # and append the none value into sensordata_all
                    if tem[0] == None:
                        none_index.append(i)
                        sensordata_all.append(tem[0])
                    else:
                        sensordata_valid.append(tem[0])
                        sensordata_all.append(tem[0])
                    tem = cursor.fetchone()
                    i = i+1
                sensordata_valid = np.array(sensordata_valid)  #  exclude none
                sensordata_all = np.array(sensordata_all)  # include none value
                if IF_PLOT:
                    plt.subplot(211)
                    plt.plot(sensordata_all)
                #     length of valid sensordata
                len_valid = len(sensordata_valid)
                #     length of all sensordata
                len_all = len(sensordata_valid)+len(none_index)
                # get into reconstruct algorithm (three choices)
                if USE_OMP:
                    recon = OMP(sensordata=sensordata_valid, original_size=len_all,
                                none_index=none_index,valid_size=len_valid)
                elif USE_COSAMP:
                    recon = CoSaMP(sensordata=sensordata_valid, original_size=len_all,
                                   none_index=none_index,valid_size=len_valid)
                else:
                    recon = Mean(sensordata=sensordata_valid, original_size=len_all,
                                 none_index=none_index,valid_size=len_valid)
                #  append this time period into the whole column
                column = np.append(column, recon.result)
                if IF_PLOT:
                    plt.subplot(212)
                    plt.plot(recon.result)
                    plt.show()
                print("Reconstruct successful")
                # image2 = Image.fromarray(recon.result[:,np.newaxis])
                # image2.show()
        #  append this column(var) into the final result
        final_result = np.c_[final_result,column]
    writer.writerows(final_result)


