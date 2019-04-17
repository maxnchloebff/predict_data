#coding:utf-8
import mysql.connector as connector
import datetime
import csv
import matplotlib.pyplot as plt
from reconstruct_module import *

USE_OMP = True
USE_COSAMP = False
SAMPLE_RATE = 1.2
IF_PLOT = True
START_VAR  = 1
VAR_MAXHOLD = 0.01
START_TIME = datetime.datetime(year=2018, month=7, day=4, hour=0, minute=0, second=0 )
TIME_DELTA = datetime.timedelta(hours=1)

def init_writer():
    f =  open('result.csv', 'w', newline='')
    writer = csv.writer(f)
    return writer

if __name__ == "__main__":
    # initialize writer and write the final_result into result.csv document
    writer = init_writer()
    
    # connect to mysql with mysql_connector
    mysql = connector.connect(user='root', password='00011122q', buffered=True, host='127.0.0.1')
    cursor = mysql.cursor()
    cursor.execute("use bigdata")

    # set the starting time with 2017-07-01-00:00:00
    time = START_TIME

    # set the length of time period
    time_delta = TIME_DELTA
    cursor.execute("select ts from table_1 ")

    #  initialize the final_result, append the ts into final_result
    final_result = []
    tem_time = cursor.fetchone()
    while tem_time is not None:
        final_result.append(tem_time)
        tem_time = cursor.fetchone()
    final_result = np.array(final_result)

    # for every var go into a loop
    for num in range(START_VAR-1,68):
        column = np.array([])
        while True:
            # judge the range of time
            if time >= datetime.datetime(year=2018, month=8, day=1, hour=0, minute=0, second=0 ):
                time = START_TIME
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
                sensor_var = np.var(sensordata_valid) # the variance of valid sensordata
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
                    if sensor_var >= VAR_MAXHOLD:
                        recon = OMP(sensordata=sensordata_valid, original_size=len_all,
                                none_index=none_index,valid_size=len_valid, sample_rate=SAMPLE_RATE)
                    else:
                        recon = Mean(sensordata=sensordata_valid, original_size=len_all,
                                none_index=none_index,valid_size=len_valid)
                elif USE_COSAMP:
                    if sensor_var >= VAR_MAXHOLD:
                        recon = CoSaMP(sensordata=sensordata_valid, original_size=len_all,
                                   none_index=none_index,valid_size=len_valid, sample_rate=SAMPLE_RATE)
                    else:
                        reocn = Mean(sensordata=sensordata_valid, original_size=len_all,
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
        #  append this column(var) into the final result
        final_result = np.c_[final_result,column]
    writer.writerows(final_result)