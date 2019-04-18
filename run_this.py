#coding:utf-8
import mysql.connector as connector
import datetime
import csv
import matplotlib.pyplot as plt
from reconstruct_module import *
from Enum_module import algorithm_mode
from Enum_module import debug_mode


"""
we have several options of MODE:
OMP COSAMP IRLS SP  IHT MEAN
OMP and IRLS and SP have better performances
OMP = IRLS > SP  and MEAN is the basic algorithm
the others have serious problem which I don't understand(not fit in this problem)

Also we can choose two debug mode:
RECONSTRUCT or TEST
"""
MODE = algorithm_mode.IRLS
DEBUG = debug_mode.TEST
SAMPLE_RATE = 0.9
IF_PLOT = True
START_VAR  = 1
VAR_MAXHOLD = 0.01
START_TIME = datetime.datetime(year=2018, month=7, day=4, hour=0, minute=0, second=0 )
END_TIME =  datetime.datetime(year=2018, month=7, day=5, hour=0, minute=0, second=0 )
TIME_DELTA = datetime.timedelta(hours=1)
EPS = 1   # 保留恢复结果的小数点后几位，基本都是一位，传感器精度决定
TEST_VAR = 1 # 我们检验的是几号变量
TIME_GAP = datetime.timedelta(minutes=20)  # 人为挖去的数据长度，以分钟来表示
TIME_CANDIDATE = range(40)  # 用于人为挖取数据时，随机挖去位置的候选list，长度为:TIME_DELTA-TIME_GAP

def init_writer():
    f =  open('result.csv', 'w', newline='')
    writer = csv.writer(f)
    return writer

def evaluate(ground_truth, reconstructed_data):
    if len(ground_truth) != len(reconstructed_data):
        print("the sizes are not matched")
        return
    length = len(ground_truth)
    result = 0
    for i in range(length):
        tem = abs((ground_truth[i]-reconstructed_data[i])/ground_truth[i] )
        result += tem
    result = result/length
    return result




if __name__ == "__main__":
    # connect to mysql with mysql_connector
    mysql = connector.connect(user='root', password='00011122q', buffered=True, host='127.0.0.1')
    cursor = mysql.cursor()
    cursor.execute("use bigdata")
    if DEBUG == debug_mode.RECONSTRUCT:
        # initialize writer and write the final_result into result.csv document
        writer = init_writer()

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
                    if sensor_var >= VAR_MAXHOLD:
                        recon = Reconstruct(sensordata=sensordata_valid,valid_size=len_valid,original_size=len_all,
                                            none_index=none_index,using_method=MODE,sample_rate=SAMPLE_RATE)
                    else:
                        recon = Reconstruct(sensordata=sensordata_valid, valid_size=len_valid, original_size=len_all,
                                            none_index=none_index, using_method=algorithm_mode.MEAN, sample_rate=SAMPLE_RATE)

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

    elif DEBUG == debug_mode.TEST:
        """
        We choose some typical vars to test our algorithm, firstly we figure 
        out when the var is not none in a certain time period, then we 
        select the origin data and artificially trim some data from it
        """
        time = START_TIME
        time_delta = TIME_DELTA
        valid_time_list = []
        # filter the data in each hour, select the data that is always valid
        while True:
            if time >= END_TIME:
                break
            cursor.execute("select var{} from table_1 where var{} is null and ts <'{}' and ts >='{}'".format(TEST_VAR+1,TEST_VAR+1,str(time+time_delta),str(time)) )
            print(time)
            if len(cursor.fetchall()) == 0:
                valid_time_list.append(time)
            time += time_delta
        time_gap = datetime.timedelta(minutes=20)
        """        
        for every time period during which all the data is valid, then trim some data
        and deposit the original sensordata in sensordata_original and deposit the 
        trimmed sensor data in sensordata_trimmed
        then we reconstruct the sensordata using sensordata_after_trimmed
        finally calculate the error using method "evaluate" and plot the result 
        """

        for valid_time in valid_time_list:
            #  initialize all the array
            time_point = datetime.timedelta(minutes=random.sample(TIME_CANDIDATE,1)[0]) # sample a random time point to trim
            sensordata_original = []
            sensordata_none = []
            sensordata_after_trimmed = []
            sensordata_trimmed = []
            none_index = []
            sensordata_reconstructed = []

            # select all the data according to valid time and time_point
            cursor.execute("select var{},ts from table_1 where ts <'{}' and ts >='{}'".format(TEST_VAR+1,str(valid_time+time_delta),str(valid_time)) )
            tem = cursor.fetchone()
            i = 0
            while tem is not None:
                if tem[1]< (valid_time + time_point) or tem[1] > (valid_time+time_point + time_gap):
                    sensordata_original.append(tem[0])
                    sensordata_none.append(tem[0])
                    sensordata_after_trimmed.append(tem[0])
                else:
                    sensordata_original.append(tem[0])
                    sensordata_none.append(None)
                    sensordata_trimmed.append(tem[0])
                    none_index.append(i)
                tem = cursor.fetchone()
                i += 1

            sensordata_none = np.array(sensordata_none)
            sensordata_original = np.array(sensordata_original)
            sensordata_trimmed = np.array(sensordata_trimmed)
            sensordata_after_trimmed = np.array(sensordata_after_trimmed)
            len_original = len(sensordata_none)
            len_valid = len(sensordata_after_trimmed)

            # then reconstruct
            print("Reconstructing " + str(valid_time))
            recon = Reconstruct(sensordata=sensordata_after_trimmed,valid_size=len_valid,original_size=len_original,
                                            none_index=none_index,using_method=MODE,sample_rate=SAMPLE_RATE)
            sensordata_reconstructed = recon.result
            sensordata_reconstructed = sensordata_reconstructed.round(EPS)
            error = evaluate(ground_truth=sensordata_trimmed, reconstructed_data=sensordata_reconstructed[none_index])
            if IF_PLOT:
                """
                画出的三幅图分别是
                1. 被人为截取后和缺失图
                2. ground_truth 图
                3. 用压缩感知恢复出的图像
                """
                plt.subplot(311)
                plt.plot(sensordata_none)
                plt.subplot(312)
                plt.plot(sensordata_original)
                plt.subplot(313)
                plt.plot(sensordata_reconstructed)
                plt.text(x=50, y=29, s="error is " + str(error * 100) + "%", fontsize=20)
                plt.show()

