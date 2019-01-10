# ------------将 odom 当中不符合条件的数据过滤掉------------
# ------------输入：原始data_odom   输出：过滤后的数据------------

import numpy as np

class filter :

    #------------1.初始化------------
    def __init__(self,data_odom):
        self.data_odom = data_odom 

        #针对非平面问题的数据筛选
        data_odom_extratct =  self.data_odom[ : , [0,5,6,7,8,9,10,11,48,49,50,51,52,53]] #包含了时间和姿态


        # 针对平面问题的数据筛选
        #只筛选特定的行
        self.data_odom_extratct = data_odom_extratct[ 1: , [0,1,2,6,8,13]].astype(np.float) #数据类型转化成float类型 1：把第一行筛选掉

    #------------2.过滤垃圾数据------------
    def filter(self):
        #------------过滤数据------------ 
        #取前1000行的数据，计算数据开始的时间
        for num in range(2000):

            #[num,1]当中，1是x,0是时间
            if self.data_odom_extratct[num,1]>0.00001: 
                self.data_start = num

                break #跳出循环

        #将最终数据筛选出来
        self.data_odom_extratct = self.data_odom_extratct[(self.data_start):,:]

        #print出来
        #print("self.data_odom_extratct")
        #print(self.data_odom_extratct[0:200,:])

        return self.data_odom_extratct