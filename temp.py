#------------ros的python的package------------
#import rospy
#import 要添加自定义的消息类型 (很重要）
#------------导入官方的package------------
import numpy as np
from scipy.optimize import minimize
import torch 
#------------导入自己的package------------
import mpc 
import plot
import dnn
import reload_dnn as reload
import data_filter 

#定义一个data_receive的class
class receive_data:
    # 初始化
    def __init__(self):
        self.time = 0 #有效初始化

    #生成x_init以及x_d
    def data_generate(self,data):
        self.ttt = 0 #无效初始化

    def callback(self,data):
        #初始化time(每30帧取1帧作为状态数据）
        if self.time % 30 == 0:
            #1.生成此时刻的x_init、以后30秒的x_d
            self.data_generate(data) #调用data_generate函数

            #今晚工作！！！


            #2.将data丢给MPC计算
            
            #2.1初始化mpc 
             #2.1.1初始化mpc的各个参数
            prediction_horizon = 10 #预测范围是10 
            state_number = 3 #状态量有两个
            input_number = 2 #输入量只有两个，线速度以及角速度
             #2.1.2实际初始化过程
            model_predict = mpc.MPC(x_init,x_d,prediction_horizon,state_number,input_number) #提示：x_init,x_d还没有

            #2.2求解MPC（控制量生成）
            self.x,self.u = model_predict.optimize() #将求解得出的实际状态量和实际控制量赋值给self.x,self.u

            #将控制信号publish出去
            self.data_publish(self.u) # self.u是控制量

            self.time = self.time+1 # 自增操作
        else:
            self.time = self.time+1 # 自增操作

    #---------------subscriber && publisher---------------
      
    #定义接收数据的函数
    def data_receive(self):
        #初始化node
        rospy.init_node('data_receive', anonymous=True)

        #初始化subscriber
        rospy.Subscriber("odom", String, callback)  # topic和消息类型需要和肖岸星商量确定一下 

        #保持python活跃
        rospy.spin()

    #定义发送数据的函数
    def data_publish(self,u):
        #初始化publisher
        pub = rospy.Publisher('cmd_vel', String, queue_size=10) # topic和消息类型需要和肖岸星商量确定一下 

        #初始化node
        rospy.init_node('data_publish', anonymous=True)

        #设置发送时间
        rate = rospy.Rate(10) # 10hz 可能还需要调整
        while not rospy.is_shutdown():
            pub.publish(u) #将数据发送出去
            rate.sleep() #控制频率

if __name__ == '__main__':
    #初始化class
    data = receive_data()

    #接收数据
    data.data_receive()