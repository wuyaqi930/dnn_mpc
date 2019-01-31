#coding=utf-8
#------------导入官方的package------------
import numpy as np
from scipy.optimize import minimize
import datetime
import time 

#------------导入ros的package------------
import rospy
from geometry_msgs.msg import Twist #导入需要的ros的package 
from nav_msgs.msg import Odometry

#------------导入personal的package------------
import mpc #导入相关的文件
import mpc_line #直线运动时候的相应规划




#定义一个controller 
#作用 1.接收odom的数据 2.处理odom数据生成控制量u 3. 将控制量u发送出去
class control:
    # 初始化(只初始化一次）
    def __init__(self):
        #时间初始化
        self.time_start = datetime.datetime.now()

        #相关参数初始化
        self.time = 0 #有效初始化
        self.prediction_horizon = 10 #初始化预测周期
        self.state_number = 3 #初始化状态量
        self.input_number = 2 #初始化输出量

        self.v_d = 1 #初始化速度
        self.w_d = np.pi/10 #初始化转动角速度(完成时间：5s）
        self.line = 20 #初始化直线轨迹长度

        self.positions = np.zeros(3) #位置信息

        # self.x_d = np.zeros(3) #初始化理想轨迹
        # self.x_init = np.zeros(3) #初始化实际轨迹

        self.x_d = np.zeros((self.prediction_horizon,self.state_number)) #初始化理想轨迹:预测周期为10,状态量为3
        self.x = np.zeros((self.prediction_horizon,self.state_number)) #初始化实际轨迹 
        self.x_init = np.zeros(self.state_number) #初始化初始轨迹

        #相关flag初始化
        self.is_mpc_line = 1 #直线版本mpc对应的flag
        self.is_mpc_target = 0 #任务版本mpc对应的flag
        
    #生成x_init以及x_d（ 通过校验 ）
    def data_generate(self,data):
        #----------1.时间相关----------
        #现在时间
        self.time_now = datetime.datetime.now()

        #获取和开始时间的时间差 (问题：是否需要放在这里？）
        self.delta_time = self.time_now - self.time_start

        #----------2.实际轨迹----------  (！！！注释：可能有问题）
        x_time = data.header.stamp #实际时间
        self.x_init[0] = data.pose.pose.position.x #实际x坐标
        self.x_init[1] = data.pose.pose.position.y #实际y坐标
        self.x_init[2] = data.pose.pose.orientation.z #实际theta坐标

        # # 将x_init赋值给x
        # self.x[0,:] = self.x_init #初始数值|剩下数值均为零

        #----------3.期望轨迹----------
        #delta_time/100取余数
        self.pure_delta_time = self.delta_time.seconds % 100 
        
        #获得理想路径：预测周期为10的时候效果会复杂很多
        #创建此时刻往后10帧的理想轨迹：调用 x_d_generate（）
        for num in range(self.prediction_horizon):
            self.x_d[num,:]=self.x_d_generate(self.pure_delta_time+num)

        return self.x_d #将相关数据返回

    #产生1帧对应的x_d（ 通过校验 ）
    def x_d_generate(self,pure_delta_time):
        #定义x_d_once
        x_d_once = np.zeros(3)
        #创建理想轨迹
        if pure_delta_time == 0: #到达原点
            x_d_once[0]= 0 # x坐标
            x_d_once[1]= 0 # y坐标
            x_d_once[2]= 0 # theta坐标
        elif 0 < pure_delta_time <20 : #直线
            x_d_once[0]= pure_delta_time*self.v_d # x坐标
            x_d_once[1]= 0 # y坐标
            x_d_once[2]= 0 # theta坐标
        elif 20 <= pure_delta_time <25 : #转弯
            x_d_once[0]=20
            x_d_once[1]=0
            x_d_once[2]=self.w_d*(pure_delta_time-20)
        elif 25 <= pure_delta_time <45 : #直线
            x_d_once[0]=20
            x_d_once[1]=(pure_delta_time-25)*self.v_d
            x_d_once[2]=np.pi/2
        elif 45 <= pure_delta_time <50: #转弯
            x_d_once[0]=20
            x_d_once[1]=20
            x_d_once[2]=np.pi/2 + self.w_d*(pure_delta_time-45)
        elif 50 <= pure_delta_time <70: #直线
            x_d_once[0]= 20 - (pure_delta_time-50)*self.v_d
            x_d_once[1]=20
            x_d_once[2]=np.pi
        elif 70 <= pure_delta_time <75: #转弯
            x_d_once[0]=0
            x_d_once[1]=20
            x_d_once[2]=np.pi+ self.w_d*(pure_delta_time-70)
        elif 75 <=pure_delta_time <95: #直线
            x_d_once[0]=0
            x_d_once[1]=20 - (pure_delta_time-75)*self.v_d
            x_d_once[2]=3*np.pi/2
        elif 95 <= pure_delta_time <100:
            x_d_once[0]=0
            x_d_once[1]=0
            x_d_once[2]=3*np.pi/2 + self.w_d*(pure_delta_time-95)

        return x_d_once

    #判断是mpc_line还是mpc_target
    def mpc_desicion(self,data):
        #获取此时的position
        self.positions[0] = data.pose.pose.position.x #x坐标
        self.positions[1] = data.pose.pose.position.y #y坐标
        self.positions[2] = data.pose.pose.position.z #z坐标
        
        #定义坐标
        x = self.positions[0]
        y = self.positions[1]

        #判断此时的位置(使用经验值)
        if 19.5 < x < 20.5 and 19.5 < y < 20.5 : # mpc-target 
            self.is_mpc_line==0
            self.is_mpc_target==1
        else : # mpc-line
            self.is_mpc_line==1
            self.is_mpc_target==0

    # 将之前的数据clear掉
    def clear(self):
        #flag的reset 
        self.is_mpc_line=0
        self.is_mpc_target=0

    #定义回调函数
    def callback(self,data):
        #初始化time(每30帧取1帧作为状态数据）
        if self.time % 30 == 0:#一秒一次
            # 1.生成此时刻的x_init、以后30秒的x_d
            x_d = self.data_generate(data) #一旦接收到相应数据，调用data_generate函数,
            x= self.x
            x_init = self.x_init 

            #2.判断是mpc_line 还是 mpc_target 
            self.mpc_desicion(data) # 根据经验值来判断

            #3.将data丢给MPC计算
            #3.1 直线版本mpc
            if self.is_mpc_line==1:
                model_predict=mpc_line.MPC(data) #提示：x_init,x_d还没有
                model_predict.optimizer()

            #3.2 任务版本mpc
            if self.is_mpc_line==1:
                model_predict = mpc.MPC(x,x_init,x_d,self.prediction_horizon,self.state_number,self.input_number) #提示：x_init,x_d还没有

                #3.2.1求解MPC（控制量生成）
                self.x,self.u = model_predict.optimize() #将求解得出的实际状态量和实际控制量赋值给self.x,self.u

            #4. 将之前的flag来reset
            self.clear()

            self.time = self.time+1 # 自增操作
        else:
            #将之前的flag来reset
            self.clear()

            self.time = self.time+1 # 自增操作
        

    #---------------subscriber && publisher---------------
    
    #定义接收数据的函数
    def data_receive(self):
        #初始化node
        rospy.init_node('data_receive', anonymous=True)

        #初始化开始时间
        self.starttime = datetime.datetime.now()

        #初始化subscriber
        #rospy.Subscriber("odom", String, callback)  # topic和消息类型需要和肖岸星商量确定一下 
        rospy.Subscriber("odom", Odometry, self.callback)  # topic和消息类型需要和肖岸星商量确定一下 

        #保持python活跃
        rospy.spin()

    #定义发送数据的函数:只发一次（可以考虑发很多次，最终要看效果如何）
    def data_publish(self,u):
        #初始化publisher
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=10) # topic和消息类型需要和肖岸星商量确定一下 
        
        #初始化twist
        twist = Twist()

        # 给twist赋值
        twist.linear.x = u[0,0] #线速度
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0 #角速度
        twist.angular.y = 0.0
        twist.angular.z = u[0,1]

        # print("twist")
        # print(twist)

        #发送twist
        pub.publish(twist) #将数据发送出去

