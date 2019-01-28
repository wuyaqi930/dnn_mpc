#coding=utf-8
#coding=utf-8
#-------------导入相关安装包-------------
#import torch
import numpy as np
from scipy.optimize import minimize
#import matplotlib.pyplot as plt
#import reload_dnn as reload #载入训练网络的package 
import time #导入时间
import datetime

#-------------导入ros的package-------------
import rospy
from nav_msgs.msg import Odometry #接收里程计信息
from tf.transformations import euler_from_quaternion, quaternion_from_euler #将接收到的四元数转化为转角信息

# 定义一个模型预测控制的类 
class MPC:

    #------------1.定义数据传入函数------------
    def __init__(self,data):
    #------------初始化赋值------------
        self.data = data
    #------------初始化相关数值------------
        self.quaternion = np.zeros(4) #四元数
        self.euler_angle = np.zeros(3) #转角信息
        self.position = np.zeros(3) #位置信息
        self.period = 300 #走一条直线花的时间
        self.u = np.zeros(2)
        self.error = 0 #和理想轨迹的误差大小

        self.turn_flag = 0 #是否转动过
        self.straight_flag = 0 #是否直线运动过

    #------------2.开始优化------------
    def optimizer(self):
        #获取此时的position
        self.position(self.data)

        #获取此时的orientation
        self.orientation(self.data)

        #执行运动
        self.control()
        
    #------------3.相关控制函数------------
    #定义发送数据的函数:只发一次（可以考虑发很多次，最终要看效果如何）
    def orientation(self,data):
        #获取此时的orientation
        self.quaternion[0] = data.pose.pose.orientation.x #四元数
        self.quaternion[1] = data.pose.pose.orientation.y #四元数
        self.quaternion[2] = data.pose.pose.orientation.z #四元数
        self.quaternion[3] = data.pose.pose.orientation.w #四元数
    
    def position(self,data):
        #获取此时的position
        self.position[0] = data.pose.pose.position.x #坐标
        self.position[1] = data.pose.pose.position.y #坐标
        self.position[2] = data.pose.pose.position.z #坐标

    def control(self):
        #orientation的四元数转化为欧拉角
        self.euler_angle = euler_from_quaternion(self.quaternion)

        self.yaw = self.euler_angle[2] #获取yaw角度
        
        #对应的轨迹
        #获取现在的时间
        self.timenow = datetime.datetime.now()

        #计算间隔时间
        self.delta_time = (self.timenow - self.starttime).total_seconds()

        if self.delta_time < self.period: #第一条直线
            #--------有误差时--------
            #计算和理想轨迹的误差
            # self.error = np.abs(self.position[1])
            self.error = self.position[1]

            #矫正误差
            if self.error > 0.2: #在左边
                #转到合适角度
                if self.yaw > (-np.pi/4) or self.yaw < (-np.pi/2): # 这个判断有点问题
                    self.turn(self.position[1]) # y = self.position[1]
                
                #直线运动(减小误差)
                if self.turn_flag ==0: #没发生转动就直行
                    self.go_straight() #！！！：是否来得及执行

            elif self.error < -0.2 : #在右边
                #转到合适角度
                if self.yaw > (np.pi/2) or self.yaw < (np.pi/4):
                    self.turn(self.position[1])

                #直线运动(减小误差)
                if self.turn_flag ==0: #没发生转动就直行
                    self.go_straight() #！！！：是否来得及执行
            
            #--------没误差时--------
            if self.turn_flag == 0 and self.straight_flag == 0: #之前没发生直行或者转动才执行
                #直线运动(减小误差)
                self.go_straight()

            #--------flag重置--------
            self.turn_flag = 0 #所有程序执行完之后重置flag
            self.straight_flag = 0

        elif self.period <= self.delta_time < 2*self.period: #第二条直线
        
        elif 2*self.period <= self.delta_time < 3*self.period #第三条直线

        elif 3*self.period <= self.delta_time < 4*self.period #第四条直线

    #转弯
    def turn(self,position_y):

        if position_y > 0 :#在左边
            #判断此时的转角
            if self.yaw > (-np.pi/4): #顺时针转动
                #定义线速度角速度
                self.u[0] = 0 #线速度为零
                self.u[1] = 1 #角速度为1，右转一圈

                #发送控制量
                self.data_publish(self.u)

                #表明数据已发送
                self.turn_flag = 1
            elif self.yaw <(-np.pi/2): #逆时针转动
                #定义线速度角速度
                self.u[0] = 0 #线速度为零
                self.u[1] = -1 #角速度为1，左转一圈

                #发送控制量
                self.data_publish(self.u)

                #表明数据已发送
                self.turn_flag = 1
        elif position_y < 0:#在右边
            #判断此时的转角
            if self.yaw > (np.pi/2): #顺时针转动
                #定义线速度角速度
                self.u[0] = 0 #线速度为零
                self.u[1] = 1 #角速度为1，右转一圈

                #发送控制量
                self.data_publish(self.u)

                #表明数据已发送
                self.turn_flag = 1
            elif self.yaw <(np.pi/4): #逆时针转动
                #定义线速度角速度
                self.u[0] = 0 #线速度为零
                self.u[1] = -1 #角速度为1，左转一圈

                #发送控制量
                self.data_publish(self.u)

                #表明数据已发送
                self.turn_flag = 1

    def data_publish(self,u):
        #初始化publisher
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=10) # topic和消息类型需要和肖岸星商量确定一下 
        
        #初始化twist
        twist = Twist()

        # 给twist赋值
        twist.linear.x = u[0] #线速度
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0 #角速度
        twist.angular.y = 0.0
        twist.angular.z = u[1]

        #发送twist
        pub.publish(twist) #将数据发送出去

    

    #直线运动
    def go_straight(self):
        #定义线速度角速度
        self.u[0] = 1 #线速度为零
        self.u[1] = 0 #角速度为1，右转一圈

        #发送控制量
        self.data_publish(self.u)

        #表明数据已发送
        self.straight_flag = 1

    
    
        
    # # 优化函数
    # def optimizer(self):
    #     # #调试 ：初始化代码 
    #     # rospy.init_node('mpc_line')

    #     #订阅话题
    #     rospy.Subscriber("odom", Odometry, self.callback)

    #     #始终订阅话题
    #     rospy.spin()

    #     time.sleep(1)

    # #回调函数
    # def callback(self, data): #应该是1s 1次
    #     #获取此时的position
    #     self.position(data)

    #     #获取此时的orientation
    #     self.orientation(data)

    #     #执行运动
    #     self.control()

    