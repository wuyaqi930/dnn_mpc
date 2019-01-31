#coding=utf-8

#-----------尝试使用非常暴力的方法实现有效的路径规划-----------
import numpy as np  
from scipy.optimize import brute 
import itertools
import time

#-------------导入ros有关的package-------------
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist 
from std_msgs.msg import Int16
from tf.transformations import euler_from_quaternion, quaternion_from_euler

#-------------初始化全局变量-------------
is_done=Int16()
is_done.data=0

#-------------定义一个模型预测的类型-------------
class MPC:

    #-------------1.初始化-------------
    def __init__(self,x_init,x_desire):
        #------------初始化赋值------------
        self.x_init = x_init #初始状态
        self.x_desire = x_desire #最终状态 

        #------------初始化参数------------
        self.Q = 100 #调节参数
    
    #------------定义目标代价函数------------
    def objective (self,x,x_desire):
        #做最终位置和预测位置之间的差值函数
        J = self.Q*np.power(x-x_desire,2) #二次型的样子

        return J

    #------------定义运动学函数------------
    def f(self,x_t,u_t): # 0 1 2 3 4 5 对应的效果怎么样，分别用一个if 

        if u_t == 0: #直行
            x_t_1=np.zeros(3) #输入输出是一维的变量，直接定义成数组就行，不要定义成矩阵

            
            x_t_1[0]= x_t[0]+ np.cos(x_t[2])*u_t[0] # 坐标x
            x_t_1[1]= x_t[1]+ np.sin(x_t[2])*u_t[0] # 坐标y
            x_t_1[2]= x_t[2]+ u_t[1] # 转角theta

        if u_t == 1: #左转一次再直行（暂定左转一次是十度）
            x_t_1=np.zeros(3) #输入输出是一维的变量，直接定义成数组就行，不要定义成矩阵
            
            x_t_1[0]= x_t[0]+ np.cos(x_t[2]+np.pi/18)*u_t[0] # 坐标x
            x_t_1[1]= x_t[1]+ np.sin(x_t[2]+np.pi/18)*u_t[0] # 坐标y
            x_t_1[2]= x_t[2]+ np.pi/18 + u_t[1] # 转角theta

        if u_t == 2: #左转两次再直行（暂定左转一次是十度）
            x_t_1=np.zeros(3) #输入输出是一维的变量，直接定义成数组就行，不要定义成矩阵
            
            x_t_1[0]= x_t[0]+ np.cos(x_t[2]+np.pi/9)*u_t[0] # 坐标x
            x_t_1[1]= x_t[1]+ np.sin(x_t[2]+np.pi/9)*u_t[0] # 坐标y
            x_t_1[2]= x_t[2]+ np.pi/9 + u_t[1] # 转角theta

        if u_t == 3: #右转一次再直行（暂定右转一次是十度）
            x_t_1=np.zeros(3) #输入输出是一维的变量，直接定义成数组就行，不要定义成矩阵
            
            x_t_1[0]= x_t[0]+ np.cos(x_t[2]-np.pi/18)*u_t[0] # 坐标x
            x_t_1[1]= x_t[1]+ np.sin(x_t[2]-np.pi/18)*u_t[0] # 坐标y
            x_t_1[2]= x_t[2]- np.pi/18 + u_t[1] # 转角theta
        
        if u_t == 4: #右转两次再直行（暂定右转一次是十度）
            x_t_1=np.zeros(3) #输入输出是一维的变量，直接定义成数组就行，不要定义成矩阵
            
            x_t_1[0]= x_t[0]+ np.cos(x_t[2]-np.pi/9)*u_t[0] # 坐标x
            x_t_1[1]= x_t[1]+ np.sin(x_t[2]-np.pi/9)*u_t[0] # 坐标y
            x_t_1[2]= x_t[2]- np.pi/9 + u_t[1] # 转角theta

        return x_t_1

    def position(self,u):
        #最终位置 假设只执行六次
            
        return self.objective(self.f (self.f(self.f(self.f(self.f(self.f(self.x_init,u[0]),u[1]), u[2]),u[3]),u[4]),u[5]),self.x_desire)    

    #结果转化为控制量
    def result_to_control(self,result):
        #创建控制量(总共六组)
        control = []
        temp = np.array([[0,0],[0,0],[0,0]])

        #创建循环进行数值转换
        for num in range(6):
            if result[num] == 0: #直行
                temp[0,0]= 1 
                temp[0,1]= 0

                #转化成list并赋值
                control.append(temp.tolist()[0])

                print("0")
            if result[num] == 1: #左转一圈
                #先转动
                temp[0,0]= 0 
                temp[0,1]= 1

                #再直行
                temp[1,0]= 1 
                temp[1,1]= 0

                # #转化成list并赋值
                # control.append(temp.tolist()[0:2])
                control = control +temp.tolist()[0:2]

                print("1")
            if result[num] == 2: #左转两圈
                #先转动
                temp[0,0]= 0 
                temp[0,1]= 1

                #再转动
                temp[1,0]= 0 
                temp[1,1]= 1

                #再直行
                temp[2,0]= 1 
                temp[2,1]= 0

                #转化成list并赋值
                #control.append(temp.tolist()[0:3])
                control = control +temp.tolist()[0:3] #直接加就行

                print("2")

            if result[num] == 3: #右转一圈
                #先转动
                temp[0,0]= 0 
                temp[0,1]= -1

                #再直行
                temp[1,0]= 1 
                temp[1,1]= 0

                #转化成list并赋值
                control.append(temp.tolist()[0:2])

                print("3")

            if result[num] == 4: #左转两圈
                #先转动
                temp[0,0]= 0 
                temp[0,1]= -1

                #再转动
                temp[1,0]= 0 
                temp[1,1]= -1

                #再直行
                temp[2,0]= 1 
                temp[2,1]= 0

                #转化成list并赋值
                control.append(temp.tolist()[0:3])

                print("4")

            #数值重置
            temp = np.array([[0,0],[0,0],[0,0]])
        
        return control

    #发送twist：一直发指令，直到一次运动结束之后就不发指令
    def callback_2(self,data): 
        #接受数据
        is_done.data= data.data #去data当中的数据

    #将控制量发布出去
    def control_publish(self,control):
        #初始化node
        rospy.init_node('mpc_brute', anonymous=True)

        #初始化publisher 
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=10) # topic和消息类型需要和肖岸星商量确定一下 
            
        #初始化twist
        twist = Twist()

        #将所有数据发送出去
        for i in range(5): #为了防止没有收到信息，连续发送五次控制命令
            # 给twist赋值（已经没有了四元素的含义了）
            twist.linear.x = np.asarray(control)[0,0] #线速度
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0 
            twist.angular.y = 0.0
            twist.angular.z = np.asarray(control)[0,1] #角速度

            #发送twist
            pub.publish(twist) #将数据发送出去

            print("twist",twist) 

    #优化函数
    def optimize(self):
        #产生所有可能
        ranges = (slice(0, 5, 1),) * 6  #只限制左转两次和右转两次（考虑计算能力的话，可以尝试多几种可能）

        #计算所有可能的cost function ,去最小值
        result = brute(self.position,ranges,disp=True, finish=None)

        #结果对应的控制量（六组控制量）
        control = self.result_to_control(result)

        #将控制量发布出去

        #初始化subscriber 
        rospy.Subscriber("is_done", Int16, self.callback_2)

        #初始化发布序列
        self.num=0

        #不断进行循环
        while 1:
            #接受到数据才执行
            if(is_done.data==1 and self.num<10):
                #发送制定序列的控制命令
                self.control_publish(control[self.num])
                
                #发送下一个
                self.num = self.num+1

                #将is_down复位
                print("is_done已重置")
                is_done.data = 0

            #延迟一下
            time.sleep(1)

        return result
    

