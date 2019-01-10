#定义一个controller 
#作用 1.接收odom的数据 2.处理odom数据生成控制量u 3. 将控制量u发送出去
class control:
    # 初始化(只初始化一次）
    def __init__(self):
        #时间初始化
        self.time_start = datetime.datetime.now()

        #相关参数初始化
        self.time = 0 #有效初始化
        self.v_d = 1 #初始化速度
        self.w_d = np.pi/10 #初始化转动角速度(完成时间：5s）
        self.line = 20 #初始化直线轨迹长度

        self.x_d = np.zeros(3) #初始化理想轨迹
        self.x_init = np.zeros(3) #初始化实际轨迹
         
    #生成x_init以及x_d
    def data_generate(self,data):
        #----------1.时间相关----------
        #现在时间
        self.time_now = datetime.datetime.now()

        #获取和开始时间的时间差 (问题：是否需要放在这里？）
        self.delta_time = self.time_now - self.time_start

        #----------2.实际轨迹----------
        x_time = data[0] #实际时间
        self.x_init[0] = data[1] #实际x坐标
        self.x_init[1] = data[2] #实际y坐标
        self.x_init[2] = data[3] #实际theta坐标

        #----------3.期望轨迹----------
        #delta_time/100取余数
        self.pure_delta_time = self.delta_time % 100 
        
        #获得理想路径
        if 0 < self.pure_delta_time <20 : #直线
            self.x_d[0]= self.pure_delta_time*self.v_d # x坐标
            self.x_d[1]= 0 # y坐标
            self.x_d[2]= 0 # theta坐标
        elif 20 <= self.pure_delta_time <25 : #转弯
            self.x_d[0]=20
            self.x_d[1]=0
            self.x_d[2]=self.w_d*(self.pure_delta_time-20)
        elif 25 <= self.pure_delta_time <45 : #直线
            self.x_d[0]=20
            self.x_d[1]=(self.pure_delta_time-25)*self.v_d
            self.x_d[2]=np.pi/2
        elif 45 <= self.pure_delta_time <50: #转弯
            self.x_d[0]=20
            self.x_d[1]=20
            self.x_d[2]=np.pi/2 + self.w_d*(self.pure_delta_time-45)
        elif 50 <= self.pure_delta_time <70: #直线
            self.x_d[0]= 20 - (self.pure_delta_time-25)*self.v_d
            self.x_d[1]=20
            self.x_d[2]=np.pi
        elif 70 <= self.pure_delta_time <75: #转弯
            self.x_d[0]=0
            self.x_d[1]=20
            self.x_d[2]=np.pi+ self.w_d*(self.pure_delta_time-70)
        elif 75 <=self.pure_delta_time <95: #直线
            self.x_d[0]=0
            self.x_d[1]=20 - (self.pure_delta_time-75)*self.v_d
            self.x_d[2]=3*np.pi/2
        elif 95 <= self.pure_delta_time <100:
            self.x_d[0]=0
            self.x_d[1]=0
            self.x_d[2]=3*np.pi/2 + self.w_d*(self.pure_delta_time-95)

        return self.x_init,self.x_d #将相关数据返回

    def callback(self,data):
        #初始化time(每30帧取1帧作为状态数据）
        if self.time % 30 == 0:
            #1.生成此时刻的x_init、以后30秒的x_d
            x_init,x_d = self.data_generate(data) #调用data_generate函数,

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
