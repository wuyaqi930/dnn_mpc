#------------导入官方的package------------
import numpy as np
from scipy.optimize import minimize
import torch 
#------------导入自己的package------------
import mpc 
import plot
import dnn
import reload_dnn as reload

#------------初始化相关数值------------

prediction_horizon = 30 #预测范围是10 

#state_number = 3 #状态量有两个
state_number = 3 #状态量有两个

input_number = 2 #输入量只有两个，线速度以及角速度


#------------初始化x_d : 30*3的矩阵------------
#x_d = 10*np.random.rand( prediction_horizon,state_number ) #初始化理想状态量（列向量）随机数

#sin=np.linspace(2, 5, num=prediction_horizon)
#x_d = 10*np.sin( sin ) #初始化理想状态量（列向量）随机

sin_1=np.linspace(2, 5, num=prediction_horizon)
sin_2=np.linspace(3, 6, num=prediction_horizon)
sin_3=np.linspace(4, 7, num=prediction_horizon)

x_d = 10*np.sin( [sin_1,sin_2,sin_3] ) #初始化理想状态量（列向量）随机数
x_d = x_d.T #构造30*3的矩阵

#x=np.zeros((prediction_horizon,state_number))#初始化实际状态量（列向量)为零
x= np.random.rand(prediction_horizon,state_number)#初始化实际状态量（列向量)为随机数

Q=1000*np.eye(prediction_horizon) #初始化Q（需要半正定）

R=10*np.eye(prediction_horizon) #初始化R（需要正定）

#u = np.zeros((prediction_horizon,input_number)) #初始化实际状态量（列向量）
u = np.random.rand(prediction_horizon,input_number) #初始化实际状态量（列向量）为随机数


#------------调用MPC类进行求解------------

#初始化MPC类
x = mpc.MPC(x,x_d,prediction_horizon,state_number,input_number)

#求解:返回状态X、控制量u
x,u = x.optimize()


##------------结果绘图------------
##初始化
#polt_it = plot.plot(x,x_d,prediction_horizon,state_number)

##调用绘图函数
#polt_it.draw()



##------------DNN------------
#if __name__ == '__main__':

    #torch.manual_seed(1)    # reproducible
    
    ##将数据载入并转化成numpy
    #data_odom = np.loadtxt('odom.txt',dtype= 'str',skiprows=0,delimiter=",")

    ##初始化DNN类
    ##dnn = dnn.neural_network(data_odom,15000,20)
    #dnn = dnn.neural_network(data_odom,15000,20)

    ##调用优化类 
    #dnn.dnn()

    ##结果画图
    #dnn.plot_loss()
    #dnn.plot_error()

    ##------------保存网络之后------------
    ##重新载入网络
    ##net = dnn.restore_params()
    ##net = reload.reload_dnn()

    ##reload = net.load_dnn()

    
    ##x=np.random.random((1,5))

    ##print("x")
    ##print(x)

    ##x_torch = torch.from_numpy(x)

    ##print("x")
    ##print(x)

    ##y = reload(x_torch.float())

    ##print("y")
    ##print(y)

    ##print("reload")
    ##print(reload)

     

    
    ##产生训练集、测试集
    #data_train, data_test, label_train, label_test=dnn.data_loader(
    

