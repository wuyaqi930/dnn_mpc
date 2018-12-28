##将数据通过绘图工具绘制出来
import matplotlib.pyplot as plt
import numpy as np

class plot :

    #------------1.初始化------------
    def __init__(self,x,x_d,prediction_horizon,state_number):
        self.x = x 
        self.x_d = x_d 
        self.prediction_horizon = prediction_horizon

        #将状态量reshape
        self.x=np.reshape(self.x,(prediction_horizon,state_number) )
        self.x_d=np.reshape(self.x_d,(prediction_horizon,state_number) )


    def draw(self):
        #绘图
        X = np.linspace(1, self.prediction_horizon, self.prediction_horizon, endpoint=True)

        plt.plot(X,self.x,label="x")
        plt.plot(X,self.x_d,label="x_d")
        plt.show()

        return 0