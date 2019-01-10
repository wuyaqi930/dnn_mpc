#------------ros的python的package------------
#import rospy
#import 要添加自定义的消息类型 (很重要）
#------------导入官方的package------------
import numpy as np
from scipy.optimize import minimize
import torch 
import datetime
#------------导入自己的package------------
import mpc 
import plot
import dnn
import reload_dnn as reload
import data_filter 
import contorller


if __name__ == '__main__':
    #初始化class
    data = contorller.control()

    #接收数据
    data.data_receive()