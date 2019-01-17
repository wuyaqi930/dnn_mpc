#coding=utf-8
import rospy

#-----------personal------------
import controller



if __name__ == '__main__':
    #init
    data = controller.control()

    #receive data
    data.data_receive()

    

