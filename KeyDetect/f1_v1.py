import numpy as np
import time
from pyfranka.franka_pybind import (MotionGenerator ,
                                    CartesianVelocities,
                                    CartesianVelocitiesFinished, 
                                    Frame ,
                                   )
from pyfranka.franka_pybind import Torques, TorquesMotionFinished
from basic.load_franka import robot, model
from basic.rigid import *
from pedal_detect import DetectPedal

# from get_database125 import *
from copy import deepcopy
from multiprocessing import shared_memory
import multiprocessing as mp
from calTarget import *
from calTarget import Redis_API
from atomics import atomicview, INT
from circularbuffer_api import CircularBuffer
import os
import pinocchio as pin
from numpy import sin, cos
from enum import Enum  
np.set_printoptions(precision=5,suppress=True)


'''
    franka-spine
    : 由rokae中的spine而来。
    分为6部分：
    Initialization
    Prepare data
    Save data : 包括机械臂q,arm光球和目标位姿。
    Zero gravity
    Auto running
    Main
    
'''
class Robot_mode(Enum):
    stop=0
    init=1
    grav=2
    auto=3
#############################################################
#-initialization
red = Redis_API('192.168.31.179',db=3)
file_path="data"  #本地
# file_path='/media/surgicalai/7E402D42402D028F/rokae_data' #u盘
name=time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
file_name=os.path.join(file_path,name)
os.mkdir(file_name)
share_list=shared_memory.ShareableList([False,file_name])
robot_count=0

qt_name="franka"
exists_file="/dev/shm/franka"
a = np.zeros((20,4),dtype=np.float64) 
qt = shared_memory.SharedMemory(create=True,size=a.nbytes,name=qt_name)
get_database = np.ndarray(a.shape, dtype=a.dtype, buffer=qt.buf)
n_value = 4
p_value = 5
#-初始给值 
center,normal = get_tailpx(red)
data =  np.array(red.read("rt_plan_line")).ravel()
nt = data[3:6]
pt =  data[:3]*0.001
get_database[n_value,:3] = nt
get_database[p_value,:3] = pt
#-tip in end
N,P,end2tip,X = cal_end2tip(red)
inv_end2tip = np.linalg.inv(end2tip)
# print('n,p= ',N,P, X)
#-
robot_status = mp.Value("i",0)
# grav_status = mp.Value("i",0)
# auto_status = mp.Value("i",0)
# init_status = mp.Value("i",0)
red.write("robot_mode",'stop')
#-initialization
#############################################################

#############################################################
#-pedal_callback
def pedal_callback(btn: str, is_pressed: bool):
    print(f'btn={btn}, is_pressed={is_pressed}')
    if btn == "a":
        if is_pressed:
            robot_status.value=Robot_mode.grav.value
        else:
            robot_status.value=Robot_mode.stop.value
    elif btn == "s":
        if is_pressed:
            robot_status.value=Robot_mode.auto.value
        else: 
            robot_status.value=Robot_mode.stop.value

padel = DetectPedal(callback=pedal_callback)

#-prepare data
def read_redis_rokae_status():
    while True:
        robot_mode = red.read("robot_mode")
    #     if robot_mode=="stop":  # 结束所有运动
    #         robot_status.value=Robot_mode.stop.value
            
    #     if robot_mode=="auto":
    #         robot_status.value=Robot_mode.auto.value
            
    #     if robot_mode=="grav":
    #         robot_status.value=Robot_mode.grav.value
            
        if robot_mode=="init_state":
            robot_status.value=Robot_mode.init.value
        time.sleep(0.001)

def read_ball_data():
    no_ball=0
    while True:
        ball1 = red.read("Data")["Robot"]
        if ball1!=[]:
            ball1 = q2tform(ball1) # 四元数
            ball1[:3,3] *= 0.001
            get_database[0:4] = ball1          
            no_ball=0            
        else:
            # print('no ball1')
            if robot_status.value==Robot_mode.auto.value:
                no_ball+=1
            if no_ball==50:
                robot_status.value=Robot_mode.stop.value
                no_ball=0
                red.write("robot_error","stop running no ball")
                red.write("robot_mode","stop")            
        data =  np.array(red.read("rt_plan_line")).ravel()
        nt = data[3:6]
        pt =  data[:3]*0.001
        get_database[n_value,:3] = nt
        get_database[p_value,:3] = pt
        # time.sleep(0.001)

def update_target():
    while True:
        ball = get_database[0:4]
        
        p2 = get_database[p_value,:3]
        n2 = get_database[n_value,:3]
        n0 = normal
        p0 = center
        if ball[2,3]!=0:
            arc = ball[:3,:3]@n0@n2
            if abs(arc-1)>1e-18:
                ang = 0
            else:
                ang = np.rad2deg(np.arccos(arc))
            if abs(ang) > 90:
                n2 = -n2
                print('目标方向调整---当前夹角：', ang)
                ang = np.rad2deg(np.arccos(ball[:3,:3]@n0@n2))
                print('调整后夹角：',ang,ball[:3,:3]@n0@n2)
            else:
                # print('ball当前夹角= ',ang)
                pass
            # 计算target ball
            dr = turn_M_to_N(ball[:3,:3]@n0, n2)
            ballt = np.eye(4)
            ballt[:3,:3] =  dr @ ball[:3,:3]
            ballt[:3,3] = p2 - ballt[:3,:3]@p0 
            get_database[6:10] = ballt
            get_database[10:14] = np.linalg.inv(ballt)
        # time.sleep(0.1)
#-prepare data
#############################################################
        
#############################################################
#-save data
from functools import wraps
robot_qc = np.zeros(7)
def count_(func):   
    @wraps(func)
    def wrapfun(*args, **kwargs):
        # global a
        # nonlocal a# 使用 nonlocal 而不是 global，因为我们是在嵌套函数中修改
        # a+=1
        result=func(*args, **kwargs)
        record_time = time.time()    
        # r_q=np.array([record_time, robot_count, ] + r.getStateData6D(RtSupportedFields.jointPos_m))
        # record_time, robot_count, ] + r.getStateData6D(RtSupportedFields.jointPos_m))
        # r_dq=np.array([record_time, robot_count, ] + r.getStateData6D(RtSupportedFields.jointVel_m))
        # r_pc=np.array([record_time, robot_count, ] + poss)
        
        r_q = np.array([record_time, robot_count, ] + robot_qc)
        ndi_r = np.array([record_time, robot_count, ] + np.array(get_database[0:4]).reshape(-1).tolist())  
        ndi_t = np.array([record_time, robot_count, ] + np.array(get_database[n_value:p_value+1,:3]).reshape(-1).tolist())  
        
        robot_q.put(r_q)
        ndi_robot.put(ndi_r)
        ndi_target.put(ndi_t)
        
        # print(a,time_)     
        return result 
    return wrapfun

shm_file_index = shared_memory.SharedMemory(create=True, size=4)
def update_file_index():
    with atomicview(buffer=shm_file_index.buf[:], atype=INT) as a:
        a.inc()
        
def get_file_index():
    with atomicview(buffer=shm_file_index.buf[:], atype=INT) as a:
        i = a.load()
    return i

def save_data(bufs):
    # data= {}
    # for v in bufs:
    #     data[v.name] = v.get_all()
    #     print("name",v.name)
    i = get_file_index()

    path_name=share_list[1]
    np.savez(os.path.join(path_name, f"{i}.npz"),
            robot_q=bufs[0].get_all(),
            ndi_robot=bufs[1].get_all(),
            ndi_target=bufs[2].get_all())
    update_file_index()

def reader_process(bufs,save_data_time):
    t_s = time.time()
    t0=time.time()
    while True:
        # if share_list[0]:
        if robot_status.value!=Robot_mode.stop.value:
            t_e = time.time()
            if t_e - t_s > 2*save_data_time:
                t_s = time.time()
                
            if t_e - t_s > save_data_time:
                t_s = time.time()
                save_data(bufs)
#-save data
#############################################################
                
#############################################################
#-zero gravity               
dyn_par_extend = np.load('config/dyn_extend.npy')
dyn_par = dyn_par_extend[:, :-1]

#- 重力补偿相关参数
inertias = []
for i in range(7):
    inertias.append(pin.Inertia.FromDynamicParameters(dyn_par[i, :10]))

minus_g = pin.Motion(np.array([0, 0, 9.81]),np.zeros(3))
inv_trans_mats = get_robotParameter()


#- 定义回调函数
@count_
def control_callback(robot_state, period):
    global timestamp
    global robot_count,robot_qc
    robot_count += 1
    robot_qc = robot_state.q
    timestamp += period.toSec()

    #- 读取当前各轴角度，理想角度，速度，科氏力，重力值
    pos         = np.array(robot_state.q)
    coriolis    = np.array(model.coriolis(robot_state))
    tau_ext     = np.array(robot_state.tau_ext_hat_filtered)
    gravity     = np.array(model.gravity(robot_state, [0, 0, -9.81]))

    #- 根据当前各轴角度生成对应的重力
    fb = [None] * 7
    c = cos(pos)
    s = sin(pos)
    var_trans_mats = np.tile(np.eye(4), (7, 1, 1))
    var_trans_mats[:, 0, 0] = c
    var_trans_mats[:, 0, 1] = -s
    var_trans_mats[:, 1, 0] = s
    var_trans_mats[:, 1, 1] = c
    Ts = inv_trans_mats @ var_trans_mats
    
    #- 迭代
    acc = minus_g

    for i in range(7):
        t = pin.SE3(Ts[i, :3, :3], np.zeros(3))
        acc = t.actInv(acc)
        fb[i] = inertias[i] * acc


    for off in range(1, 7):
        t = pin.SE3(Ts[7-off])
        fb[6-off] += fb[7-off].se3Action(t)

    torque = [fb[i].np[-1] for i in range(7)]
    
    gravity_derived = np.array(torque)
    output = - gravity + gravity_derived  + 0.3*tau_ext  #- 纯重力补偿
    
    output = Torques((output + coriolis).tolist())
    if timestamp > 2000 or robot_status.value!=Robot_mode.grav.value:
        #- 如果时间戳超过设定极值
        return TorquesMotionFinished(output)
    return output 
#-zero gravity               
#############################################################

#############################################################
#-auto running
@count_
def vel_callback(robot_state, period):
    global timestamp, xs,st
    global robot_count,robot_qc
    robot_count += 1
    robot_qc = robot_state.q
     
    timestamp += period.toSec()
    if timestamp == 0:
        init_pos = np.array(robot_state.O_T_EE).reshape(4,4,order='F')
        xs[0] = init_pos[:3, 3]
        xs[2] = vee(log3(init_pos[:3, :3])) 
    posc = np.array(robot_state.O_T_EE).reshape(4,4,order='F')
    # 新的期望位置
    ball = get_database[:4]
    #-目标点改变*****************************************
    ballt = get_database[6:10]
    inv_ballt = get_database[10:14]
    #-******************************************
    dr_ball = inv_ballt @ ball
    dr_end0 = x @ dr_ball @ inv_x 
    
    #-使用规划xs
    poscr = exp3(hat(xs[2]))
    # poscr = posc[:3,:3]   #- check...
    rt = poscr @ dr_end0[:3,:3].T
    delta_p = rt @ dr_end0[:3,3]
    dr_end0 = poscr[:3,:3] @ dr_end0[:3,:3] @ poscr.T
    dr_end = vee(log3(dr_end0[:3,:3]))
    
    deg = np.rad2deg(np.linalg.norm(dr_end))
    ds = np.linalg.norm(delta_p)*1000
    #-a
    dxq1 = np.zeros((4,3))
    dxq1[0] = delta_p
    dxq1[1] = xs[1]  # 
    
    # dxq1[0] = np.zeros(3)  # pos unchanged
    # dxq1[1] = np.zeros(3)  # 
    dxq1[2] = dr_end #
    dxq1[3] = xs[3]
    
    a = - KKK @ dxq1   
    a = np.clip(a, -amax, amax)  
    xs = A @ xs + B @ a      
    
    # print(timestamp,ds,deg)
    output = np.append(xs[1] ,xs[3])   
    cartesian_vel_obj = CartesianVelocities(output.tolist())
    # if  timestamp>20 or (ds < 0.5 and deg < 0.5): 
    if robot_status.value!=Robot_mode.auto.value:  
        output = output*np.exp2(-st*0.05)
        print(timestamp,ds,deg,output,st) 
        st += 1
        if max(abs(output))<0.001:
            print('slow done.')
            return CartesianVelocitiesFinished(CartesianVelocities([0, 0, 0, 0, 0, 0]))
        else:  
            cartesian_vel_obj = CartesianVelocities(output.tolist())
    return cartesian_vel_obj
#-auto running
#############################################################

              
                
if __name__ == '__main__':
    data_size=[20000,20000]
    save_data_time=0.5
    w_ps = []
    bufs = []
    ##- robot + ndi 
    # rokae_q=CircularBuffer("rokae_q", (data_size[0], 8))
    # rokae_dq=CircularBuffer("rokae_dq", (data_size[0], 8))
    # rokae_p=CircularBuffer("rokae_p", (data_size[0], 18))
    # rokae_pc=CircularBuffer("rokae_pc", (data_size[0], 18))
    # bufs +=[rokae_q,rokae_dq,rokae_p,rokae_pc]
    robot_q = CircularBuffer("robot_q", (data_size[0], 9)) #- 7+2
    bufs +=[robot_q]
    ndi_robot=CircularBuffer("ndi_robot", (data_size[0], 18))
    ndi_target=CircularBuffer("ndi_target", (data_size[0], 8)) #- 6+2
    
    bufs +=[ndi_robot,ndi_target]
    
    reader = mp.Process(target=reader_process, args=(bufs,save_data_time, ))
    reader.start() 
    ##data
    
    read_redis = mp.Process(target=read_redis_rokae_status, name="read_redis_rokae_status")
    read_redis.start()
    update_data = mp.Process(target=read_ball_data, name="read_ball_data")
    update_data.start() 
    update_data2 = mp.Process(target=update_target, name="update_target")
    update_data2.start() 
    
    #-connection
    t_max = 60
    x = get_x(red)
    inv_x = np.linalg.inv(x)
    print("franka start...")
    while True:
        timestamp = 0
        if robot_status.value==Robot_mode.grav.value:
            print("start grav",robot_status.value)

            try:
                robot.robot_control(control_callback)
            except Exception as e:
                red.write("robot_error","error:"+e)
            
        if robot_status.value==Robot_mode.auto.value:
            print("start a",robot_status.value)
            try:
                center,normal = get_tailpx(red)
                
                st = 0
                timestamp = 0
                KKK,A,B = get_kk()
                xs = np.zeros((4, 3))
                dxq1 = np.zeros((4, 3))
                amax = 0.5 * 0.99
                robot.robot_control(cartesian_velocities_handle=vel_callback)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                red.write("robot_mode","error:"+e)
            red.write("robot_mode","stop")
            
        if robot_status.value==Robot_mode.init.value:
            print('Move to initial point...')
            q1 = np.load('config/q0.npy')
            try:
                mg = MotionGenerator(0.2, q1)
                robot.robot_control(joint_positions_handle=mg.operator)
                
                print('Finished at initial point.')
            except Exception as e:
                red.write("robot_error","error:"+e)
            red.write("robot_mode","stop")
        time.sleep(0.01)   
            
    
    
    try:
        [w.join() for w in w_ps]  
        reader.join()  
    except KeyboardInterrupt:  
        # 终止进程  
        [w.terminate() for w in w_ps] 
        reader.terminate()  
    finally:  
        save_data(bufs)
        # 关闭并删除共享内存块
        [buf.clear() for buf in bufs]
        shm_file_index.close()
        shm_file_index.unlink()
