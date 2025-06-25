import os
import numpy as np

from detect_balls2 import detect_spheres_from_dicom
from cal_math import match_balls_by_geometry, rigidTransform3D, transform3d, calculate_angle, Q2M




class AppPlanVeri:
    
    def __init__(self, npz_path: str):
        self._data_dict = dict()
        if os.path.exists(npz_path) and npz_path.endswith(".npz"):
            self._data_dict = dict(np.load(npz_path))
        # print(self._data_dict)
        
    def find_balls_at_ct(self):
        if not self._data_dict:
            return
        data = self._data_dict["ct_volume"]
        pixel_spacing = self._data_dict["ct_pixel_spacing"]
        thickness = self._data_dict["ct_thickness"]
        pos_ori = self._data_dict["ct_pos_ori"]
        # pos_ori = [-94, 29.456, -98.176]
        steel_balls = detect_spheres_from_dicom(data=data, pixel_spacing=pixel_spacing, thickness=thickness, pos_ori=pos_ori, 
                                        output_points=10, r_range=0.75, rlen=5, rmax=5)
        print("Detected steels:")
        model_steel = self._data_dict["model_steel"]
        ct_steel_balls, _, _ = match_balls_by_geometry(model_steel, steel_balls[:, :3], verbose=True)
        return ct_steel_balls
    
    def cal_t0(self):
        if not self._data_dict:
            return
        model_steel = self._data_dict["model_steel"]
        model_light = self._data_dict["model_light"]
        rom_light = self._data_dict["rom_light"]
        ct_steel_balls = self.find_balls_at_ct()
        T_model_to_ct, _ = rigidTransform3D(model_steel.T, ct_steel_balls.T)
        ct_light_balls = transform3d(model_light, T_model_to_ct)
        t1, _ = rigidTransform3D(rom_light.T, ct_light_balls.T)  # 光球到ct的位姿
        t0 = np.linalg.inv(t1)  # ct到光球的位姿
        return t0, t1
    
    def turn_M_to_R_(self, M0):
        M=M0/np.linalg.norm(M0)
        p=np.sqrt(1-M[0]**2)
        q=np.sqrt(1-M[1]**2)
        cosr=M[2]/q/p
        sinr=-M[0]*M[1]/p/q
        gamma=np.arctan2(sinr,cosr)
        beta1=np.arctan(-p*sinr/(p*cosr+q))
        alpha1=beta1+gamma
        x1=p*np.cos(alpha1)
        y1=q*np.cos(beta1)
        beta2=beta1-np.pi
        alpha2=beta2+gamma
        x2=p*np.cos(alpha2)
        y2=q*np.cos(beta2)
        if x1+y1<x2+y2:
            r1=np.array([x2,p*np.sin(alpha2),M[0]])
            r2=np.array([-q*np.sin(beta2),y2,M[1]])
        else:
            r1=np.array([x1,p*np.sin(alpha1),M[0]])
            r2=np.array([-q*np.sin(beta1),y1,M[1]])
        r3=np.cross(r1,r2)
        result=np.array([r1,r2,r3])
        return result

    def pos2tra(self, pos, n):
        p = np.eye(4)
        p[:3, :3] = self.turn_M_to_R_(n)
        p[:3, 3:] = pos.reshape(3,1)
        
        return p
    
    def get_robot_tail_pose(self, tra_inv, ball, tail_X, robot_drf):
        #  """通过机械臂光球位置计算法向量，针尖位置，位姿"""
        n = ball[:3, :3] @ np.array(tail_X)
        pos = (ball @ np.array(robot_drf + [1,]))[:3]
        
        # 转换坐标系                        
        r_n = (tra_inv @ np.array((n.tolist() + [0, ])))[:3]
        r_p = (tra_inv @ np.array((pos.tolist() + [1, ])))[:3]
        
        # d = get_degree(r_n, np.array([0,0,1]))
        # if d > 90:
        #     r_n = -r_n
        if np.dot(n, tail_X) > 0:
            r_n = -r_n
            
        p = self.pos2tra(r_p, r_n.ravel())
        
        return n, r_n, r_p, p

    def cal_plan_ver(self):
        if not self._data_dict:
            return
        model_robot = self._data_dict.get("model_robot")
        rom_robot = self._data_dict.get("rom_robot")
        T_model_to_rom, _ = rigidTransform3D(model_robot[:4].T, rom_robot.T)
        hole_pos = transform3d(model_robot[4:], T_model_to_rom)
        hole_dis = hole_pos[0] - hole_pos[1]  # 孔上，孔下
        tail_x = hole_dis / np.linalg.norm(hole_dis)
        tail_p = hole_pos[0]
        # tail_x = self._data_dict["tail_x"]
        # tail_p = self._data_dict["tail_p"]
        # print(f"tail_x={tail_x}")
        # print(f'tail_p={tail_p}')
        # T_robot = self._data_dict["robot"]  # 机器人到NDI的4次矩阵
        # T_spine = self._data_dict["spine"]  # 光球到NDI的4次矩阵
        # print(f"t_robot={T_robot}")
        # print(f"t_spine={T_spine}")
        spine = self._data_dict["spine"]
        robot = self._data_dict["robot"]
        T_spine = Q2M(spine[3:], spine[:3])
        T_robot = Q2M(robot[3:], robot[:3])
        
        T0, T1 = self.cal_t0()
        # print(f"spine={T_spine}, robot={T_robot}, T0={T0}")
        # tra_inv = np.linalg.inv(T_spine @ T0)
        # n, r_n, r_p, pose = self.get_robot_tail_pose(tra_inv, T_robot, tail_x.tolist(), tail_p.tolist())
        # cal_pose1 = r_p + r_n * 140
        # print("cal 针尖", cal_pose1, pose[:3, 3])  # 计算出来的针尖

        ct_robot_x = (T1@np.linalg.inv(T_spine)@T_robot)[:3,:3]@np.array(tail_x)
        ct_robot_p = transform3d(np.array(tail_p).reshape(1,3),T1@np.linalg.inv(T_spine)@T_robot) # 针尾
        ct_robot_p1 = ct_robot_p - ct_robot_x * 150  # 针尖
        print(ct_robot_x, ct_robot_p, ct_robot_p1)
        
        plan_pose = self._data_dict["ct_plan_pose"]
        print(f"plan pose={plan_pose}")
        
        # plan_x_dis = plan_pose[0] - plan_pose[1]
        # ct_plan_x = plan_x_dis / np.linalg.norm(plan_pose[0])
        # plan_to_veri_dis = np.linalg.norm(plan_pose[0] - cal_pose1)
        # plan_to_veri_angle = calculate_angle(plan_x_dis, r_n)
        # print(f"plan to veri dis = {plan_to_veri_dis}")
        # print(f"plan to veri angle={plan_to_veri_angle}")
        
        plan_x_dis = plan_pose[1] - plan_pose[0]
        ct_plan_x = plan_x_dis / np.linalg.norm(plan_x_dis)
        dis_err = np.linalg.norm(plan_pose[0] - ct_robot_p1)
        angle_err = calculate_angle(ct_plan_x, ct_robot_x)
        print(f"plan to veri dis = {dis_err}")
        print(f"plan to veri angle={angle_err}")
        return dis_err, angle_err


if __name__ == "__main__":
    npz_path = "D:/VSCODES/DetectPoints/精度验证/CT-MP25080-250606-8.npz"
    
    app = AppPlanVeri(npz_path=npz_path)
    app.cal_plan_ver()
