import os
import numpy as np

from detect_balls import detect_spheres_from_dicom
from cal_math import match_balls_by_geometry, rigidTransform3D, transform3d


data_path = os.path.join(os.path.expanduser('~'), 'spine/dataStore/npz_path')


class AppPlanVeri:
    
    def __init__(self, npz_file: str):
        self._data_dict = dict()
        npz_path = os.path.join(data_path, npz_file)
        if os.path.exists(npz_path) and npz_file.endswith(".npz"):
            self._data_dict = dict(np.load(npz_path))
            
    def _calculate_angle(self, v1, v2, degrees=True):
        """计算两个三维向量的夹角，返回角度或弧度"""
        v1 = np.array(v1)
        v2 = np.array(v2)

        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        
        return np.degrees(theta) if degrees else theta
    
    # 寻找steel_balls
    def find_steel_balls_and_cal_t0(self):
        # CT_DATA
        data = self._data_dict.get("ct_volume")
        pixel_spacing = self._data_dict.get("ct_pixel_spacing")
        thickness = self._data_dict.get("ct_thickness")
        pos_ori = self._data_dict.get("ct_pos_ori")
        # MODEL_DATA
        model_steel = self._data_dict.get("model_steel")
        model_light = self._data_dict.get("model_light")
        rom_light = self._data_dict.get("rom_light")
        # CAL_DATA
        save_ct_steel_balls = self._data_dict.get("cal_ct_steel_balls")
        save_ct_light_balls = self._data_dict.get("cal_ct_light_balls")
        save_t0 = self._data_dict.get("cal_ct2ball_mat")
        print("save cal data:")
        print(f"ct steel balls: {save_ct_steel_balls}")
        print(f"ct light balls: {save_ct_light_balls}")
        print(f"T0: {save_t0}")
        print("================")
        # 重新识别钢球
        steel_balls = detect_spheres_from_dicom(data=data, pixel_spacing=pixel_spacing, thickness=thickness, pos_ori=pos_ori, output_points=10,
                                                r_range=2.5, rlen=5, rmax=4)
        ct_steel_balls, _, _ = match_balls_by_geometry(model_steel, steel_balls[:, :3], verbose=True)
        # 重新计算T0
        T_model_to_ct, _ = rigidTransform3D(model_steel.T, ct_steel_balls.T)
        ct_light_balls = transform3d(model_light, T_model_to_ct)
        T1, _ = rigidTransform3D(rom_light.T, ct_light_balls.T)
        T0 = np.linalg.inv(T1)
        print(f"recal data:")
        print(f"ct steel balls: {ct_steel_balls}")
        print(f"ct light balls: {ct_light_balls}")
        print(f"T0: {T0}")
        print("================")
        return T0, T1
        
    def cal_plan_ver_dis_angle(self):
        if not self._data_dict:
            return
        # ROBOT_DATA
        model_robot = self._data_dict.get("model_robot")
        rom_robot = self._data_dict.get("rom_robot")
        # SAVE_CAL_TAIL_X, TAIL_P
        save_tail_x = self._data_dict.get("tail_x")
        save_tail_p = self._data_dict.get("tail_p")
        print("save cal data: ")
        print(f"tail x: {save_tail_x}")
        print(f"tail p: {save_tail_p}")
        print("================")
        
        # 计算TAIL_X, TAIL_P
        T_model_to_rom, _ = rigidTransform3D(model_robot[:4].T, rom_robot.T)
        hole_pos = transform3d(model_robot[4:], T_model_to_rom)
        hole_dis = hole_pos[0] - hole_pos[1]  # 孔上，孔下
        tail_x = hole_dis / np.linalg.norm(hole_dis)
        tail_p = hole_pos[0]
        print("recal data:")
        print(f"tail x: {tail_x}")
        print(f"tail p: {tail_p}")
        print("================")
        
        # 扎针时相机下机器人和光球的位姿计算 -> CT里针尖和针尾的位置，机器人位置该扎的位置
        T_robot = self._data_dict.get("robot")
        T_spine = self._data_dict.get("spine")
        needle_length = self._data_dict.get("op_tool_length")
        T0, T1 = self.find_steel_balls_and_cal_t0()
        T_robot_to_ct = (T1 @ np.linalg.inv(T_spine) @ T_robot)
        ct_tail_x = T_robot_to_ct[:3, :3] @ np.array(tail_x)
        ct_tail_p = transform3d(np.array(tail_p).reshape(1, 3), T_robot_to_ct)
        ct_needle = ct_tail_p - ct_tail_x * int(needle_length)
        print("cal ct needle data:")
        print(f"ct tail pos: {ct_tail_p}")
        print(f"ct needle pos: {ct_needle}")
        print("================")
        
        # 规划的CT里针尖和针尾的位置
        ct_plan_pose = self._data_dict.get("ct_plan_pose")
        ct_plan_hole_dis = ct_plan_pose[1] - ct_plan_pose[0]
        ct_plan_tail_x = ct_plan_hole_dis / np.linalg.norm(ct_plan_hole_dis)
        print("plan ct needle data:")
        print(f"ct plan tail pos: {ct_plan_pose[1]}")
        print(f"ct plan needle pos: {ct_plan_pose[0]}")
        print("================")
        
        # 计算规划和实际要插的针的位置误差和角度误差
        dis_plan_veri_err = np.linalg.norm(ct_plan_pose[0] - ct_needle)
        angle_plan_veri_err = self._calculate_angle(ct_plan_tail_x, ct_tail_x)
        return dis_plan_veri_err, angle_plan_veri_err


if __name__ == "__main__":
    
    app = AppPlanVeri(npz_file="c:/Users/huang/CT-MP25075-250605-8_2025-06-27_10_38_17.npz")
    dis_plan_veri_err, angle_plan_veri_err = app.cal_plan_ver_dis_angle()
    print(f"dis plan veri err: {dis_plan_veri_err}")
    print(f"angle plan veri err: {angle_plan_veri_err}")
