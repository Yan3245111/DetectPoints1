import os
import pydicom
import numpy as np

from cal_math import match_balls_by_geometry, rigidTransform3D, transform3d, Q2M
from detect_balls2 import detect_spheres_from_dicom
from model_rom_data import plan_ball, plan_body, plan_robot, rom_body, rom_robot, pose_to_ct, spine, robot


npz_save_data = dict()

def load_dicom(dicom_path: str):
    global npz_save_data
    npz_save_data = dict()
    """ 加载dicom数据
    Args:
        dicom_path (str): dicom文件夹路径
    Returns:
        tuple[np.ndarray, list, float, np.ndarray]: dicom数据，像素间距，层厚，原始坐标原点
    """
    dicom_files = []
    for root, _, files in os.walk(dicom_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    dicom_files.sort()
    
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    patient_id = str(slices[0].PatientID)  # 存文件名的时候使用
    pixel_spacing = slices[0].PixelSpacing
    if hasattr(slices[0], 'SpacingBetweenSlices'):
        thickness = slices[0].SpacingBetweenSlices
    else:
        thickness = abs(slices[1].ImagePositionPatient[2] - abs(slices[0].ImagePositionPatient[2]))
    pos_ori = slices[-1].ImagePositionPatient
    
    print("====== pos_ori: ", pos_ori, slices[1].ImagePositionPatient)
    print("====== pixel_spacing: ", pixel_spacing)
    print("====== thickness: ", thickness)
    # 存储数据
    npz_save_data["ct_volume"] = np.stack([s.pixel_array for s in slices])
    npz_save_data["ct_pixel_spacing"] = np.array(pixel_spacing)
    npz_save_data["ct_thickness"] = np.array(thickness)
    npz_save_data["ct_pos_ori"] = np.array(slices[-1].ImagePositionPatient)  # 存储-1的原点
    
    # 钢球检测
    balls_ct_pos = detect_spheres_from_dicom(data=np.stack([s.pixel_array for s in slices]), pixel_spacing=pixel_spacing, thickness=thickness, 
                                        pos_ori=slices[-1].ImagePositionPatient, r_range=0.75, rlen=5, rmax=5, output_points=10)
    print("================== ball count : ", balls_ct_pos)
    # 根据实际模型匹配出正确结果 
    matched_points, _, _ = match_balls_by_geometry(plan_ball, balls_ct_pos[:, :3],
                                                                    verbose=True)
    balls_ct_pos_t = matched_points[:, :3]

    ct_plan_ball_mat, _ = rigidTransform3D(plan_ball.T, balls_ct_pos_t.T)
    print("模型到CT坐标的刚性变换矩阵", ct_plan_ball_mat)

    ct_plan_ball_body = transform3d(plan_body, ct_plan_ball_mat)
    print("ct下光球位置", ct_plan_ball_body)

    T1, _ = rigidTransform3D(rom_body.T, ct_plan_ball_body.T)

    print("rom到CT的刚性变换矩阵", T1)
    T0 = np.linalg.inv(T1)
    
    # 存储数据
    npz_save_data["cal_ct_steel_balls"] = balls_ct_pos_t
    npz_save_data["cal_ct_light_balls"] = ct_plan_ball_body
    npz_save_data["cal_ct2ball_mat"] = ct_plan_ball_mat
    
    npz_save_data["model_steel"] = np.array(plan_ball)
    npz_save_data["model_light"] = np.array(plan_body)
    npz_save_data["rom_light"] = np.array(rom_body)
    npz_save_data["model_robot"] = np.array(plan_robot)
    npz_save_data["rom_robot"] = np.array(rom_robot)
    npz_save_data["op_tool_length"] = np.array(140)
    npz_save_data["ct_plan_pose"] = np.array(pose_to_ct)
    npz_save_data["spine"] = np.array(spine)  # 已经转换好的4 x 4 Q2M以后的
    npz_save_data["robot"] = np.array(robot)  # 已经转换好的4 x 4 Q2M以后的
    np.savez(f'{patient_id}.npz', **npz_save_data)
    print("save npz file succeed")
    

load_dicom(dicom_path="C:/Users/YangLiangZhu/Desktop/泰州精度验证/Tai0606/9/dicom_data9_gui")