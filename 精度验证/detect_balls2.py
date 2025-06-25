import os
import pydicom
import numpy as np
import taichi as ti
from copy import deepcopy

ti.init(arch=ti.cpu, kernel_profiler=False, device_memory_GB=2)

B = 400000
N = 500
minvote = 10

head_b = ti.field(ti.i32, shape=())
head_x = ti.field(ti.i32, shape=())


xp = ti.field(dtype=ti.i32, shape=N)
yp = ti.field(dtype=ti.i32, shape=N)
zp = ti.field(dtype=ti.i32, shape=N)
Rp = ti.field(dtype=ti.i32, shape=N)
hu_p = ti.field(dtype=ti.i32, shape=N)

x1_ti = ti.field(ti.i32, shape=())
x2_ti = ti.field(ti.i32, shape=())
y1_ti = ti.field(ti.i32, shape=())
y2_ti = ti.field(ti.i32, shape=())
z1_ti = ti.field(ti.i32, shape=())
z2_ti = ti.field(ti.i32, shape=())


def voxel_to_mm(voxel_point, pixel_spacing, thickness, pos_ori):
    """
    将体素坐标转换为DICOM毫米坐标
    参数：
        voxel_point: 体素坐标(x, y, z)
        pixel_spacing: 像素间距 (x, y)方向 [mm/pixel]
        thickness: 层厚 [mm]
        pos_ori: DICOM原始坐标原点 [x, y, z] (mm)
    返回：
        mm_coords: 三维空间毫米坐标 (x, y, z)
    """
    # 注意这里轴系的对应关系：
    # voxel X -> 切片序列方向 (DICOM Z)
    # voxel Y -> 图像行方向
    # voxel Z -> 图像列方向
    return np.array([
        -pos_ori[0] - (voxel_point[2] * pixel_spacing[0]),  # X方向计算
        -(voxel_point[1] * pixel_spacing[1] + pos_ori[1]),  # Y方向计算
        voxel_point[0] * thickness + pos_ori[2]              # Z方向计算
    ])

def detect_spheres_from_dicom(data: np.ndarray, pixel_spacing: list, thickness: float, pos_ori:np.ndarray,
                              r_range: float, rlen: int, rmax: float, output_points: int):

    data = np.flip(data, axis=0)
    x, y, z = data.shape
    max_val = np.max(data)
    min_val = np.min(data)
    minval = round(max_val * 0.5)
    r_range_voxel = round(r_range / pixel_spacing[0])
    z_range = round(r_range / thickness)
    rlen_voxel = round(rlen / pixel_spacing[0])
    rmax_voxel = round(rmax / pixel_spacing[0])


    region_bounds = ((0, x), (0, y), (0, z))
    xrange, yrange, zrange_tuple = region_bounds
    x1_ti[None], x2_ti[None] = xrange
    y1_ti[None], y2_ti[None] = yrange
    z1_ti[None], z2_ti[None] = zrange_tuple

    img_data = ti.field(dtype=ti.i32, shape=(x, y, z))
    img_data.from_numpy(data)

    xvote = ti.field(dtype=ti.i32, shape=x)
    yvote = ti.field(dtype=ti.i32, shape=y)
    zvote = ti.field(dtype=ti.i32, shape=z)

    bon_x = ti.field(dtype=ti.i32, shape=B)
    bon_y = ti.field(dtype=ti.i32, shape=B)
    bon_z = ti.field(dtype=ti.i32, shape=B)

    z_scale = ti.field(dtype=ti.f32, shape=())  # 添加缩放比例
    z_scale[None] = thickness / pixel_spacing[0]

    @ti.kernel
    def vote_XYZ():
        for j, k in ti.ndrange((y1_ti[None], y2_ti[None]), (z1_ti[None], z2_ti[None])):
            tmp = 0
            for i in range(x1_ti[None], x2_ti[None] - 1):
                if (img_data[i, j, k] > minval) ^ (img_data[i + 1, j, k] > minval):
                    index = ti.atomic_add(head_b[None], 1)
                    # print(index)
                    bon_x[index] = i
                    bon_y[index] = j
                    bon_z[index] = k
                    if tmp == 0:
                        tmp = i
                    else:
                        cx = ti.ceil((tmp + i) / 2.0, int)
                        xvote[cx] += 1
                        tmp = i
        for i in range(B):
            if bon_x[i] != 0:
                for j in range(B):
                    if bon_x[i] == bon_x[j] and bon_z[i] == bon_z[j] and ti.abs(bon_y[j] - bon_y[i]) < 2:
                        cx = ti.ceil((bon_y[j] + bon_y[i]) / 2.0, int)
                        yvote[cx] += 1
                        # print(yvote[cx])

        for i in range(B):
            if bon_x[i] != 0:
                for j in range(B):
                    if bon_x[i] == bon_x[j] and bon_y[i] == bon_y[j] and ti.abs(bon_z[j] - bon_z[i]) < 2:
                        cx = ti.ceil((bon_z[j] + bon_z[i]) / 2.0, int)
                        zvote[cx] += 1
                        # print(zvote[cx])

    # 用于判断每个voxel是否可能是球心，粗劣判断是否在球体中心+去重
    @ti.kernel
    def localMax():
        for i in range(rmax_voxel, x - rmax_voxel):
            for j in range(rmax_voxel, y - rmax_voxel):
                for k in range(rmax_voxel, z - rmax_voxel):
                    # 投票值足够才考虑
                    if xvote[i] > minvote and yvote[j] > minvote and zvote[k] > minvote:
                        # 判断该点是否在所有方向上有亮点包围
                        valid = False
                        for r in range(r_range_voxel, rmax_voxel + 1):
                            if img_data[i - r, j, k] > minval and img_data[i + r, j, k] > minval and \
                                    img_data[i, j - r, k] > minval and img_data[i, j + r, k] > minval and \
                                    img_data[i, j, k - r] > minval and img_data[i, j, k + r] > minval:
                                valid = True
                                break

                        # 重复点去除（距离已有点小于一定范围则跳过）
                        if valid:
                            is_duplicate = False
                            for idx in range(head_x[None]):
                                dx = xp[idx] - i
                                dy = yp[idx] - j
                                dz = zp[idx] - k
                                if dx * dx + dy * dy + dz * dz < rlen_voxel * rlen_voxel:  # √9=3以内算重复
                                    is_duplicate = True
                            if not is_duplicate:
                                index = ti.atomic_add(head_x[None], 1)
                                xp[index] = i
                                yp[index] = j
                                zp[index] = k

    # 测量出球体大小
    @ti.kernel
    def vote_R():
        for i in range(head_x[None]):
            x0 = xp[i]
            y0 = yp[i]
            z0 = zp[i]
            max_r = 0
            max_brightness = 0.0
            for r in range(r_range_voxel, rmax_voxel + 1):
                rz = ti.cast(r / z_scale[None], ti.i32)  # z方向上的半径折算
                if x0 - r < 0 or x0 + r >= x or \
                        y0 - r < 0 or y0 + r >= y or \
                        z0 - rz < 0 or z0 + rz >= z:
                    continue
                brightness = (
                    img_data[x0 + r, y0, z0] + img_data[x0 - r, y0, z0] + img_data[x0, y0 + rz, z0] +
                    img_data[x0, y0 - r, z0] + img_data[x0, y0, z0 + r] + img_data[x0, y0, z0 - rz]
                ) / 6.0
                if img_data[x0 + r, y0, z0] > minval and img_data[x0 - r, y0, z0] > minval and \
                        img_data[x0, y0 + r, z0] > minval and img_data[x0, y0 - r, z0] > minval and \
                        img_data[x0, y0, z0 + rz] > minval and img_data[x0, y0, z0 - rz] > minval:
                    if brightness > max_brightness:
                        max_brightness = brightness
                        max_r = r
                if max_r > 0:
                    Rp[i] = r
                    hu_p[i] = int(max_brightness)
    vote_XYZ()
    localMax()
    vote_R()
    xp_np = xp.to_numpy()
    yp_np = yp.to_numpy()
    zp_np = zp.to_numpy()
    Rp_np = Rp.to_numpy()
    hu_p_np = hu_p.to_numpy()

    valid_mask = Rp_np > 0
    filtered_centers = (np.stack((xp_np[valid_mask], yp_np[valid_mask], zp_np[valid_mask], Rp_np[valid_mask],
                                  hu_p_np[valid_mask]), axis=-1).
                        astype(np.float32))
    sorted_centers = filtered_centers[filtered_centers[:, 4].argsort()[::-1]]
    # for i, (x, y, z, r, hu) in enumerate(sorted_centers):
    #     print(f"Center {i}: ({x}, {y}, {z}), r = {r}, brightness = {hu}")
    lps_to_ras = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    pos_ori = pos_ori @ lps_to_ras
    points = sorted_centers[:output_points]  # shape: (N, 3)
    print(points)

    # 要乘以IJK 到 RAS的毫米坐标
    points_copy = deepcopy(points)
    points[:, 0] = points_copy[:, 2] * pixel_spacing[0] * -1 + pos_ori[0]
    points[:, 1] = points_copy[:, 1] * pixel_spacing[1] + pos_ori[1]
    points[:, 2] = points_copy[:, 0] * thickness * -1 + pos_ori[2]

    return points

if __name__ == '__main__':
    data_path = os.path.join(os.path.expanduser("~"), "spine/dataStore/dicom_data")

    # 示例：只检测 z ∈ [100, 200]，x ∈ [150, 250]，y ∈ [100, 200]
    # ts = time.time()
    dicom_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    dicom_files.sort()
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    data = np.stack([s.pixel_array for s in slices])
    z, y, x = data.shape
    # todo: 需要IJK不同排序看翻转哪个轴和3D slicer体素对应
    pixel_spacing = slices[0].PixelSpacing
    if hasattr(slices[0], 'SpacingBetweenSlices'):
        thickness = slices[0].SpacingBetweenSlices
    else:
        thickness = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    print("体厚度=", thickness, pixel_spacing)

    # pos_ori = slices[0].ImagePositionPatient
    # print(slices[0].ImagePositionPatient)
    # print(f"原点={pos_ori}")
    # pos_ori_list = [
    #     [156.024, -55.668, -360.214],
    #     [156.024, -55.668, -634.713],
    #     [156.024, 209.557, -360.214],
    #     [156.024, 209.557, -634.713],
    #     [-107.201, -55.668, -633.713],
    #     [-107.201, -55.668, -360.214],
    #     [-107.201, 209.557, -634.713],
    #     [-107.201, 209.557, -360.214]
    # ]
    axis = slices[0].ImageOrientationPatient
    row_cosines = np.array(axis[0:3])
    col_cosines = np.array(axis[3:6])
    z_cosines = np.cross(row_cosines, col_cosines)



    lps_to_ras = np.array([[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]])
    pos_ori = slices[-1].ImagePositionPatient @ lps_to_ras
    print("pos ori", pos_ori)



    spheres = detect_spheres_from_dicom(data=data, pixel_spacing=pixel_spacing, thickness=thickness, pos_ori=pos_ori,
                                        r_range=1.5, rmax=5, rlen=4, output_points=15)

    print("Detected spheres:")
    print(spheres)

