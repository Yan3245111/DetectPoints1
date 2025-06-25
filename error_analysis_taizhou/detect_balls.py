
import numpy as np
import taichi as ti
import itertools
import glob
import os
ti.init(arch=ti.cpu, kernel_profiler=False, device_memory_GB=2)




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
                if x0 - r < 0 or x0 + r >= x or \
                        y0 - r < 0 or y0 + r >= y or \
                        z0 - r < 0 or z0 + r >= z:
                    continue
                brightness = (
                    img_data[x0 + r, y0, z0] + img_data[x0 - r, y0, z0] + img_data[x0, y0 + r, z0] +
                    img_data[x0, y0 - r, z0] + img_data[x0, y0, z0 + r] + img_data[x0, y0, z0 - r]
                ) / 6.0
                if img_data[x0 + r, y0, z0] > minval and img_data[x0 - r, y0, z0] > minval and \
                        img_data[x0, y0 + r, z0] > minval and img_data[x0, y0 - r, z0] > minval and \
                        img_data[x0, y0, z0 + r] > minval and img_data[x0, y0, z0 - r] > minval:
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
    filtered_centers = (np.stack((zp_np[valid_mask], yp_np[valid_mask], xp_np[valid_mask], Rp_np[valid_mask],
                                  hu_p_np[valid_mask]), axis=-1).
                        astype(np.float32))
    sorted_centers = filtered_centers[filtered_centers[:, 4].argsort()[::-1]]
    # for i, (x, y, z, r, hu) in enumerate(sorted_centers):
    #     print(f"Center {i}: ({x}, {y}, {z}), r = {r}, brightness = {hu}")
    points = sorted_centers[:output_points]  # shape: (N, 3)
    # print("tisu:",points)

    # points_copy = deepcopy(points)
    # # points[:, 0] = (points_copy[:, 2] * pixel_spacing[0]) + pos_ori[0]
    # # points[:, 1] = points_copy[:, 1] * pixel_spacing[1] + pos_ori[1]
    # # # points[:, 1] = points_copy[:, 1] * pixel_spacing[1] - pos_ori[1]
    # # points[:, 2] = points_copy[:, 0] * thickness + pos_ori[2]
    # points[:, 0] = pos_ori[0]+points_copy[:, 2] * pixel_spacing[0]*
    # points[:, 1] = pos_ori[1]
    # points[:, 2] = pos_ori[2]

    return points
