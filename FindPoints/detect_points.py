from copy import deepcopy
import taichi as ti
import numpy as np
import pydicom
import math
import os
    
    
def detect_spheres_from_dicom(dicom_path: str, num_spheres: int = 3, r_range: float = 2.5, rlen: int = 0,
                            rmax: int = 7):
    ti.init(arch=ti.cpu, kernel_profiler=False, device_memory_GB=2)
    
    B = 400000      # 可能得光球边界点
    minval = 3000 # 二值化强度阈值
    N = 20         # xyz投票数
    minvote = 30   # 局部最大投票值
    pNum = 300     # 候选球个数
    
    max_points_per_sphere = 200

    vote_points = ti.field(dtype=ti.i32, shape=(N, N, N, max_points_per_sphere,  3))
    vote_counts = ti.field(dtype=ti.i32, shape=(N, N, N))


    head_b = ti.field(ti.i32, shape=())
    headx = ti.field(ti.i32, shape=())
    heady = ti.field(ti.i32, shape=())
    headz = ti.field(ti.i32, shape=())
    headrP = ti.field(ti.i32, shape=())

    xp = ti.field(dtype=ti.i32, shape=N)
    yp = ti.field(dtype=ti.i32, shape=N)
    zp = ti.field(dtype=ti.i32, shape=N)
    
    # Load DICOM
    dicom_files = []
    for root, dirs, files in os.walk(dicom_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    dicom_files.sort()
    slices = list()
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'pixel_array'):  # 确保文件包含像素数据
                slices.append(ds)
            else:
                print(f"⚠️ 无像素数据: {f}")
                # print(ds.pixel_array.shape)
        except Exception as e:
            print(f"❌ 读取失败: {f}, 错误: {e}")
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    data = np.stack([s.pixel_array for s in slices])
    data = np.flip(data, axis=0)
    x, y, z = data.shape
    pixel_spacing = slices[0].PixelSpacing  # x,y轴间隔
    thickness = slices[0].SliceThickness    # z轴间隔

    z_range = r_range
    r_range = int(r_range / pixel_spacing[0])
    z_range = int(r_range / thickness)
    rlen = int(rlen / pixel_spacing[0])
    rmax = int(rmax / pixel_spacing[0])


    img_data = ti.field(dtype=ti.i32, shape=(x, y, z))
    img_data.from_numpy(data)

    xvote = ti.field(dtype=ti.i32, shape=x)
    yvote = ti.field(dtype=ti.i32, shape=y)
    zvote = ti.field(dtype=ti.i32, shape=z)

    bon_x = ti.field(dtype=ti.i32, shape=B)
    bon_y = ti.field(dtype=ti.i32, shape=B)
    bon_z = ti.field(dtype=ti.i32, shape=B)

    rvote = ti.Vector.field(n=rmax, dtype=ti.i32, shape=(N, N, N))
    rpoll = ti.Vector.field(n=5, dtype=ti.i32, shape=pNum)
    
    @ti.kernel
    def vote_XYZ():
        for j, k in ti.ndrange(y, z):
            tmp = 0
            for i in range(x - 1):
                if (img_data[i, j, k] > minval) ^ (img_data[i + 1, j, k] > minval):
                    index = ti.atomic_add(head_b[None], 1)
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
                    if bon_x[i] == bon_x[j] and bon_z[i] == bon_z[j] and ti.abs(bon_y[j] - bon_y[i]) < 10:
                        cx = ti.ceil((bon_y[j] + bon_y[i]) / 2.0, int)
                        yvote[cx] += 1

        for i in range(B):
            if bon_x[i] != 0:
                for j in range(B):
                    if bon_x[i] == bon_x[j] and bon_y[i] == bon_y[j] and ti.abs(bon_z[j] - bon_z[i]) < 10:
                        cx = ti.ceil((bon_z[j] + bon_z[i]) / 2.0, int)
                        zvote[cx] += 1
    
    # 找极值，在相邻的7个圆心里找出得票最大的一个                    
    @ti.kernel
    def localMax():
        for i in range(3, x - 3):
            if xvote[i] > minvote and xvote[i] == ti.max(xvote[i - 3], xvote[i - 2], xvote[i - 1], xvote[i], xvote[i + 1], xvote[i + 2], xvote[i + 3]):
                index = ti.atomic_add(headx[None], 1)
                xp[index] = i

        for i in range(3, y - 3):
            if yvote[i] > minvote and yvote[i] == ti.max(yvote[i - 3], yvote[i - 2], yvote[i - 1], yvote[i], yvote[i + 1], yvote[i + 2], yvote[i + 3]):
                index = ti.atomic_add(heady[None], 1)
                yp[index] = i

        for i in range(3, z - 3):
            if zvote[i] > minvote and zvote[i] == ti.max(zvote[i - 3], zvote[i - 2], zvote[i - 1], zvote[i], zvote[i + 1], zvote[i + 2], zvote[i + 3]):
                index = ti.atomic_add(headz[None], 1)
                zp[index] = i
    
    # 半径投票，投票给所有圆心及半径
    @ti.kernel
    def vote_R():
        for i in range(B):
            for o, p, q in ti.ndrange(N, N, N):
                if xp[o] > 0 and yp[p] > 0 and zp[q] > 0:
                    if ti.abs(bon_x[i] - xp[o]) <= r_range and ti.abs(bon_y[i] - yp[p]) <= r_range and ti.abs(bon_z[i] - zp[q]) <= z_range:
                        tmp = ti.ceil(((bon_x[i] - xp[o]) ** 2 + (bon_y[i] - yp[p]) ** 2 + (bon_z[i] - zp[q]) ** 2) ** 0.5, int)
                        rvote[o, p, q][tmp] += 1
                        is_duplicate = False
                        for k in range(vote_counts[o, p, q]):
                            if (vote_points[o, p, q, k, 0] == bon_x[i] and
                                    vote_points[o, p, q, k, 1] == bon_y[i] and
                                    vote_points[o, p, q, k, 2] == bon_z[i]):
                                is_duplicate = True
                                break

                        # 如果未存储过，则新增
                        if not is_duplicate:
                            point_count = ti.atomic_add(vote_counts[o, p, q], 1)
                            if point_count < 1000:
                                vote_points[o, p, q, point_count, 0] = bon_x[i]
                                vote_points[o, p, q, point_count, 1] = bon_y[i]
                                vote_points[o, p, q, point_count, 2] = bon_z[i]
                                
    # vote_counts[o,p,q] ：表示 有多少个边界点投票给这个球心（即 (o,p,q) 位置的球心）。
    # rvote[i,j,k][r]：表示 有多少个边界点投票给这个球心 + 半径 r。
    @ti.kernel
    def rpolling():
        for i, j, k in ti.ndrange(N, N, N):
            tmp = 0
            for t in range(1, rmax):
                if rvote[i, j, k][t] >= rvote[i, j, k][tmp] and rvote[i, j, k][t] > 20:
                    tmp = t
            if tmp != 0:
                index = ti.atomic_add(headrP[None], 1)
                rpoll[index][0] = xp[i]
                rpoll[index][1] = yp[j]
                rpoll[index][2] = zp[k]
                rpoll[index][3] = tmp
                rpoll[index][4] = rvote[i, j, k][tmp]

    @ti.kernel
    def cirlocMax():
        for i, j in ti.ndrange(pNum, pNum):
            if ti.abs(rpoll[i][0] - rpoll[j][0]) < rlen and ti.abs(rpoll[i][1] - rpoll[j][1]) < rlen and ti.abs(rpoll[i][2] - rpoll[j][2]) < rlen:
                if rpoll[i][4] < rpoll[j][4]:
                    rpoll[i][4] = 0
                    
    vote_XYZ()
    localMax()
    vote_R()
    rpolling()
    cirlocMax()
    
    lps_to_ras = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    pos_ori = slices[-1].ImagePositionPatient
    pos_ori = pos_ori @ lps_to_ras
    
    points = np.empty((num_spheres, 5), dtype=np.float32)
    for i in range(num_spheres):
        tmp = 0
        for j in range(pNum):
            if rpoll[j][4] > rpoll[tmp][4]:
                tmp = j
        points[i] = [rpoll[tmp][k] for k in range(5)]
        rpoll[tmp][4] = 0
        
    # 寻找投票圆心 + 半径的边界点
    center_to_boundaries = {}
    for i in range(num_spheres):
        x, y, z, r, votes = points[i]
        print(f"x, y, z vote={votes}")
        # 找到 (x,y,z) 对应的 (i,j,k) 索引
        i = np.where(xp.to_numpy() == x)[0][0]
        j = np.where(yp.to_numpy() == y)[0][0]
        k = np.where(zp.to_numpy() == z)[0][0]
        radius_r_points = []
        for idx in range(vote_counts[i, j, k]):
            bx = vote_points[i, j, k, idx, 0]
            by = vote_points[i, j, k, idx, 1]
            bz = vote_points[i, j, k, idx, 2]
            distance = math.sqrt((bx - x) ** 2 + (by - y) ** 2 + (bz - z) ** 2)
            if abs(distance - r) < 1.5:
                radius_r_points.append((bx, by, bz))
        center_to_boundaries[(x, y, z)] = radius_r_points
        print(f"detect vote = {len(radius_r_points)}")
        
        
    points_copy = deepcopy(points)
    points[:, 0] = points_copy[:, 2] * pixel_spacing[0] * -1 + pos_ori[0]
    points[:, 1] = points_copy[:, 1] * pixel_spacing[1] + pos_ori[1]
    points[:, 2] = points_copy[:, 0] * thickness * -1 + pos_ori[2]
    return points, center_to_boundaries
    
    
    
if __name__ == '__main__':
    spheres, center_bound = detect_spheres_from_dicom(dicom_path="C:/Users/YangLiangZhu/Desktop/泰州CT模型/0605脊柱实验数据/dicom_data_bad_01", num_spheres=10)
    print("Detected spheres:")
    print(spheres)
    # print(center_bound)
