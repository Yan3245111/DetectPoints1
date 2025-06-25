from copy import deepcopy
import taichi as ti
import numpy as np
import pydicom
import os

ti.init(arch=ti.cpu, kernel_profiler=False, device_memory_GB=2)

B = 400000      # 可能得光球边界点
minval = 3000 # 二值化强度阈值
N = 40         # xyz投票数
minvote = 10   # 局部最大投票值
pNum = 300     # 候选球个数


head_b = ti.field(ti.i32, shape=())
headx = ti.field(ti.i32, shape=())
heady = ti.field(ti.i32, shape=())
headz = ti.field(ti.i32, shape=())
headrP = ti.field(ti.i32, shape=())

xp = ti.field(dtype=ti.i32, shape=N)
yp = ti.field(dtype=ti.i32, shape=N)
zp = ti.field(dtype=ti.i32, shape=N)

vote_points = ti.field(dtype=ti.i32, shape=(N, N, N, pNum,  3))
vote_counts = ti.field(dtype=ti.i32, shape=(N, N, N))

"""
DataLlj-获取至少3个光球的位置
2-根据实际模型计算第四个光球的位置 都是相对于CT的坐标
3-然后要对应1234哪个球是哪个球，定义一个原点光球，拿到相对位置xyz，还要计算其旋转矩阵
4-将spine-nav/spine-service/calibration.py 里的t0进行替换即可
"""
def detect_spheres_from_dicom(data, pixel_spacing, thickness, pos_ori, num_spheres: int = 3, r_range: float = 2.5, rlen: int = 5,
                              rmax: int = 4):


    data = np.flip(data, axis=0)  # z轴反转
    x, y, z = data.shape

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

    # 边界投票：把所有的边界点找出来，并使用（上一次的边界+第二次的边界）/2计算出圆心投票+1
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
                        # 半径过滤
                        if abs(tmp - i) < rmax * 2:
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

    # 寻找局部最大值，并记录下来得票最大的值
    @ti.kernel
    def localMax():
        for i in range(2, x - 2):
            if xvote[i] > minvote and xvote[i] == ti.max(xvote[i - 2], xvote[i - 1], xvote[i], xvote[i + 1], xvote[i + 2]):
                index = ti.atomic_add(headx[None], 1)
                xp[index] = i

        for i in range(2, y - 2):
            if yvote[i] > minvote and yvote[i] == ti.max(yvote[i - 2], yvote[i - 1], yvote[i], yvote[i + 1], yvote[i + 2]):
                index = ti.atomic_add(heady[None], 1)
                yp[index] = i

        for i in range(2, z - 2):
            if zvote[i] > minvote and zvote[i] == ti.max(zvote[i - 2], zvote[i - 1], zvote[i], zvote[i + 1], zvote[i + 2]):
                index = ti.atomic_add(headz[None], 1)
                zp[index] = i

    # @ti.kernel
    # def localMax():
    #     for i in range(3, x - 3):
    #         if xvote[i] > minvote and xvote[i] == ti.max(xvote[i - 3], xvote[i - 2], xvote[i - 1], xvote[i], xvote[i + 1], xvote[i + 2], xvote[i + 3]):
    #             index = ti.atomic_add(headx[None], 1)
    #             xp[index] = i
    #
    #     for i in range(3, y - 3):
    #         if yvote[i] > minvote and yvote[i] == ti.max(yvote[i - 3], yvote[i - 2], yvote[i - 1], yvote[i], yvote[i + 1], yvote[i + 2], yvote[i + 3]):
    #             index = ti.atomic_add(heady[None], 1)
    #             yp[index] = i
    #
    #     for i in range(3, z - 3):
    #         if zvote[i] > minvote and zvote[i] == ti.max(zvote[i - 3], zvote[i - 2], zvote[i - 1], zvote[i], zvote[i + 1], zvote[i + 2], zvote[i + 3]):
    #             index = ti.atomic_add(headz[None], 1)
    #             zp[index] = i

    @ti.kernel
    def vote_R():
        for i in range(B):
            for o, p, q in ti.ndrange(N, N, N):
                if xp[o] > 0 and yp[p] > 0 and zp[q] > 0:
                    if ti.abs(bon_x[i] - xp[o]) <= r_range and ti.abs(bon_y[i] - yp[p]) <= r_range and ti.abs(bon_z[i] - zp[q]) <= z_range:
                        tmp = ti.ceil(((bon_x[i] - xp[o]) ** 2 + (bon_y[i] - yp[p]) ** 2 + (bon_z[i] - zp[q]) ** 2) ** 0.5, int)
                        rvote[o, p, q][tmp] += 1
                        # 把边界记录下来
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
                            if point_count < pNum:
                                vote_points[o, p, q, point_count, 0] = bon_x[i]
                                vote_points[o, p, q, point_count, 1] = bon_y[i]
                                vote_points[o, p, q, point_count, 2] = bon_z[i]



    # 根据体素点把没用的过滤掉
    def count_points_in_space(center, boundary_points):
        cx, cy, cz = center
        counts = [0] * 8
        for x, y, z in boundary_points:
            region = 0
            if x > cx: region += 1
            if y > cy: region += 2
            if z > cz: region += 4
            counts[region] += 1
        return counts

    def filter_spheres():
        """过滤不符合条件的球心（8个象限至少各有一个点）"""
        valid_spheres = []
        for i, j, k in ti.ndrange(N, N, N):
            if xp[i] > 0 and yp[j] > 0 and zp[k] > 0:
                # 获取该候选球心的所有边界点
                boundary_points = []
                for idx in range(vote_counts[i, j, k]):
                    x = vote_points[i, j, k, idx, 0]
                    y = vote_points[i, j, k, idx, 1]
                    z = vote_points[i, j, k, idx, 2]
                    boundary_points.append((x, y, z))

                if len(boundary_points) == 0:
                    continue

                # 检查8个象限是否都有点
                center = (xp[i], yp[j], zp[k])
                counts = count_points_in_space(center, boundary_points)

                # 如果所有象限至少有一个点，则保留
                if all(c > 0 for c in counts):
                    valid_spheres.append((i, j, k))
        return valid_spheres

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


    def lps_to_ras_matrix(lps_coords):
        """
        使用变换矩阵进行转换（适合批量处理）
        """
        transform = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        return np.dot(lps_coords, transform.T)

    # Run pipeline
    vote_XYZ()
    localMax()
    # print(xp.to_numpy())
    # print(yp.to_numpy())
    # print(zp.to_numpy())
    vote_R()
    # 新增：过滤球心，返回的是球心在投票空间的索引位置
    # valid_spheres_np = np.array(filter_spheres() , dtype=np.int32)
    # if valid_spheres_np.shape[0] > 0:
    rpolling()
    cirlocMax()

    # Collect top spheres
    points = np.empty((num_spheres, 5), dtype=np.float32)
    for i in range(num_spheres):
        tmp = 0
        for j in range(pNum):
            if rpoll[j][4] > rpoll[tmp][4]:
                tmp = j
        points[i] = [rpoll[tmp][k] for k in range(5)]
        rpoll[tmp][4] = 0
    lps_to_ras = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    pos_ori = pos_ori @ lps_to_ras
    for i in range(num_spheres):
        z_index = points[i][0]
        y_index = points[i][1]
        x_index = points[i][2]
        
        x_ras = x_index * pixel_spacing[0] * -1 + pos_ori[0]
        y_ras = y_index * pixel_spacing[1] + pos_ori[1]
        z_ras = z_index * thickness * -1 + pos_ori[2]
        points[i] = [x_ras, y_ras, z_ras, points[i][3], points[i][4]]
    return  np.array([list(p) for p in points if p[4] > 0])



if __name__ == '__main__':

    path1 = "C:/Users/YangLiangZhu/Desktop/泰州精度验证/Tai0605/1/CT-MP25075-250605-yan/S3010"
    path2 = "C:/Users/YangLiangZhu/Desktop/泰州精度验证/Tai0605/8/CT-MP25075-250605-yan/S10010"
        # Load DICOM
    dicom_files = []
    for root, dirs, files in os.walk(path2):
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
    
    pixel_spacing = slices[0].PixelSpacing  # x,y轴间隔
    if hasattr(slices[0], 'SpacingBetweenSlices'):
        thickness = slices[0].SpacingBetweenSlices
    else:
        thickness = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])

    pos_ori = slices[-1].ImagePositionPatient  # 原点
    print(pos_ori)
    ImageOrientationPatient = slices[0].ImageOrientationPatient  # 方向向量

    spheres = detect_spheres_from_dicom(data=data, pixel_spacing=pixel_spacing, thickness=thickness, pos_ori=pos_ori, 
                                        num_spheres=10)
    print("Detected spheres:")
    print(spheres)
