import numpy as np
from pyquaternion import Quaternion
from PyQt5.QtWidgets import QApplication


def Q2M(orient, t):
    pos = np.eye(4)
    mat = Quaternion(orient).rotation_matrix
    pos[0:3, 0:3] = mat
    pos[0:3, 3] = np.array(t)
    return pos


def transform3d(pts, pose):
    return (pose[:3, :3] @ pts.T + pose[:3, 3:]).T


# CT_pose
ct_plan_pose = np.array(
    [
        [-20.224718918268803, 105.07568677145187, -248.67599487304688],
        [-103.82022378044645, -45.935541307755, -248.67599487304685],
    ]
)

T0 = np.array(
    [
        64.1534857717078,
        414.824907026382,
        -43.14880150067543,
        -0.45983618151597266,
        0.5911549662238985,
        0.424507246417676,
        0.5088026039796899,
    ]
)

robot = np.array(
    [
        445.4141845703125,
        -13.681041717529297,
        -2017.0556640625,
        0.8675985336303711,
        -0.334163635969162,
        0.33674076199531555,
        0.14904040098190308,
    ]
)

spine = np.array(
    [
        471.61273193359375,
        -169.43695068359375,
        -1804.6583251953125,
        0.9891894459724426,
        0.020467620342969894,
        0.14518778026103973,
        -0.0024198840837925673,
    ]
)

T_ct_2_spine = Q2M(orient=T0[3:], t=T0[:3])
T_robot_2_ndi = Q2M(orient=robot[3:], t=T0[:3])
T_spine_2_ndi = Q2M(orient=spine[3:], t=spine[:3])

T_ct_2_ndi = T_spine_2_ndi @ T_ct_2_spine
ndi_plan_pos = transform3d(ct_plan_pose, T_ct_2_ndi)
print(np.linalg.norm((ndi_plan_pos[0] - ndi_plan_pos[1])))

spine_plan_pos = transform3d(ct_plan_pose, T_ct_2_spine)
print(spine_plan_pos)
