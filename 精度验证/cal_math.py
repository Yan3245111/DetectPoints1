import itertools
import numpy as np
from pyquaternion import Quaternion

def match_balls_by_geometry(actual_points, detected_points, verbose=False):
    def pairwise_distances(pts):
        return np.linalg.norm(pts[:, None] - pts[None, :], axis=2)

    def extract_dist_features(dist_matrix):
        return dist_matrix[np.triu_indices(4, k=1)]  # 提取上三角的6个值

    min_error = float('inf')
    best_match = None
    best_ordered_pts = None

    num_detected = detected_points.shape[0]
    if num_detected < 4:
        raise ValueError("至少需要4个检测点进行匹配。")

    for combo in itertools.combinations(range(num_detected), 4):
        candidate_points = detected_points[list(combo)]
        for perm in itertools.permutations(range(4)):
            ordered_candidate = candidate_points[list(perm)]

            actual_dists = extract_dist_features(pairwise_distances(actual_points))
            candidate_dists = extract_dist_features(pairwise_distances(ordered_candidate))

            error = np.linalg.norm(actual_dists - candidate_dists)

            if error < min_error:
                min_error = error
                best_match = [combo[i] for i in perm]
                best_ordered_pts = ordered_candidate

    if verbose:
        print("best match：", best_match)
        for i, pt in enumerate(best_ordered_pts):
            print(f"act steel {i + 1} detect {pt}")
        print("all dis：", min_error)

    return best_ordered_pts, best_match, min_error

def rigidTransform3D(A, B):
    assert A.shape == B.shape

    if A.shape[0] != 3:
        raise Exception(f"matrix A is not 3xN, it is {A.shape[0]}x{A.shape[1]}")
    if B.shape[0] != 3:
        raise Exception(f"matrix B is not 3xN, it is {B.shape[0]}x{B.shape[1]}")

    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    # print('H', Am.shape, H.shape)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3,3] = t.ravel()
    
    err = (pose[:3,:3]@A + pose[:3,3:]) - B
    
    return pose, err


def transform3d(pts, pose):
    return (pose[:3,:3]@pts.T+pose[:3,3:]).T


def Q2M(orient, t):
    pos=np.eye(4)
    mat = Quaternion(orient).rotation_matrix  # w x y z
    pos[0:3, 0:3] = mat
    pos[0:3, 3] = np.array(t)
    return pos
    
    
def calculate_angle(v1, v2, degrees=True):
    """计算两个三维向量的夹角，返回角度或弧度"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    
    return np.degrees(theta) if degrees else theta


if __name__ == "__main__":
    needle = np.array([
        140.6393585205078,
        -131.3714141845703,
        -2077.384033203125,
        0.9778265953063965,
        0.10426448285579681,
        0.18136756122112274,
        -0.00948282890021801
    ])
    pos = Q2M(orient=needle[3:], t=needle[0:3])
    print(pos)
