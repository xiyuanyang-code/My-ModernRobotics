"""Stereo calibration, rectification, disparity, and depth utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from robotics_perception.camera_model import CameraParameters, build_checkerboard_object_points
from robotics_perception.calibration import find_checkerboard_corners, list_images


@dataclass
class StereoParameters:
    """Container for stereo calibration results."""

    left: CameraParameters
    right: CameraParameters
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    image_size: Tuple[int, int]

    @property
    def baseline(self) -> float:
        """Stereo baseline length in the same unit as checkerboard square size."""
        return float(np.linalg.norm(self.T.reshape(3)))


def collect_stereo_points(
    left_dir: str | Path,
    right_dir: str | Path,
    checkerboard_size: Tuple[int, int],
    square_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int], list[tuple[Path, Path]]]:
    """Collect matched checkerboard corners from synchronized stereo images."""
    left_paths = list_images(left_dir)
    right_paths = list_images(right_dir)

    if len(left_paths) != len(right_paths):
        print(f"[WARN] Number of left/right images differs: {len(left_paths)} vs {len(right_paths)}")

    object_template = build_checkerboard_object_points(checkerboard_size, square_size)
    object_points_list: list[np.ndarray] = []
    left_points_list: list[np.ndarray] = []
    right_points_list: list[np.ndarray] = []
    used_pairs: list[tuple[Path, Path]] = []
    image_size: tuple[int, int] | None = None

    for left_path, right_path in zip(left_paths, right_paths):
        left_img = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            print(f"[WARN] Could not read pair: {left_path}, {right_path}")
            continue

        h, w = left_img.shape[:2]
        if image_size is None:
            image_size = (w, h)

        ok_l, corners_l = find_checkerboard_corners(left_img, checkerboard_size)
        ok_r, corners_r = find_checkerboard_corners(right_img, checkerboard_size)

        if ok_l and ok_r and corners_l is not None and corners_r is not None:
            object_points_list.append(object_template.copy())
            left_points_list.append(corners_l.astype(np.float32))
            right_points_list.append(corners_r.astype(np.float32))
            used_pairs.append((left_path, right_path))
        else:
            print(f"[WARN] Checkerboard not found in pair: {left_path.name}, {right_path.name}")

    if image_size is None:
        raise RuntimeError("No readable stereo images found.")

    return object_points_list, left_points_list, right_points_list, image_size, used_pairs


def calibrate_stereo_camera(
    object_points_list: list[np.ndarray],
    left_points_list: list[np.ndarray],
    right_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> StereoParameters:
    """Estimate left/right camera intrinsics and stereo extrinsics.

    A common workflow is:
      1. Calibrate left camera.
      2. Calibrate right camera.
      3. Run cv2.stereoCalibrate with fixed intrinsics.
    """
    # Ensure float32 for OpenCV compatibility.
    obj_f32 = [pts.astype(np.float32) for pts in object_points_list]
    left_f32 = [pts.astype(np.float32) for pts in left_points_list]
    right_f32 = [pts.astype(np.float32) for pts in right_points_list]

    # 1. Calibrate left and right cameras independently.
    _, K_left, dist_left, _, _ = cv2.calibrateCamera(
        obj_f32, left_f32, image_size, None, None
    )
    _, K_right, dist_right, _, _ = cv2.calibrateCamera(
        obj_f32, right_f32, image_size, None, None
    )

    # 2. Stereo calibration with fixed intrinsics.
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        obj_f32,
        left_f32,
        right_f32,
        K_left,
        dist_left,
        K_right,
        dist_right,
        image_size,
        flags=flags,
    )

    left_cam = CameraParameters(K=K_left, dist=dist_left, image_size=image_size)
    right_cam = CameraParameters(K=K_right, dist=dist_right, image_size=image_size)

    return StereoParameters(
        left=left_cam,
        right=right_cam,
        R=R,
        T=T,
        E=E,
        F=F,
        image_size=image_size,
    )


def stereo_rectify(
    stereo: StereoParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification transforms.

    Suggested return format:
        R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right
    """
    # Compute rectification rotation matrices and projection matrices.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        stereo.left.K, stereo.left.dist,
        stereo.right.K, stereo.right.dist,
        stereo.image_size,
        stereo.R, stereo.T,
        alpha=0,  # crop to valid pixels
    )

    # Compute rectification maps for both cameras.
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        stereo.left.K, stereo.left.dist, R1, P1, stereo.image_size, cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        stereo.right.K, stereo.right.dist, R2, P2, stereo.image_size, cv2.CV_32FC1
    )

    return R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right


def rectify_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    map1_left: np.ndarray,
    map2_left: np.ndarray,
    map1_right: np.ndarray,
    map2_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply precomputed rectification maps to a stereo pair."""
    rect_l = cv2.remap(left_img, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    rect_r = cv2.remap(right_img, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)
    return rect_l, rect_r


def compute_disparity_sgbm(
    rectified_left: np.ndarray,
    rectified_right: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 5,
) -> np.ndarray:
    """Compute disparity from rectified stereo images.

    Returns:
        disparity: Floating-point disparity map in pixels.
    """
    # Convert to grayscale if needed.
    if rectified_left.ndim == 3:
        gray_l = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    else:
        gray_l = rectified_left
    if rectified_right.ndim == 3:
        gray_r = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_r = rectified_right

    # Ensure num_disparities is divisible by 16.
    num_disparities = max(16, (num_disparities // 16) * 16)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # OpenCV returns fixed-point disparity scaled by 16.
    disparity_fixed = stereo.compute(gray_l, gray_r)
    disparity = disparity_fixed.astype(np.float32) / 16.0

    return disparity


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    """Convert disparity to depth using Z = fx * B / d.

    Invalid or non-positive disparity should be assigned np.nan or zero.
    """
    depth = np.full_like(disparity, np.nan, dtype=np.float64)
    valid = disparity > 0
    depth[valid] = (fx * baseline) / disparity[valid]
    return depth
