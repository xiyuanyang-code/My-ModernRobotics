"""Camera model utilities for the robotics perception homework.

Students should complete the TODO sections. OpenCV may be used, but the report
must explain the underlying pinhole camera model and distortion model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class CameraParameters:
    """Container for single-camera calibration parameters.

    Attributes:
        K: Camera intrinsic matrix with shape (3, 3).
        dist: Distortion coefficients, usually shape (5,), (8,), or (1, N).
        image_size: Image size as (width, height).
    """

    K: np.ndarray
    dist: np.ndarray
    image_size: Tuple[int, int]

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])


def build_checkerboard_object_points(
    checkerboard_size: Tuple[int, int],
    square_size: float,
) -> np.ndarray:
    """Create 3D checkerboard corner coordinates in the board frame.

    Args:
        checkerboard_size: Number of inner corners as (cols, rows).
        square_size: Physical square size, usually in meters.

    Returns:
        object_points: Array with shape (cols * rows, 3). All z coordinates
            should be zero because the checkerboard is planar.

    Example:
        For checkerboard_size=(3, 2), square_size=0.1, the points should be:
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0],
         [0.0, 0.1, 0.0], [0.1, 0.1, 0.0], [0.2, 0.1, 0.0]]
    """
    cols, rows = checkerboard_size

    # Generate (x, y) grid indices for all inner corners.
    # np.mgrid[0:cols, 0:rows] produces two arrays of shape (cols, rows);
    # transpose to (rows, cols) then reshape to (cols*rows, 2).
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # shape (cols*rows, 2)
    # Scale by square_size and append z=0 column.
    object_points = np.hstack(
        [grid.astype(np.float64) * square_size,
         np.zeros((cols * rows, 1), dtype=np.float64)]
    )
    return object_points


def project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
) -> np.ndarray:
    """Project 3D points into the image using OpenCV's camera model.

    This helper is fully implemented so that students can use it in their
    reprojection-error code.
    """
    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points.astype(np.float64),
        rvec.astype(np.float64),
        tvec.astype(np.float64),
        K.astype(np.float64),
        dist.astype(np.float64),
    )
    return image_points.reshape(-1, 2)


def compute_reprojection_error(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[float, list[float]]:
    """Compute average reprojection error.

    Args:
        object_points_list: List of 3D point arrays, one per image.
        image_points_list: List of detected 2D point arrays, one per image.
        rvecs: Rotation vectors estimated for each image.
        tvecs: Translation vectors estimated for each image.
        K: Camera intrinsic matrix.
        dist: Distortion coefficients.

    Returns:
        mean_error: Mean reprojection error in pixels over all points.
        per_view_errors: One mean error per image.
    """
    per_view_errors: list[float] = []
    for obj_pts, img_pts, rvec, tvec in zip(
        object_points_list, image_points_list, rvecs, tvecs
    ):
        # Project 3D object points back into the image.
        projected = project_points(obj_pts, rvec, tvec, K, dist)
        # Euclidean pixel error per point, then mean for this view.
        errors = np.linalg.norm(projected - img_pts.reshape(-1, 2), axis=1)
        per_view_errors.append(float(np.mean(errors)))

    mean_error = float(np.mean(per_view_errors))
    return mean_error, per_view_errors


def undistort_image(image: np.ndarray, camera: CameraParameters) -> np.ndarray:
    """Undistort an image using calibrated camera intrinsics and distortion."""
    return cv2.undistort(image, camera.K, camera.dist)
