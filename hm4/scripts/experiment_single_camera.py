#!/usr/bin/env python3
"""Experiment: Single-camera calibration with different configurations.

Tests different image subsets, distortion flags, and image counts to find
the best calibration setup that minimizes intrinsic error against ground truth.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotics_perception.calibration import (
    calibrate_single_camera,
    collect_calibration_points,
    list_images,
)
from robotics_perception.camera_model import compute_reprojection_error


def report(name, K, dist, obj_pts, img_pts, rvecs, tvecs, K_gt):
    """Print and return calibration metrics vs ground truth."""
    mean_err, per_view = compute_reprojection_error(
        obj_pts, img_pts, rvecs, tvecs, K, dist
    )
    fx_e = abs(K[0, 0] - K_gt[0, 0]) / K_gt[0, 0] * 100
    fy_e = abs(K[1, 1] - K_gt[1, 1]) / K_gt[1, 1] * 100
    cx_e = abs(K[0, 2] - K_gt[0, 2]) / K_gt[0, 2] * 100
    cy_e = abs(K[1, 2] - K_gt[1, 2]) / K_gt[1, 2] * 100
    avg_intrinsic_err = (fx_e + fy_e + cx_e + cy_e) / 4

    print(f"  {name}")
    print(f"    K: fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print(f"    Reproj: {mean_err:.4f} px | fx_err: {fx_e:.2f}% fy_err: {fy_e:.2f}% cx_err: {cx_e:.2f}% cy_err: {cy_e:.2f}%")
    print(f"    Dist: {dist.ravel()}")
    print(f"    Avg intrinsic err: {avg_intrinsic_err:.2f}%")

    return {
        "name": name, "reproj": mean_err,
        "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "fx_err": fx_e, "fy_err": fy_e, "cx_err": cx_e, "cy_err": cy_e,
        "avg_intrinsic_err": avg_intrinsic_err,
        "dist": dist.ravel().tolist(),
    }


def main():
    with open("sample_data/synthetic/ground_truth.json") as f:
        gt = json.load(f)
    K_gt = np.array(gt["K_left"])

    image_paths = list_images("sample_data/synthetic/single")
    checkerboard_size = (9, 6)
    square_size = 0.025

    obj_all, img_all, image_size, used_paths = collect_calibration_points(
        image_paths, checkerboard_size, square_size
    )
    print(f"Total valid images: {len(used_paths)}")

    obj_f32 = [p.astype(np.float32) for p in obj_all]
    img_f32 = [p.astype(np.float32) for p in img_all]
    all_results = []

    # ── [A] Baseline: all 12 images, default ──
    print("\n" + "=" * 60)
    cam, rvecs, tvecs, _, _ = calibrate_single_camera(obj_all, img_all, image_size)
    all_results.append(report("[A] All 12 images, default", cam.K, cam.dist, obj_all, img_all, rvecs, tvecs, K_gt))

    # ── [B] Remove outliers: skip calib_001 (idx=1) and calib_010 (idx=10) ──
    keep = [i for i in range(len(obj_all)) if i not in (1, 10)]
    obj_sub = [obj_all[i] for i in keep]
    img_sub = [img_all[i] for i in keep]
    cam, rvecs, tvecs, _, _ = calibrate_single_camera(obj_sub, img_sub, image_size)
    all_results.append(report("[B] 10 images (remove worst 2)", cam.K, cam.dist, obj_sub, img_sub, rvecs, tvecs, K_gt))

    # ── [C] Fix aspect ratio fx=fy ──
    flags_ar = cv2.CALIB_FIX_ASPECT_RATIO
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_f32, img_f32, image_size, None, None, flags=flags_ar)
    all_results.append(report("[C] All 12, fix aspect ratio (fx=fy)", K, dist, obj_all, img_all, rvecs, tvecs, K_gt))

    # ── [D] Zero tangential distortion ──
    flags_zt = cv2.CALIB_ZERO_TANGENT_DIST
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_f32, img_f32, image_size, None, None, flags=flags_zt)
    all_results.append(report("[D] All 12, zero tangential dist", K, dist, obj_all, img_all, rvecs, tvecs, K_gt))

    # ── [E] Fix aspect ratio + zero tangential ──
    flags_arzt = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_f32, img_f32, image_size, None, None, flags=flags_arzt)
    all_results.append(report("[E] All 12, fix AR + zero tangential", K, dist, obj_all, img_all, rvecs, tvecs, K_gt))

    # ── [F] Fix aspect ratio + zero tangential + remove outliers ──
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj_f32[i] for i in keep], [img_f32[i] for i in keep],
        image_size, None, None, flags=flags_arzt
    )
    all_results.append(report("[F] 10 images, fix AR + zero tangential", K, dist, obj_sub, img_sub, rvecs, tvecs, K_gt))

    # ── [G] Only 3 distortion params (k1, k2, p1) ──
    flags_3k = cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_f32, img_f32, image_size, None, None, flags=flags_3k)
    all_results.append(report("[G] All 12, 3-param dist (k1,k2,p1)", K, dist, obj_all, img_all, rvecs, tvecs, K_gt))

    # ── Generate more images and recalibrate ──
    extra_dir = "sample_data/synthetic_extra"
    subprocess.run([
        sys.executable, "scripts/generate_synthetic_data.py",
        "--output_dir", extra_dir, "--num_images", "30", "--seed", "42"
    ], check=True, capture_output=True)

    extra_paths = list_images(f"{extra_dir}/single")
    obj_extra, img_extra, _, used_extra = collect_calibration_points(extra_paths, checkerboard_size, square_size)
    print(f"\n  Generated {len(used_extra)} extra images")

    # [H] 30 images, default
    cam, rvecs, tvecs, _, _ = calibrate_single_camera(obj_extra, img_extra, image_size)
    all_results.append(report("[H] 30 fresh images, default", cam.K, cam.dist, obj_extra, img_extra, rvecs, tvecs, K_gt))

    # [I] 30 images + fix AR + zero tangential
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [p.astype(np.float32) for p in obj_extra],
        [p.astype(np.float32) for p in img_extra],
        image_size, None, None, flags=flags_arzt
    )
    all_results.append(report("[I] 30 fresh images, fix AR + zero tangential", K, dist, obj_extra, img_extra, rvecs, tvecs, K_gt))

    # [J] Combined 12+30 = 42 images, default
    obj_combined = obj_all + obj_extra
    img_combined = img_all + img_extra
    cam, rvecs, tvecs, _, _ = calibrate_single_camera(obj_combined, img_combined, image_size)
    all_results.append(report("[J] 42 images (12+30), default", cam.K, cam.dist, obj_combined, img_combined, rvecs, tvecs, K_gt))

    # [K] 42 images + fix AR + zero tangential
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [p.astype(np.float32) for p in obj_combined],
        [p.astype(np.float32) for p in img_combined],
        image_size, None, None, flags=flags_arzt
    )
    all_results.append(report("[K] 42 images (12+30), fix AR + zero tangential", K, dist, obj_combined, img_combined, rvecs, tvecs, K_gt))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by avg intrinsic error)")
    print("=" * 60)
    sorted_r = sorted(all_results, key=lambda x: x["avg_intrinsic_err"])
    print(f"{'Config':<45} | {'Reproj':>7} | {'fx%':>6} | {'fy%':>6} | {'Avg%':>6}")
    print("-" * 80)
    for r in sorted_r:
        print(f"{r['name']:<45} | {r['reproj']:>7.4f} | {r['fx_err']:>6.2f} | {r['fy_err']:>6.2f} | {r['avg_intrinsic_err']:>6.2f}")

    best = sorted_r[0]
    print(f"\nBEST: {best['name']}")
    print(f"  Reproj={best['reproj']:.4f} px, Avg intrinsic err={best['avg_intrinsic_err']:.2f}%")

    Path("outputs/single").mkdir(parents=True, exist_ok=True)
    with open("outputs/single/experiment_all_variants.json", "w") as f:
        json.dump(sorted_r, f, indent=2)
    print("\nResults saved to outputs/single/experiment_all_variants.json")

    shutil.rmtree(extra_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
