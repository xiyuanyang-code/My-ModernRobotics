#!/usr/bin/env python3
"""Experiment: Stereo calibration with different configurations.

Tests different calibration flags, outlier removal, and image counts
to find the best stereo setup that minimizes intrinsic and baseline error.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotics_perception.calibration import collect_calibration_points, list_images
from robotics_perception.stereo_calibration import collect_stereo_points


def compute_reproj_err(K, dist, obj_pts_list, img_pts_list):
    """Compute mean reprojection error via solvePnP per view."""
    errors = []
    for obj_pts, img_pts in zip(obj_pts_list, img_pts_list):
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts.astype(np.float64), img_pts.astype(np.float64),
            K.astype(np.float64), dist.astype(np.float64),
            flags=cv2.SOLVEPNP_IPPE
        )
        if ok:
            proj, _ = cv2.projectPoints(
                obj_pts.astype(np.float64), rvec, tvec,
                K.astype(np.float64), dist.astype(np.float64)
            )
            proj = proj.reshape(-1, 2)
            err = np.mean(np.linalg.norm(proj - img_pts.reshape(-1, 2), axis=1))
            errors.append(err)
    return float(np.mean(errors)) if errors else 999.0


def stereo_report(name, K_l, dist_l, K_r, dist_r, R, T, E, F,
                  obj_pts, l_pts, r_pts, K_gt_l, K_gt_r, baseline_gt):
    """Print and return stereo calibration metrics vs ground truth."""
    reproj_l = compute_reproj_err(K_l, dist_l, obj_pts, l_pts)
    reproj_r = compute_reproj_err(K_r, dist_r, obj_pts, r_pts)

    baseline = float(np.linalg.norm(T.reshape(3)))
    fx_err_l = abs(K_l[0, 0] - K_gt_l[0, 0]) / K_gt_l[0, 0] * 100
    fy_err_l = abs(K_l[1, 1] - K_gt_l[1, 1]) / K_gt_l[1, 1] * 100
    fx_err_r = abs(K_r[0, 0] - K_gt_r[0, 0]) / K_gt_r[0, 0] * 100
    fy_err_r = abs(K_r[1, 1] - K_gt_r[1, 1]) / K_gt_r[1, 1] * 100
    bl_err = abs(baseline - baseline_gt) / baseline_gt * 100
    avg_err = (fx_err_l + fy_err_l + fx_err_r + fy_err_r) / 4

    print(f"\n  {name}")
    print(f"    Left K:  fx={K_l[0,0]:.2f} fy={K_l[1,1]:.2f}")
    print(f"    Right K: fx={K_r[0,0]:.2f} fy={K_r[1,1]:.2f}")
    print(f"    Left reproj: {reproj_l:.4f} px | Right reproj: {reproj_r:.4f} px")
    print(f"    fx_err L: {fx_err_l:.2f}% R: {fx_err_r:.2f}%")
    print(f"    Baseline: {baseline:.4f}m (err {bl_err:.2f}%)")

    return {
        "name": name,
        "fx_l": K_l[0, 0], "fy_l": K_l[1, 1],
        "fx_err_l": fx_err_l, "fy_err_l": fy_err_l,
        "fx_r": K_r[0, 0], "fy_r": K_r[1, 1],
        "fx_err_r": fx_err_r, "fy_err_r": fy_err_r,
        "reproj_l": reproj_l, "reproj_r": reproj_r,
        "baseline": baseline, "baseline_err": bl_err,
        "avg_intrinsic_err": avg_err,
    }


def main():
    with open("sample_data/synthetic/ground_truth.json") as f:
        gt = json.load(f)
    K_gt_l = np.array(gt["K_left"])
    K_gt_r = np.array(gt["K_right"])
    baseline_gt = gt["baseline_m"]

    obj_stereo, left_pts, right_pts, img_size, used_pairs = collect_stereo_points(
        "sample_data/synthetic/stereo/left",
        "sample_data/synthetic/stereo/right",
        (9, 6), 0.025
    )

    obj_f32 = [p.astype(np.float32) for p in obj_stereo]
    left_f32 = [p.astype(np.float32) for p in left_pts]
    right_f32 = [p.astype(np.float32) for p in right_pts]
    flags_arzt = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST

    all_results = []

    # ── [A] Default: CALIB_FIX_INTRINSIC ──
    print("=" * 60)
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_f32, left_f32, img_size, None, None)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_f32, right_f32, img_size, None, None)
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_f32, left_f32, right_f32, K_l, dist_l, K_r, dist_r, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    all_results.append(stereo_report(
        "[A] Default (FIX_INTRINSIC)", K_l, dist_l, K_r, dist_r, R, T, E, F,
        obj_stereo, left_pts, right_pts, K_gt_l, K_gt_r, baseline_gt
    ))

    # ── [B] Refine intrinsics (flags=0) ──
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_f32, left_f32, img_size, None, None)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_f32, right_f32, img_size, None, None)
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_f32, left_f32, right_f32, K_l, dist_l, K_r, dist_r, img_size, flags=0
    )
    all_results.append(stereo_report(
        "[B] Refine intrinsics (flags=0)", K_l, dist_l, K_r, dist_r, R, T, E, F,
        obj_stereo, left_pts, right_pts, K_gt_l, K_gt_r, baseline_gt
    ))

    # ── [C] Refine + fix aspect ratio ──
    flags_ar = cv2.CALIB_FIX_ASPECT_RATIO
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_f32, left_f32, img_size, None, None, flags=flags_ar)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_f32, right_f32, img_size, None, None, flags=flags_ar)
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_f32, left_f32, right_f32, K_l, dist_l, K_r, dist_r, img_size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    all_results.append(stereo_report(
        "[C] Refine + fix AR", K_l, dist_l, K_r, dist_r, R, T, E, F,
        obj_stereo, left_pts, right_pts, K_gt_l, K_gt_r, baseline_gt
    ))

    # ── [D] Refine + fix AR + zero tangential ──
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_f32, left_f32, img_size, None, None, flags=flags_arzt)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_f32, right_f32, img_size, None, None, flags=flags_arzt)
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_f32, left_f32, right_f32, K_l, dist_l, K_r, dist_r, img_size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    all_results.append(stereo_report(
        "[D] Refine + fix AR + zero tangential", K_l, dist_l, K_r, dist_r, R, T, E, F,
        obj_stereo, left_pts, right_pts, K_gt_l, K_gt_r, baseline_gt
    ))

    # ── [E] Remove outlier pair_001 (idx=1) ──
    keep = [i for i in range(len(obj_stereo)) if i != 1]
    obj_sub = [obj_stereo[i] for i in keep]
    l_sub = [left_pts[i] for i in keep]
    r_sub = [right_pts[i] for i in keep]
    obj_sub_f32 = [p.astype(np.float32) for p in obj_sub]
    l_sub_f32 = [p.astype(np.float32) for p in l_sub]
    r_sub_f32 = [p.astype(np.float32) for p in r_sub]

    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_sub_f32, l_sub_f32, img_size, None, None, flags=flags_arzt)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_sub_f32, r_sub_f32, img_size, None, None, flags=flags_arzt)
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_sub_f32, l_sub_f32, r_sub_f32, K_l, dist_l, K_r, dist_r, img_size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    all_results.append(stereo_report(
        "[E] Remove pair_001 + refine + fix AR + zero tang", K_l, dist_l, K_r, dist_r, R, T, E, F,
        obj_sub, l_sub, r_sub, K_gt_l, K_gt_r, baseline_gt
    ))

    # ── Generate more stereo images ──
    extra_dir = "sample_data/synthetic_extra"
    subprocess.run([
        sys.executable, "scripts/generate_synthetic_data.py",
        "--output_dir", extra_dir, "--num_images", "30", "--seed", "42"
    ], check=True, capture_output=True)

    obj_extra, l_extra, r_extra, _, used_extra = collect_stereo_points(
        f"{extra_dir}/stereo/left", f"{extra_dir}/stereo/right", (9, 6), 0.025
    )
    print(f"\n  Generated {len(used_extra)} extra stereo pairs")

    # ── [F] 30 fresh images + refine ──
    obj_e32 = [p.astype(np.float32) for p in obj_extra]
    l_e32 = [p.astype(np.float32) for p in l_extra]
    r_e32 = [p.astype(np.float32) for p in r_extra]
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_e32, l_e32, img_size, None, None, flags=flags_arzt)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_e32, r_e32, img_size, None, None, flags=flags_arzt)
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_e32, l_e32, r_e32, K_l, dist_l, K_r, dist_r, img_size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    all_results.append(stereo_report(
        "[F] 30 fresh + refine + fix AR + zero tang", K_l, dist_l, K_r, dist_r, R, T, E, F,
        obj_extra, l_extra, r_extra, K_gt_l, K_gt_r, baseline_gt
    ))

    # ── [G] Combined 12+30 = 42 images ──
    obj_combined = obj_stereo + obj_extra
    l_combined = left_pts + l_extra
    r_combined = right_pts + r_extra
    obj_c32 = [p.astype(np.float32) for p in obj_combined]
    l_c32 = [p.astype(np.float32) for p in l_combined]
    r_c32 = [p.astype(np.float32) for p in r_combined]
    _, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_c32, l_c32, img_size, None, None, flags=flags_arzt)
    _, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_c32, r_c32, img_size, None, None, flags=flags_arzt)
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_c32, l_c32, r_c32, K_l, dist_l, K_r, dist_r, img_size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    all_results.append(stereo_report(
        "[G] 42 combined + refine + fix AR + zero tang", K_l, dist_l, K_r, dist_r, R, T, E, F,
        obj_combined, l_combined, r_combined, K_gt_l, K_gt_r, baseline_gt
    ))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by avg intrinsic error)")
    print("=" * 70)
    sorted_r = sorted(all_results, key=lambda x: x["avg_intrinsic_err"])
    print(f"{'Config':<50} | {'fx%_L':>5} | {'fx%_R':>5} | {'BL%':>5} | {'Avg%':>6}")
    print("-" * 80)
    for r in sorted_r:
        print(f"{r['name']:<50} | {r['fx_err_l']:>5.1f} | {r['fx_err_r']:>5.1f} | {r['baseline_err']:>5.2f} | {r['avg_intrinsic_err']:>6.2f}")

    best = sorted_r[0]
    print(f"\nBEST: {best['name']}")
    print(f"  Left fx_err={best['fx_err_l']:.2f}% Right fx_err={best['fx_err_r']:.2f}% Baseline err={best['baseline_err']:.2f}%")

    Path("outputs/stereo").mkdir(parents=True, exist_ok=True)
    with open("outputs/stereo/experiment_stereo_variants.json", "w") as f:
        json.dump(sorted_r, f, indent=2)
    print("\nResults saved to outputs/stereo/experiment_stereo_variants.json")

    shutil.rmtree(extra_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
