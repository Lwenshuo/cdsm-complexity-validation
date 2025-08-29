#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complexity_validation.py  (unified)
--------------------------------------------------
Empirically validates dense decomposition complexity vs matrix dimension
for square, tall, and wide cases. Also times least-squares, pseudoinverse,
and randomized truncated SVD(k). Fits t ≈ a * n^p (log–log).

What it measures
----------------
Square (n x n):
  - LU solve (np.linalg.solve)
  - Cholesky solve (SPD)
  - QR (reduced)
  - Economy SVD

Tall (m x n, m = rho * n, rho >= 1):
  - QR (reduced)
  - Economy SVD
  - Least squares via lstsq
  - Pseudoinverse via pinv
  - Randomized truncated SVD(k)

Wide (m x n, m = rho * n, 0 < rho < 1):
  - QR (reduced)
  - Economy SVD
  - Least squares via lstsq (minimum-norm)
  - Pseudoinverse via pinv
  - Randomized truncated SVD(k)

Outputs
-------
- complexity_timings.csv : raw timings (shape, rho, m_rows, n_cols, n, method, time_s, rep, ...)
- complexity_exponents.json : fitted exponents p for each (shape, rho, method)
- plot_lu_chol.png : LU vs Cholesky (square)
- plot_svd_qr.png  : SVD vs QR (square)
- plot_tall_rho{ρ}_methods.png : tall methods per ρ
- plot_wide_rho{ρ}_methods.png : wide methods per ρ

Usage
-----
$ python complexity_validation.py
# Customize sizes (n), repeats, rho lists, and truncated rank k:
$ python complexity_validation.py --sizes 80,120,160,200,240 --reps 3 \
    --rhos_tall 1.5,2.0,3.0 --rhos_wide 0.5,0.75 \
    --k 16 --rand_oversamp 8 --rand_power 1

Notes
-----
- Defaults are conservative; increase sizes/reps for smoother exponents (~3).
- Randomized SVD shows k-dependent scaling when k << min(m,n).
- No external deps beyond NumPy, pandas, matplotlib.
"""

from __future__ import annotations
import argparse
import time
import math
import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Utilities
# --------------------------

def make_spd_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    # SPD via A A^T + n I
    return A @ A.T + n * np.eye(n)

def time_once(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0

def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)

def chol_solve(A_spd: np.ndarray, b: np.ndarray) -> np.ndarray:
    L = np.linalg.cholesky(A_spd)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

def qr_decomp(A: np.ndarray):
    return np.linalg.qr(A, mode="reduced")

def svd_decomp(A: np.ndarray):
    return np.linalg.svd(A, full_matrices=False)

def lstsq_solve(A: np.ndarray, b: np.ndarray):
    # returns x, residuals, rank, s
    return np.linalg.lstsq(A, b, rcond=None)

def pinv_svd(A: np.ndarray):
    return np.linalg.pinv(A, rcond=1e-12)

def fit_power(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """Fit log(ys) = log(a) + p log(xs); return (a, p)."""
    x = np.log(xs)
    y = np.log(ys)
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    loga, p = coef
    return float(np.exp(loga)), float(p)

def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def parse_float_list(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


# --------------------------
# Randomized truncated SVD
# --------------------------

def randomized_svd(A: np.ndarray, k: int, oversamp: int = 10, n_iter: int = 1, seed: Optional[int] = None):
    """
    Simple randomized SVD (range finder + small SVD).
    Returns U_k, s_k, Vt_k with rank k.
    """
    m, n = A.shape
    ell = min(n, k + oversamp)
    rng = np.random.default_rng(seed)
    Omega = rng.normal(size=(n, ell))  # n x ell
    Y = A @ Omega                      # m x ell

    # Power iterations (optional) to improve spectrum separation
    for _ in range(max(0, n_iter)):
        Y = A @ (A.T @ Y)

    # Orthonormal basis of range(Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")  # m x ell

    # Project to smaller matrix
    B = Q.T @ A                              # ell x n
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)  # (ell x n) SVD

    k_eff = min(k, Ub.shape[1], Vt.shape[0])
    U_k = Q @ Ub[:, :k_eff]  # m x k
    s_k = s[:k_eff]          # k
    Vt_k = Vt[:k_eff, :]     # k x n
    return U_k, s_k, Vt_k


# --------------------------
# Experiments
# --------------------------

def run_square_experiments(sizes: List[int], reps: int, seed_base: int,
                           c: int, m_groups: int, a_per_group: int):
    """
    Returns a list of timing rows (dicts) for square n x n matrices.
    """
    rows = []
    M_formula = 6 + m_groups * a_per_group
    for n in sizes:
        for r in range(reps):
            seed = seed_base + 17 * n + r
            rng = np.random.default_rng(seed)
            A_general = rng.normal(size=(n, n))
            A_spd = make_spd_matrix(n, seed + 1)
            b = rng.normal(size=(n,))

            t_lu  = time_once(lu_solve,  A_general, b)
            t_ch  = time_once(chol_solve, A_spd, b)
            t_qr  = time_once(qr_decomp,  A_general)
            t_svd = time_once(svd_decomp, A_general)

            implied_M = n / float(c)

            for method, t in (("LU_solve", t_lu),
                              ("Cholesky_solve", t_ch),
                              ("QR", t_qr),
                              ("SVD", t_svd)):
                rows.append({
                    "shape": "square",
                    "rho": np.nan,
                    "m_rows": n,
                    "n_cols": n,
                    "n": n,  # size parameter used for fitting
                    "k": np.nan,
                    "method": method,
                    "time_s": t,
                    "rep": r,
                    "M_implied": implied_M,
                    "M_formula": M_formula,
                })
    return rows


def run_rect_experiments(kind: str, sizes: List[int], reps: int, seed_base: int,
                         rhos: List[float], k_trunc: int, oversamp: int, n_power: int):
    """
    kind: "tall" or "wide"
    For each n in sizes, and each rho in rhos:
       tall: m = ceil(rho*n), m >= n
       wide: m = max(1, floor(rho*n)), m < n when rho<1
    Returns timing rows.
    """
    assert kind in ("tall", "wide")
    rows = []
    for n in sizes:
        for rho in rhos:
            if kind == "tall":
                m = int(math.ceil(rho * n))
                if m < n:  # enforce tall
                    m = n
            else:
                m = int(math.floor(rho * n))
                if m >= n:
                    m = max(1, n - 1)  # enforce wide

            for r in range(reps):
                seed = seed_base + 7919 * (int(1000 * rho)) + 17 * n + r
                rng = np.random.default_rng(seed)
                A = rng.normal(size=(m, n))
                b = rng.normal(size=(m,))

                # Effective k (cannot exceed min(m,n)-1)
                k_eff = max(1, min(k_trunc, min(m, n) - 1))

                # Methods
                # 1) QR
                t_qr = time_once(qr_decomp, A)

                # 2) SVD
                t_svd = time_once(svd_decomp, A)

                # 3) Least squares (minimum-norm when underdetermined)
                t_lstsq = time_once(lstsq_solve, A, b)

                # 4) Pseudoinverse via SVD
                t_pinv = time_once(pinv_svd, A)

                # 5) Randomized truncated SVD(k)
                t_randsvd = time_once(randomized_svd, A, k_eff, oversamp, n_power, seed + 123)

                for method, t, kused in (
                    ("QR", t_qr, np.nan),
                    ("SVD", t_svd, np.nan),
                    ("LS_QR", t_lstsq, np.nan),
                    ("Pinv_SVD", t_pinv, np.nan),
                    ("RandSVD_k", t_randsvd, k_eff),
                ):
                    rows.append({
                        "shape": kind,
                        "rho": float(rho),
                        "m_rows": m,
                        "n_cols": n,
                        "n": n,
                        "k": kused,
                        "method": method,
                        "time_s": t,
                        "rep": r,
                        "M_implied": np.nan,
                        "M_formula": np.nan,
                    })
    return rows


# --------------------------
# Plotting & Fitting
# --------------------------

def fit_all_exponents(df: pd.DataFrame) -> Dict:
    """
    Fit t ≈ a * n^p for each (shape, rho, method).
    Returns nested dict for JSON export.
    """
    result = {}
    for shape in sorted(df["shape"].unique()):
        df_s = df[df["shape"] == shape]
        if shape not in result:
            result[shape] = {}

        if shape == "square":
            # rho = null key
            result[shape]["rho=null"] = {}
            for method in sorted(df_s["method"].unique()):
                g = df_s[df_s["method"] == method].groupby("n")["time_s"].mean().reset_index()
                a_coef, p = fit_power(g["n"].values, g["time_s"].values)
                result[shape]["rho=null"][method] = {"a": a_coef, "p_exponent": p}
        else:
            for rho in sorted(df_s["rho"].dropna().unique()):
                key = f"rho={rho}"
                result[shape][key] = {}
                for method in sorted(df_s["method"].unique()):
                    g = df_s[(df_s["rho"] == rho) & (df_s["method"] == method)] \
                        .groupby("n")["time_s"].mean().reset_index()
                    if len(g) >= 2 and g["time_s"].gt(0).all():
                        a_coef, p = fit_power(g["n"].values, g["time_s"].values)
                        result[shape][key][method] = {"a": a_coef, "p_exponent": p}
                    else:
                        result[shape][key][method] = {"a": float("nan"), "p_exponent": float("nan")}
    return result


def plot_square(df: pd.DataFrame, outdir: str):
    df_sq = df[df["shape"] == "square"]

    # LU vs Cholesky
    plt.figure()
    for method in ["LU_solve", "Cholesky_solve"]:
        g = df_sq[df_sq["method"] == method].groupby("n")["time_s"].mean().reset_index()
        if len(g):
            plt.loglog(g["n"], g["time_s"], marker="o", label=method)
    plt.xlabel("Matrix size n")
    plt.ylabel("Time (s)")
    plt.title("LU vs Cholesky (log–log) — expect slope ≈ 3")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "plot_lu_chol.png")
    plt.savefig(path, dpi=180)

    # SVD vs QR
    plt.figure()
    for method in ["QR", "SVD"]:
        g = df_sq[df_sq["method"] == method].groupby("n")["time_s"].mean().reset_index()
        if len(g):
            plt.loglog(g["n"], g["time_s"], marker="o", label=method)
    plt.xlabel("Matrix size n")
    plt.ylabel("Time (s)")
    plt.title("SVD vs QR (log–log) — both ~O(n^3), SVD has larger constant")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "plot_svd_qr.png")
    plt.savefig(path, dpi=180)


def plot_rect_by_rho(df: pd.DataFrame, kind: str, outdir: str):
    assert kind in ("tall", "wide")
    df_k = df[df["shape"] == kind]
    if df_k.empty:
        return

    rho_list = sorted(df_k["rho"].dropna().unique())
    methods = ["QR", "SVD", "LS_QR", "Pinv_SVD", "RandSVD_k"]

    for rho in rho_list:
        plt.figure()
        for method in methods:
            g = df_k[(df_k["rho"] == rho) & (df_k["method"] == method)] \
                .groupby("n")["time_s"].mean().reset_index()
            if len(g):
                plt.loglog(g["n"], g["time_s"], marker="o", label=method)
        plt.xlabel("n (number of columns)")
        plt.ylabel("Time (s)")
        title = f"{kind.capitalize()} matrices (rho={rho}) — log–log"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        rho_safe = str(rho).replace(".", "p")
        path = os.path.join(outdir, f"plot_{kind}_rho{rho_safe}_methods.png")
        plt.savefig(path, dpi=180)


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate dense decomposition complexity vs n with square/tall/wide & truncated SVD(k).")
    ap.add_argument("--sizes", type=str, default="60,90,120,160,200",
                    help="comma-separated base n values (number of columns).")
    ap.add_argument("--reps", type=int, default=2, help="repetitions per size (avg).")
    ap.add_argument("--c", type=int, default=1, help="proportional constant in n = c*M (square only, for reporting).")
    ap.add_argument("--m", type=int, default=3, help="number of arm-shape groups m (square only, for reporting).")
    ap.add_argument("--a", type=int, default=1, help="arm-angle variables per group a (square only, for reporting).")
    ap.add_argument("--rhos_tall", type=str, default="1.5,2.0",
                    help="comma-separated rho values for tall matrices (rho >= 1).")
    ap.add_argument("--rhos_wide", type=str, default="0.5",
                    help="comma-separated rho values for wide matrices (0 < rho < 1).")
    ap.add_argument("--k", type=int, default=12, help="target rank for randomized truncated SVD(k).")
    ap.add_argument("--rand_oversamp", type=int, default=6, help="oversampling parameter for randomized SVD.")
    ap.add_argument("--rand_power", type=int, default=1, help="power iterations for randomized SVD.")
    ap.add_argument("--seed_base", type=int, default=1000, help="base RNG seed.")
    ap.add_argument("--outdir", type=str, default=".", help="output directory.")
    ap.add_argument("--do_square", action="store_true", help="run square experiments (off by default if not set).")
    ap.add_argument("--do_tall", action="store_true", help="run tall experiments (off by default if not set).")
    ap.add_argument("--do_wide", action="store_true", help="run wide experiments (off by default if not set).")
    args = ap.parse_args()

    sizes = parse_int_list(args.sizes)
    reps = args.reps
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # If user didn't specify any, run all by default:
    if not (args.do_square or args.do_tall or args.do_wide):
        run_square = True
        run_tall = True
        run_wide = True
    else:
        run_square = args.do_square
        run_tall = args.do_tall
        run_wide = args.do_wide

    rows = []

    # Square
    if run_square:
        rows += run_square_experiments(
            sizes=sizes, reps=reps, seed_base=args.seed_base,
            c=args.c, m_groups=args.m, a_per_group=args.a
        )

    # Tall
    if run_tall:
        rhos_tall = parse_float_list(args.rhos_tall) if args.rhos_tall else []
        if rhos_tall:
            rows += run_rect_experiments(
                kind="tall", sizes=sizes, reps=reps, seed_base=args.seed_base + 11,
                rhos=rhos_tall, k_trunc=args.k, oversamp=args.rand_oversamp, n_power=args.rand_power
            )

    # Wide
    if run_wide:
        rhos_wide = parse_float_list(args.rhos_wide) if args.rhos_wide else []
        if rhos_wide:
            rows += run_rect_experiments(
                kind="wide", sizes=sizes, reps=reps, seed_base=args.seed_base + 22,
                rhos=rhos_wide, k_trunc=args.k, oversamp=args.rand_oversamp, n_power=args.rand_power
            )

    # Aggregate to DataFrame
    df = pd.DataFrame(rows)

    # File paths
    csv_path = f"{outdir}/complexity_timings.csv"
    json_path = f"{outdir}/complexity_exponents.json"

    # Save CSV
    df.to_csv(csv_path, index=False)

    # Fit exponents -> JSON
    exponents = fit_all_exponents(df)
    with open(json_path, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(exponents, f, indent=2)

    # Plots
    if run_square:
        plot_square(df, outdir)
    if run_tall:
        plot_rect_by_rho(df, kind="tall", outdir=outdir)
    if run_wide:
        plot_rect_by_rho(df, kind="wide", outdir=outdir)

    # Console summary
    print("=== Empirical complexity exponents (t ≈ a * n^p) ===")
    # Square
    if run_square:
        print("[square]")
        df_sq = df[df["shape"] == "square"]
        for method in sorted(df_sq["method"].unique()):
            g = df_sq[df_sq["method"] == method].groupby("n")["time_s"].mean().reset_index()
            a_coef, p = fit_power(g["n"].values, g["time_s"].values)
            print(f"  {method:>16}: p ≈ {p:.2f}, a ≈ {a_coef:.3e}")
    # Tall / Wide
    for kind in ("tall", "wide"):
        if kind == "tall" and not run_tall:
            continue
        if kind == "wide" and not run_wide:
            continue
        df_k = df[df["shape"] == kind]
        if df_k.empty:
            continue
        print(f"[{kind}]")
        for rho in sorted(df_k["rho"].dropna().unique()):
            print(f"  rho={rho}:")
            for method in sorted(df_k["method"].unique()):
                g = df_k[(df_k["rho"] == rho) & (df_k["method"] == method)] \
                    .groupby("n")["time_s"].mean().reset_index()
                if len(g) >= 2 and g["time_s"].gt(0).all():
                    a_coef, p = fit_power(g["n"].values, g["time_s"].values)
                    print(f"    {method:>16}: p ≈ {p:.2f}, a ≈ {a_coef:.3e}")

    # Final paths
    print("\nSaved:")
    print(f"- {csv_path}\n- {json_path}")
    if run_square:
        print(f"- {outdir}/plot_lu_chol.png")
        print(f"- {outdir}/plot_svd_qr.png")
    if run_tall:
        for rho in sorted(df[df['shape']=="tall"]["rho"].dropna().unique()):
            print(f"- {outdir}/plot_tall_rho{str(rho).replace('.', 'p')}_methods.png")
    if run_wide:
        for rho in sorted(df[df['shape']=="wide"]["rho"].dropna().unique()):
            print(f"- {outdir}/plot_wide_rho{str(rho).replace('.', 'p')}_methods.png")

    # Parameters echo
    print("\nParameters:")
    print(f"  sizes={sizes}, reps={reps}, seed_base={args.seed_base}")
    print(f"  square: c={args.c}, m={args.m}, a={args.a} (M=6+ma={6 + args.m * args.a})")
    print(f"  tall rhos: {args.rhos_tall}")
    print(f"  wide rhos: {args.rhos_wide}")
    print(f"  RandSVD: k={args.k}, oversamp={args.rand_oversamp}, power={args.rand_power}")
    print("Note: For rectangles, fit is w.r.t. n (number of columns). Keep rho fixed to see slope≈3.")
    

if __name__ == "__main__":
    main()
