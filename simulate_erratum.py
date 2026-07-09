#!/usr/bin/env python3
"""
Generate Figure 1 for the erratum: max-weight (MW) vs modified max-weight (MMW).

This script simulates the two-sided queueing example from the erratum using the
corrected non-CRP compatibility graph

    E = {(1, 1), (1, 2), (2, 2)},   E_r = {(1, 2)}.

The modified max-weight policy uses only the non-redundant diagonal edges
{(1, 1), (2, 2)}.  The full max-weight policy uses all edges in E, with ties
broken in favor of the diagonal edges.

The script creates two plot panels, without the diagnostic exact/green line:
  1. profit loss versus eta;
  2. log(profit loss) versus log(eta), with fitted slopes.

Recommended command for manuscript-quality output:

    python simulate_erratum_fig1.py --preset paper --outdir results/fig1

Quick smoke test:

    python simulate_erratum_fig1.py --preset quick --outdir results/quick

The paper preset uses 10% burn-in, i.e., burn-in length = 0.10 * measured length.
It uses longer runs at eta=10000 because the MMW chain mixes slowly there.

Dependencies: numpy, matplotlib.  numba is optional but strongly recommended.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for environments without numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

import matplotlib.pyplot as plt


# ----- Model primitives -----------------------------------------------------
# Corrected graph: E={(1,1),(1,2),(2,2)} with redundant edge (1,2).
# Fluid optimum: lambda*=mu*=(1,1), chi_11*=chi_22*=1, chi_12*=0.
N_TYPES = 2
GAMMA_STAR = 4.5  # fluid objective: (5-1)*1 + (4-1)*1 - 1.5*1 - 1*1

POLICY_MW = 0
POLICY_MMW = 1
POLICY_NAMES = {POLICY_MW: "MW", POLICY_MMW: "MMW"}


@dataclass(frozen=True)
class RunConfig:
    etas: Tuple[int, ...]
    steps_default: int
    reps_default: int
    burn_fraction: float
    eta10000_steps: int | None = None
    eta10000_reps_mw: int | None = None
    eta10000_reps_mmw: int | None = None
    seed: int = 12345


def preset_config(name: str) -> RunConfig:
    """Return preset simulation settings."""
    etas = (10, 100, 500, 1000, 2000, 5000, 10000)
    if name == "quick":
        return RunConfig(
            etas=etas,
            steps_default=200_000,
            reps_default=2,
            burn_fraction=0.10,
            eta10000_steps=500_000,
            eta10000_reps_mw=2,
            eta10000_reps_mmw=2,
            seed=12345,
        )
    if name == "paper":
        return RunConfig(
            etas=etas,
            steps_default=10_000_000,
            reps_default=3,
            burn_fraction=0.10,
            # eta=10000 needs more samples, especially for MMW.
            eta10000_steps=100_000_000,
            eta10000_reps_mw=4,
            eta10000_reps_mmw=12,
            seed=12345,
        )
    raise ValueError(f"Unknown preset: {name}")


@njit(cache=True)
def _simulate_one_replication(
    eta: int,
    policy: int,
    measured_steps: int,
    burn_steps: int,
    seed: int,
) -> float:
    """Simulate one replication and return average profit rate.

    State variables are post-matching queue lengths:
      qc1, qc2: customer queues 1 and 2
      qs1, qs2: server queues 1 and 2

    Uniformization constant is c = 4*eta, the maximum total arrival rate.
    Reward is accumulated over uniformized states after burn-in.
    """
    np.random.seed(seed)

    qc1 = 0
    qc2 = 0
    qs1 = 0
    qs2 = 0

    sigma = (eta ** (2.0 / 3.0)) * (2.0 ** (-1.0 / 3.0))
    c_uniform = 4.0 * eta
    total_steps = burn_steps + measured_steps
    profit_sum = 0.0

    for t in range(total_steps):
        # Two-price policy with tau_max=0 and theta=phi=(1,1).
        lam1 = float(eta) if qc1 <= 0 else float(eta) - sigma
        lam2 = float(eta) if qc2 <= 0 else float(eta) - sigma
        mu1 = float(eta) if qs1 <= 0 else float(eta) - sigma
        mu2 = float(eta) if qs2 <= 0 else float(eta) - sigma

        if t >= burn_steps:
            # Scaled inverse demand/supply curves:
            # F1(x)=5-x, F2(x)=4-x, G1(x)=1.5*x, G2(x)=x.
            rev = lam1 * (5.0 - lam1 / eta) + lam2 * (4.0 - lam2 / eta)
            cost = mu1 * (1.5 * mu1 / eta) + mu2 * (mu2 / eta)
            holding = qc1 + qc2 + qs1 + qs2
            profit_sum += rev - cost - holding

        # Uniformized transition.
        u = np.random.random() * c_uniform

        if u < lam1:
            # Customer type 1 arrival.
            if policy == POLICY_MW:
                # Full graph: customer 1 is compatible only with server 1.
                if qs1 > 0:
                    qs1 -= 1
                else:
                    qc1 += 1
            else:
                # MMW: diagonal edge only.
                if qs1 > 0:
                    qs1 -= 1
                else:
                    qc1 += 1

        elif u < lam1 + lam2:
            # Customer type 2 arrival.
            if policy == POLICY_MW:
                # Full graph: compatible with server 1 and server 2.
                # Tie-breaking favors diagonal edge (2,2), i.e., server 2.
                if qs2 >= qs1 and qs2 > 0:
                    qs2 -= 1
                elif qs1 > 0:
                    qs1 -= 1
                else:
                    qc2 += 1
            else:
                # MMW: diagonal edge only.
                if qs2 > 0:
                    qs2 -= 1
                else:
                    qc2 += 1

        elif u < lam1 + lam2 + mu1:
            # Server type 1 arrival.
            if policy == POLICY_MW:
                # Full graph: compatible with customer 1 and customer 2.
                # Tie-breaking favors diagonal edge (1,1), i.e., customer 1.
                if qc1 >= qc2 and qc1 > 0:
                    qc1 -= 1
                elif qc2 > 0:
                    qc2 -= 1
                else:
                    qs1 += 1
            else:
                # MMW: diagonal edge only.
                if qc1 > 0:
                    qc1 -= 1
                else:
                    qs1 += 1

        elif u < lam1 + lam2 + mu1 + mu2:
            # Server type 2 arrival.
            if policy == POLICY_MW:
                # Full graph: server 2 is compatible only with customer 2.
                if qc2 > 0:
                    qc2 -= 1
                else:
                    qs2 += 1
            else:
                # MMW: diagonal edge only.
                if qc2 > 0:
                    qc2 -= 1
                else:
                    qs2 += 1
        else:
            # Dummy transition under uniformization.
            pass

    return profit_sum / measured_steps


def _run_policy_eta(
    eta: int,
    policy: int,
    measured_steps: int,
    burn_steps: int,
    reps: int,
    seed_base: int,
) -> Dict[str, float]:
    profits: List[float] = []
    losses: List[float] = []
    start = time.time()

    for r in range(reps):
        seed = seed_base + 10_000 * eta + 100 * policy + r
        avg_profit = _simulate_one_replication(
            int(eta), int(policy), int(measured_steps), int(burn_steps), int(seed)
        )
        loss = eta * GAMMA_STAR - avg_profit
        profits.append(avg_profit)
        losses.append(loss)

    elapsed = time.time() - start
    losses_arr = np.asarray(losses, dtype=float)
    profits_arr = np.asarray(profits, dtype=float)
    se_loss = float(losses_arr.std(ddof=1) / math.sqrt(reps)) if reps > 1 else float("nan")

    return {
        "eta": float(eta),
        "policy": float(policy),
        "avg_profit_mean": float(profits_arr.mean()),
        "avg_profit_sd": float(profits_arr.std(ddof=1)) if reps > 1 else 0.0,
        "loss_mean": float(losses_arr.mean()),
        "loss_sd": float(losses_arr.std(ddof=1)) if reps > 1 else 0.0,
        "loss_se": se_loss,
        "steps": float(measured_steps),
        "burn_steps": float(burn_steps),
        "reps": float(reps),
        "elapsed_seconds": float(elapsed),
    }


def run_experiment(config: RunConfig) -> List[Dict[str, float]]:
    """Run all eta-policy combinations."""
    rows: List[Dict[str, float]] = []
    for eta in config.etas:
        for policy in (POLICY_MW, POLICY_MMW):
            measured_steps = config.steps_default
            reps = config.reps_default
            if eta == 10000 and config.eta10000_steps is not None:
                measured_steps = config.eta10000_steps
                if policy == POLICY_MW and config.eta10000_reps_mw is not None:
                    reps = config.eta10000_reps_mw
                if policy == POLICY_MMW and config.eta10000_reps_mmw is not None:
                    reps = config.eta10000_reps_mmw
            burn_steps = int(round(config.burn_fraction * measured_steps))
            print(
                f"Running eta={eta:5d}, policy={POLICY_NAMES[policy]:3s}, "
                f"steps={measured_steps:,}, burn={burn_steps:,}, reps={reps}",
                flush=True,
            )
            row = _run_policy_eta(
                eta=eta,
                policy=policy,
                measured_steps=measured_steps,
                burn_steps=burn_steps,
                reps=reps,
                seed_base=config.seed,
            )
            rows.append(row)
            print(
                f"  loss={row['loss_mean']:.3f} +/- {row['loss_se']:.3f} "
                f"(elapsed {row['elapsed_seconds']:.1f}s)",
                flush=True,
            )
    return rows


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eta",
        "policy_name",
        "policy",
        "avg_profit_mean",
        "avg_profit_sd",
        "loss_mean",
        "loss_sd",
        "loss_se",
        "steps",
        "burn_steps",
        "reps",
        "elapsed_seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["policy_name"] = POLICY_NAMES[int(row["policy"])]
            writer.writerow(out)


def read_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "eta": float(r["eta"]),
                "policy": float(r["policy"]),
                "avg_profit_mean": float(r["avg_profit_mean"]),
                "avg_profit_sd": float(r.get("avg_profit_sd", 0.0)),
                "loss_mean": float(r["loss_mean"]),
                "loss_sd": float(r.get("loss_sd", 0.0)),
                "loss_se": float(r.get("loss_se", "nan")),
                "steps": float(r.get("steps", 0.0)),
                "burn_steps": float(r.get("burn_steps", 0.0)),
                "reps": float(r.get("reps", 0.0)),
                "elapsed_seconds": float(r.get("elapsed_seconds", 0.0)),
            })
    return rows


def _series(rows: List[Dict[str, float]], policy: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered = sorted([r for r in rows if int(r["policy"]) == policy], key=lambda r: r["eta"])
    eta = np.asarray([r["eta"] for r in filtered], dtype=float)
    loss = np.asarray([r["loss_mean"] for r in filtered], dtype=float)
    se = np.asarray([r["loss_se"] for r in filtered], dtype=float)
    return eta, loss, se


def make_figure(rows: List[Dict[str, float]], outdir: Path, basename: str = "erratum_fig1") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    eta_mw, loss_mw, se_mw = _series(rows, POLICY_MW)
    eta_mmw, loss_mmw, se_mmw = _series(rows, POLICY_MMW)

    if not np.array_equal(eta_mw, eta_mmw):
        raise ValueError("MW and MMW eta grids do not match")

    slope_mw, intercept_mw = np.polyfit(np.log(eta_mw), np.log(loss_mw), 1)
    slope_mmw, intercept_mmw = np.polyfit(np.log(eta_mmw), np.log(loss_mmw), 1)

    # Combined manuscript-style two-panel figure.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(eta_mw, loss_mw, "o-", label="MW")
    ax.plot(eta_mmw, loss_mmw, "s--", label="MMW")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("profit loss")
    ax.legend(frameon=True)

    ax = axes[1]
    log_eta = np.log(eta_mw)
    ax.plot(log_eta, np.log(loss_mw), "o-", label=f"MW, slope={slope_mw:.2f}")
    ax.plot(log_eta, np.log(loss_mmw), "s--", label=f"MMW, slope={slope_mmw:.2f}")
    ax.set_xlabel(r"$\log(\eta)$")
    ax.set_ylabel("log (profit loss)")
    ax.legend(frameon=True)

    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{basename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Separate panels, useful if the paper template places them manually.
    fig1, ax1 = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    ax1.plot(eta_mw, loss_mw, "o-", label="MW")
    ax1.plot(eta_mmw, loss_mmw, "s--", label="MMW")
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel("profit loss")
    ax1.legend(frameon=True)
    for ext in ("png", "pdf"):
        fig1.savefig(outdir / f"{basename}_loss_vs_eta.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    ax2.plot(log_eta, np.log(loss_mw), "o-", label=f"MW, slope={slope_mw:.2f}")
    ax2.plot(log_eta, np.log(loss_mmw), "s--", label=f"MMW, slope={slope_mmw:.2f}")
    ax2.set_xlabel(r"$\log(\eta)$")
    ax2.set_ylabel("log (profit loss)")
    ax2.legend(frameon=True)
    for ext in ("png", "pdf"):
        fig2.savefig(outdir / f"{basename}_loglog.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    with (outdir / f"{basename}_slopes.txt").open("w") as f:
        f.write(f"MW slope:  {slope_mw:.6f}\n")
        f.write(f"MMW slope: {slope_mmw:.6f}\n")

    print(f"Saved plots to {outdir}")
    print(f"MW slope:  {slope_mw:.3f}")
    print(f"MMW slope: {slope_mmw:.3f}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate erratum MW vs MMW simulation plots.")
    parser.add_argument("--preset", choices=("quick", "paper"), default="paper",
                        help="Simulation preset. 'paper' is long; 'quick' is for smoke tests.")
    parser.add_argument("--outdir", type=Path, default=Path("results/fig1"),
                        help="Output directory for CSV and plots.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional CSV path. Defaults to OUTDIR/erratum_fig1_results.csv.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation and plot from --csv.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override measured steps for all etas except eta=10000 override.")
    parser.add_argument("--reps", type=int, default=None,
                        help="Override replications for all etas except eta=10000 override.")
    parser.add_argument("--burn-frac", type=float, default=None,
                        help="Burn-in as a fraction of measured steps. Default preset value is 0.10.")
    parser.add_argument("--eta10000-steps", type=int, default=None,
                        help="Override measured steps for eta=10000.")
    parser.add_argument("--eta10000-reps-mw", type=int, default=None,
                        help="Override MW replications for eta=10000.")
    parser.add_argument("--eta10000-reps-mmw", type=int, default=None,
                        help="Override MMW replications for eta=10000.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed base.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = preset_config(args.preset)

    # Apply command-line overrides.
    config = RunConfig(
        etas=config.etas,
        steps_default=args.steps if args.steps is not None else config.steps_default,
        reps_default=args.reps if args.reps is not None else config.reps_default,
        burn_fraction=args.burn_frac if args.burn_frac is not None else config.burn_fraction,
        eta10000_steps=args.eta10000_steps if args.eta10000_steps is not None else config.eta10000_steps,
        eta10000_reps_mw=(args.eta10000_reps_mw
                          if args.eta10000_reps_mw is not None else config.eta10000_reps_mw),
        eta10000_reps_mmw=(args.eta10000_reps_mmw
                           if args.eta10000_reps_mmw is not None else config.eta10000_reps_mmw),
        seed=args.seed if args.seed is not None else config.seed,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv if args.csv is not None else args.outdir / "erratum_fig1_results.csv"

    if not NUMBA_AVAILABLE:
        print("WARNING: numba is not installed. The simulation will be slow.", file=sys.stderr)

    if args.plot_only:
        rows = read_csv(csv_path)
    else:
        rows = run_experiment(config)
        write_csv(rows, csv_path)
        print(f"Saved CSV to {csv_path}")

    make_figure(rows, args.outdir, basename="erratum_fig1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
