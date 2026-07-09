r"""
Simulation for the erratum of "Dynamic Pricing and Matching for Two-Sided Queues"
(Varma, Bumpensanti, Maguluri, Wang, Operations Research 71(1), 2023).

Reproduces Section 3 / Figure 1: profit loss of the two-price policy combined with
  (a) the max-weight (MW) matching policy, and
  (b) the modified max-weight (MMW) matching policy
on the 2x2 example whose compatibility graph does NOT satisfy the CRP condition.

Model (from the original paper, Sections 2-5):
  - Continuous-time MDP, simulated via uniformization as a discrete-time chain.
  - Two customer (demand) types j in {1,2} and two server (supply) types i in {1,2}.
  - Poisson arrivals; price sets the arrival rate through demand/supply curves.
        Demand curves   F_1(l)=5-l ,  F_2(l)=4-l        (price as a function of rate)
        Supply curves   G_1(m)=1.5 m , G_2(m)=m
  - Fluid solution used by the paper:  lambda* = mu* = (1,1),  chi*_ij = 1[i=j].
        These curves are designed so each diagonal link is marginally balanced:
            customer 1 (F_1) with server 1 (G_1):  MR_1 = MC_1 = 3
            customer 2 (F_2) with server 2 (G_2):  MR_2 = MC_2 = 2
        (MR_j = F_j(1)+F_j'(1),  MC_i = G_i(1)+G_i'(1).)

  - Compatibility graph  E = {(1,1), (1,2), (2,2)}   with edges written (server i, customer j):
        server 1 -> customer 1  (dedicated) and customer 2  (redundant)
        server 2 -> customer 2  (dedicated)
    The redundant edge (1,2) joins the expensive server 1 (marginal cost 3) to the
    low-value customer 2 (marginal revenue 2), so using it is not worthwhile: chi*_12 = 0,
    E_r = {(1,2)}, and E\E_r = {(1,1),(2,2)}.  This makes the diagonal lambda*=mu*=1 the
    unique fluid optimum (see verify_fluid()).
  - Two-price policy (paper Eqs 10-11), with tau_max = 0 and sigma = eta^{2/3} n^{-1/3}:
        lambda_j = eta*lambda*_j              if q^(c)_j <= tau_max (= 0, i.e. queue empty)
                 = eta*lambda*_j - theta_j*sigma   otherwise
        mu_i     = eta*mu*_i                   if q^(s)_i <= tau_max
                 = eta*mu*_i     - phi_i*sigma      otherwise
  - MW matching uses all edges of E (ties broken in favour of (1,1),(2,2) over (1,2)).
    MMW matching uses only E\E_r = {(1,1),(2,2)}.

Profit loss (paper Definition 4 / Eq 9):
        L^eta = eta*pi*  -  E[ realized revenue rate ]  +  s * E[ sum of queue lengths ],
  where pi* = <F(lambda*),lambda*> - <G(mu*),mu*>  is the fluid optimal value AT the
  fluid solution used by the paper (= 4.5 for lambda*=mu*=1), s = 1 is the holding cost.

  Because rates are a deterministic function of the (random) state, the realized revenue
  rate is a deterministic function of the state, so we accumulate the per-slot revenue
  *reduction* relative to eta*pi* directly (low variance, no catastrophic cancellation).
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                      # pragma: no cover
    HAVE_NUMBA = False
    def njit(*a, **k):
        def wrap(f):
            return f
        return (wrap if not a else wrap(a[0]))

# --------------------------------------------------------------------------- #
#  Problem constants                                                          #
# --------------------------------------------------------------------------- #
N_TYPES = 2                    # n = m = 2
S_HOLD  = 1.0                  # unit holding cost s
LAM_STAR = 1.0                 # lambda*_1 = lambda*_2 = 1
MU_STAR  = 1.0                 # mu*_1 = mu*_2 = 1
# theta, phi : positive constants of the two-price policy (unspecified in the erratum;
# they only affect the constant, not the scaling exponent). Use 1 for all types.
THETA = 1.0
PHI   = 1.0

# Demand/supply curves and the fluid optimal value pi* at lambda*=mu*=1.
def F1(x): return 5.0 - x
def F2(x): return 4.0 - x
def G1(x): return 1.5 * x
def G2(x): return x

PI_STAR = (LAM_STAR * F1(LAM_STAR) + LAM_STAR * F2(LAM_STAR)
           - (MU_STAR * G1(MU_STAR) + MU_STAR * G2(MU_STAR)))   # = 7 - 2.5 = 4.5


def verify_fluid():
    """Grid-search the fluid QP to confirm lambda*=mu*=1 (chi diagonal) is the optimum."""
    # graph E={(1,1),(1,2),(2,2)};  a=chi11, b=chi22, e=chi12 (server1->customer2)
    def obj(a, b, e):
        l1, l2 = a, e + b          # customer arrival rates
        m1, m2 = a + e, b          # server arrival rates
        return (l1 * F1(l1) + l2 * F2(l2)) - (m1 * G1(m1) + m2 * G2(m2))
    gs = np.linspace(0, 3, 151)
    A, B, D = np.meshgrid(gs, gs, gs, indexing='ij')
    V = obj(A, B, D)
    k = np.unravel_index(np.argmax(V), V.shape)
    bx = (A[k], B[k], D[k])
    return dict(opt_chi=bx, opt_val=float(V[k]),
                diagonal_val=obj(1, 1, 0), pi_star_used=PI_STAR)


# --------------------------------------------------------------------------- #
#  Core simulator (uniformized DTMC)                                          #
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=True)
def _simulate(eta, sigma, c, use_mw, n_steps, burn, seed,
              th1, th2, ph1, ph2):
    np.random.seed(seed)

    # queue state: customers (c) and servers (s), types 1,2
    qc1 = 0; qc2 = 0; qs1 = 0; qs2 = 0

    base_lam = eta * 1.0       # eta * lambda* (lambda*=1)
    base_mu  = eta * 1.0       # eta * mu*     (mu*=1)

    sum_revloss = 0.0          # sum over slots of (eta*pi* - realized revenue rate)
    sum_hold    = 0.0          # sum over slots of (qc1+qc2+qs1+qs2)
    cnt = 0

    for k in range(n_steps):
        # --- rates from the two-price policy (tau_max = 0) ---
        lam1 = base_lam if qc1 == 0 else base_lam - th1 * sigma
        lam2 = base_lam if qc2 == 0 else base_lam - th2 * sigma
        mu1  = base_mu  if qs1 == 0 else base_mu  - ph1 * sigma
        mu2  = base_mu  if qs2 == 0 else base_mu  - ph2 * sigma

        if k >= burn:
            # per-slot revenue reduction relative to eta*pi*
            #   demand reduction  (>=0)      minus   supply cost reduction (>=0)
            # demand j: eta*Fj(1) - lam_j*Fj(lam_j/eta)
            dred = 0.0
            if qc1 != 0:
                dred += eta * (5.0 - 1.0) - lam1 * (5.0 - lam1 / eta)
            if qc2 != 0:
                dred += eta * (4.0 - 1.0) - lam2 * (4.0 - lam2 / eta)
            sred = 0.0
            if qs1 != 0:
                sred += eta * (1.5 * 1.0) - mu1 * (1.5 * mu1 / eta)
            if qs2 != 0:
                sred += eta * (1.0 * 1.0) - mu2 * (1.0 * mu2 / eta)
            sum_revloss += dred - sred
            sum_hold    += (qc1 + qc2 + qs1 + qs2)
            cnt += 1

        # --- sample the (at most one) arrival ---
        u = np.random.random() * c      # scale by c so thresholds are the raw rates
        # order: customer1, customer2, server1, server2, (idle)
        # Graph E = {(1,1),(1,2),(2,2)}:
        #   customer 1  <- served only by server 1
        #   customer 2  <- served by server 2 (dedicated) or server 1 (redundant, MW only)
        #   server   1  -> serves customer 1 (dedicated) or customer 2 (redundant, MW only)
        #   server   2  -> serves only customer 2
        if u < lam1:
            # customer 1 arrives; compatible servers: {1} only
            if qs1 > 0:
                qs1 -= 1
            else:
                qc1 += 1
        elif u < lam1 + lam2:
            # customer 2 arrives; compatible servers: {2}, plus {1} if MW
            s2 = qs2 > 0
            s1 = (qs1 > 0) and use_mw
            if s1 and s2:
                if qs2 >= qs1:          # tie -> server 2 (dedicated edge (2,2))
                    qs2 -= 1
                else:
                    qs1 -= 1
            elif s2:
                qs2 -= 1
            elif s1:
                qs1 -= 1
            else:
                qc2 += 1
        elif u < lam1 + lam2 + mu1:
            # server 1 arrives; compatible customers: {1}, plus {2} if MW
            c1 = qc1 > 0
            c2 = (qc2 > 0) and use_mw
            if c1 and c2:
                if qc1 >= qc2:          # tie -> customer 1 (dedicated edge (1,1))
                    qc1 -= 1
                else:
                    qc2 -= 1
            elif c1:
                qc1 -= 1
            elif c2:
                qc2 -= 1
            else:
                qs1 += 1
        elif u < lam1 + lam2 + mu1 + mu2:
            # server 2 arrives; compatible customers: {2} only
            if qc2 > 0:
                qc2 -= 1
            else:
                qs2 += 1
        # else: idle slot, no change

    L = (sum_revloss + S_HOLD * sum_hold) / cnt
    mean_hold = sum_hold / cnt
    return L, mean_hold


def simulate(eta, policy, n_steps, burn, seed):
    """policy = 'MW' or 'MMW'. Returns (profit_loss, mean_total_queue)."""
    sigma = eta ** (2.0 / 3.0) * N_TYPES ** (-1.0 / 3.0)     # sigma = eta^{2/3} n^{-1/3}
    c = 4.0 * eta                                            # uniformization constant
    use_mw = (policy == 'MW')
    return _simulate(float(eta), sigma, c, use_mw,
                     int(n_steps), int(burn), int(seed),
                     THETA, THETA, PHI, PHI)


# --------------------------------------------------------------------------- #
#  Experiment driver                                                          #
# --------------------------------------------------------------------------- #
def run_experiment(etas, n_steps=100_000_000, burn=15_000_000, n_seeds=3, seed0=12345):
    """Run each (eta, policy) configuration with `n_seeds` independent seeds and average.

    Returns (results, queues, details) where results[policy][i] is the mean profit loss
    over seeds for etas[i]; details[policy][i] = (mean_L, std_L, per_seed_list)."""
    results = {'MW': [], 'MMW': []}
    queues  = {'MW': [], 'MMW': []}
    details = {'MW': [], 'MMW': []}
    for pol_k, policy in enumerate(('MW', 'MMW')):
        for idx, eta in enumerate(etas):
            Ls, qs = [], []
            for s in range(n_seeds):
                # distinct seed per (policy, eta, seed-index)
                seed = seed0 + 100000 * pol_k + 1000 * idx + s
                L, mq = simulate(eta, policy, n_steps, burn, seed)
                Ls.append(L)
                qs.append(mq)
            meanL, stdL = float(np.mean(Ls)), float(np.std(Ls))
            results[policy].append(meanL)
            queues[policy].append(float(np.mean(qs)))
            details[policy].append((meanL, stdL, Ls))
            print(f"  {policy:3s}  eta={eta:6d}   L^eta = {meanL:10.4f} +/- {stdL:8.4f}"
                  f"   (n_seeds={n_seeds})   mean_queue = {np.mean(qs):8.3f}", flush=True)
    return results, queues, details


def fit_slope(etas, losses):
    """Least-squares slope of log(L) vs log(eta)."""
    x = np.log(np.asarray(etas, float))
    y = np.log(np.asarray(losses, float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


# font sizes for the figures
LABEL_FS  = 20     # axis labels
TICK_FS   = 16     # x/y tick labels
LEGEND_FS = 16     # legend text


def _linear_panel(ax, etas, mw, mmw):
    ax.plot(etas, mw,  'o-', color='tab:blue',   label='MW')
    ax.plot(etas, mmw, 's-', color='tab:orange', label='MMW')
    ax.set_xlabel(r'$\eta$', fontsize=LABEL_FS)
    ax.set_ylabel('profit loss', fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS)
    ax.grid(True, color='0.9')


def _loglog_panel(ax, etas, mw, mmw):
    lx = np.log(etas)
    s_mw, b_mw   = fit_slope(etas, mw)
    s_mmw, b_mmw = fit_slope(etas, mmw)
    ax.plot(lx, np.log(mw),  'o', color='tab:blue')
    ax.plot(lx, np.log(mmw), 's', color='tab:orange')
    ax.plot(lx, s_mw * lx + b_mw,   '-',  color='tab:blue',
            label=f'MW, slope={s_mw:.2f}')
    ax.plot(lx, s_mmw * lx + b_mmw, '--', color='tab:orange',
            label=f'MMW, slope={s_mmw:.2f}')
    ax.set_xlabel(r'$\log(\eta)$', fontsize=LABEL_FS)
    ax.set_ylabel('log(profit loss)', fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS)
    ax.grid(True, color='0.9')
    return s_mw, s_mmw


def make_figures(etas, results, out_dir='.'):
    """Write the two separate EPS panels (as referenced by the LaTeX source) plus a
    combined PNG preview.  Returns the fitted (MW, MMW) log-log slopes."""
    etas = np.asarray(etas, float)
    mw, mmw = np.asarray(results['MW']), np.asarray(results['MMW'])

    # --- separate EPS panel 1: linear (mw_vs_mmw.eps) ---
    fig1, ax1 = plt.subplots(figsize=(5.5, 4.3))
    _linear_panel(ax1, etas, mw, mmw)
    fig1.tight_layout()
    p1 = os.path.join(out_dir, 'mw_vs_mmw.eps')
    fig1.savefig(p1, format='eps', bbox_inches='tight')
    plt.close(fig1)

    # --- separate EPS panel 2: log-log (log_log_mw_vs_mmw_eta.eps) ---
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.3))
    s_mw, s_mmw = _loglog_panel(ax2, etas, mw, mmw)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, 'log_log_mw_vs_mmw_eta.eps')
    fig2.savefig(p2, format='eps', bbox_inches='tight')
    plt.close(fig2)

    # --- combined PNG preview ---
    figc, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))
    _linear_panel(axL, etas, mw, mmw)
    _loglog_panel(axR, etas, mw, mmw)
    figc.tight_layout()
    pc = os.path.join(out_dir, 'mw_vs_mmw.png')
    figc.savefig(pc, dpi=150, bbox_inches='tight')
    plt.close(figc)

    print(f"Saved:\n  {p1}\n  {p2}\n  {pc}")
    print(f"Fitted slopes:  MW = {s_mw:.3f} (paper 0.51)   "
          f"MMW = {s_mmw:.3f} (paper 0.37)")
    return s_mw, s_mmw


ETAS = [10, 100, 500, 1000, 2000, 5000, 10000]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Reproduce Figure 1 of the erratum to Varma et al. (2023): "
                    "profit loss of max-weight (MW) vs modified max-weight (MMW) "
                    "matching under the two-price policy.")
    p.add_argument('--steps', type=int, default=100_000_000,
                   help='simulation slots per (eta, policy) configuration')
    p.add_argument('--burn', type=int, default=15_000_000,
                   help='burn-in slots discarded before averaging')
    p.add_argument('--quick', action='store_true',
                   help='fast smoke test (8M steps) -- slopes are noisier')
    p.add_argument('--seeds', type=int, default=3,
                   help='number of independent seeds to average per configuration')
    p.add_argument('--out-dir', default='.', help='directory for the output figures')
    p.add_argument('--results', default='results.json',
                   help='cache file for the simulated profit losses')
    p.add_argument('--plot-only', action='store_true',
                   help='skip simulation and re-plot from the cached --results file')
    args = p.parse_args(argv)

    n_steps, burn = (8_000_000, 1_000_000) if args.quick else (args.steps, args.burn)
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, args.results)

    fv = verify_fluid()
    print("Fluid check (graph E={(1,1),(1,2),(2,2)}):")
    print(f"  fluid optimum  chi(11,22,12) = "
          f"({fv['opt_chi'][0]:.3f}, {fv['opt_chi'][1]:.3f}, {fv['opt_chi'][2]:.3f}),"
          f"  value = {fv['opt_val']:.4f}")
    print(f"  diagonal lambda*=mu*=1  value = {fv['diagonal_val']:.4f}"
          f"  ->  pi* used for profit loss = {fv['pi_star_used']:.4f}\n")

    if args.plot_only:
        with open(results_path) as fh:
            cached = json.load(fh)
        mean = cached.get('mean', cached)          # support old/new cache formats
        results = {k: [mean[k][str(e)] for e in ETAS] for k in ('MW', 'MMW')}
    else:
        print(f"Running (steps={n_steps:,}, burn={burn:,}, seeds={args.seeds}) ...")
        t0 = time.time()
        results, queues, details = run_experiment(ETAS, n_steps, burn, n_seeds=args.seeds)
        print(f"\nElapsed: {time.time() - t0:.1f} s")
        cache = {
            'n_seeds': args.seeds, 'steps': n_steps, 'burn': burn,
            'mean': {pol: {str(e): details[pol][i][0] for i, e in enumerate(ETAS)}
                     for pol in ('MW', 'MMW')},
            'std':  {pol: {str(e): details[pol][i][1] for i, e in enumerate(ETAS)}
                     for pol in ('MW', 'MMW')},
            'per_seed': {pol: {str(e): details[pol][i][2] for i, e in enumerate(ETAS)}
                         for pol in ('MW', 'MMW')},
        }
        with open(results_path, 'w') as fh:
            json.dump(cache, fh, indent=2)
        print(f"Wrote {results_path}")

    make_figures(ETAS, results, args.out_dir)


if __name__ == '__main__':
    main()
