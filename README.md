# Erratum Figure 1 Simulation

This repository contains code to reproduce the numerical experiment in Figure 1 of the erratum for **“Dynamic Pricing and Matching for Two-Sided Queues.”**

The script compares:

- **MW**: max-weight matching using all compatibility edges.
- **MMW**: modified max-weight matching using only non-redundant edges.

The simulation uses the corrected non-CRP example

\[
E=\{(1,1),(1,2),(2,2)\}, \qquad E_r=\{(1,2)\},
\]

so that the modified max-weight policy uses only the diagonal edges \((1,1)\) and \((2,2)\). Ties under MW are broken in favor of the diagonal edges.

## Model parameters

The example uses:

\[
n=m=2, \qquad s=1, \qquad \tau^\eta_{\max}=0,
\]

\[
\sigma^\eta=\eta^{2/3}n^{-1/3},
\]

with inverse demand curves

\[
F_1(\lambda_1)=5-\lambda_1, \qquad F_2(\lambda_2)=4-\lambda_2,
\]

and inverse supply curves

\[
G_1(\mu_1)=1.5\mu_1, \qquad G_2(\mu_2)=\mu_2.
\]

For this corrected graph, the fluid optimum is

\[
\lambda^*=\mu^*=\mathbf 1_2, \qquad \chi^*_{11}=\chi^*_{22}=1, \qquad \chi^*_{12}=0,
\]

and the redundant edge is \((1,2)\). The fluid objective value is

\[
\gamma^*=4.5.
\]

## Requirements

Use Python 3.10 or newer. Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

`numba` is optional in principle, but strongly recommended. Without it, the long paper-quality simulations will be slow.

## Usage

### Quick smoke test

Use this to check that the script runs:

```bash
python simulate_erratum_fig1.py --preset quick --outdir results/quick
```

### Paper-quality run

Use this to generate the manuscript-style figure:

```bash
python simulate_erratum_fig1.py --preset paper --outdir results/fig1
```

The paper preset uses 10% burn-in, meaning

\[
\text{burn-in steps}=0.10\times \text{measured steps}.
\]

The preset uses longer runs at \(\eta=10000\), especially for MMW, because the MMW chain mixes slowly at large \(\eta\).

## Output files

The script writes the following files to the output directory:

```text
erratum_fig1.png
erratum_fig1.pdf
erratum_fig1_loss_vs_eta.png
erratum_fig1_loss_vs_eta.pdf
erratum_fig1_loglog.png
erratum_fig1_loglog.pdf
erratum_fig1_results.csv
erratum_fig1_slopes.txt
```

The combined figure `erratum_fig1.png` / `erratum_fig1.pdf` contains two panels:

1. Profit loss versus \(\eta\).
2. Log-log plot of profit loss versus \(\eta\), with fitted slopes.

The CSV file stores the simulated mean profit, profit loss, standard deviation, standard error, number of measured steps, burn-in steps, number of replications, and elapsed runtime for every \((\eta,\text{policy})\) pair.

## Replotting from an existing CSV

To regenerate plots without rerunning simulations:

```bash
python simulate_erratum_fig1.py --plot-only --csv results/fig1/erratum_fig1_results.csv --outdir results/fig1
```

## Custom runs

You can override the preset settings. For example:

```bash
python simulate_erratum_fig1.py \
  --preset quick \
  --steps 1000000 \
  --reps 5 \
  --burn-frac 0.10 \
  --outdir results/custom
```

Useful options:

```text
--steps                 measured steps for all eta values except eta=10000 override
--reps                  replications for all eta values except eta=10000 override
--burn-frac             burn-in fraction relative to measured steps
--eta10000-steps        measured steps for eta=10000
--eta10000-reps-mw      MW replications for eta=10000
--eta10000-reps-mmw     MMW replications for eta=10000
--seed                  base random seed
```

## Reproducibility notes

- The script uses a deterministic base seed by default.
- The paper preset is intentionally long to reduce Monte Carlo noise at large \(\eta\).
- The figure does not include the exact MMW diagnostic line; it only plots MW and MMW, matching the manuscript-style figure.
