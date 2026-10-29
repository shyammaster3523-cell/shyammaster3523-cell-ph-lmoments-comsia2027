# L-Moments for Phase-Type Distributions: Kronecker Algebraic Representation

Code and results for the paper by C. Shyam Sunder, Dibu A.S. and Sathya P.
(Department of Statistics and Data Science, CHRIST (Deemed to be University), Bengaluru),
*ComSIA 2027*, Springer Lecture Notes in Networks and Systems.

The code computes the L-moments of any phase-type (PH) distribution exactly, using the Kronecker
representation of the probability weighted moments (Theorem 1). It fits PH models by L-moment
matching, classical moment matching and EM, and reproduces every table and figure in the paper.

## Setup
```bash
pip install -r requirements.txt
python download_data.py          # downloads AutoClaims and danishuni (public CRAN data) into ./data (created automatically)
```
Python 3.12 was used (NumPy 2.4, SciPy 1.17, mpmath 1.3). Scripts can be run from any folder.
All files sit in one folder. Results (`*.json`, `contam.jsonl`, `ratio_pts.npy`) and figures (`fig*.pdf`, `fig*.eps`)
are read and written there; the result files reported in the paper are included.

## Core library
| File | Contents |
|---|---|
| `phlm.py` | PWMs and L-moments of PH laws: dense route (Theorem 1), spectral route (Corollary 1), sample L-moments, CF-1 form |
| `fitlib.py` | sparse triangular route (Corollary 2), automatic route choice, CF-1 L-moment and classical fitting (L-BFGS-B, multi-start), EM for general PH, goodness of fit (KS, AD, upper-tail AD), data loading |

## Reproducing the paper
| Paper item | Command |
|---|---|
| Validation of Theorem 1 (exponential, Coxian-2 example, quadrature) | `python check.py` |
| Fig. 1 and operation counts (Sect. 4) | `python exp_timing.py`, `python dense_big.py`, `python run_tri_timing.py` |
| Numerical stability, 40-digit reference (Sect. 4) | `python exp_stab.py` |
| Jacobian and singular sets (Sect. 5) | `python exp_jac.py`, `python exp_jac2.py` |
| Proposition 1: exact symbolic proof and 50-digit check | `python prop3.py` |
| Simulated PH laws (Sect. 6.1) | `python run_synth.py` |
| Table 1, contamination (Sect. 6.2; about 12 min) | `python exp_contam2.py 200 0 16` |
| L-moment and classical fits (Sect. 6.4–6.6) | `python run_fit.py <auto\|danish\|danishx> <L\|C> <m> <starts>`, e.g. `python run_fit.py auto L 3 20` |
| Order-4 nested warm start | `python run_nested.py auto L 4 0 4` |
| EM from different starts (Sect. 6.4) | `python run_em.py auto 3 Lmom Cmom quant rand0 rand1 ... rand9` |
| Table 2, Danish severity models (Sect. 6.5) | `python run_danish_tail2.py` |
| Table 3, consolidated comparison (Sect. 6.6) | `python summarize.py` |
| Figs. 2 and 3 | `python figs3.py` |

Dataset keys: `auto` = AutoClaims (`PAID`, n = 6,773); `danish` = Danish fire losses (`danishuni`,
n = 2,167, in MDKK); `danishx` = Danish excesses over the 1 MDKK reporting threshold (x − 1).

Run times in the paper were measured on one core of an Intel Xeon at 2.1 GHz. EM runs from random
starts take up to about a minute each, and `exp_contam2.py` accepts a cell range, so the grid can be
run in pieces (`python exp_contam2.py 200 0 5`, then `5 10`, then `10 16`).

## Data sources
- AutoClaims: R package `insuranceData`, https://CRAN.R-project.org/package=insuranceData
- Danish fire losses: `danishuni` in the R package `fitdistrplus`, https://CRAN.R-project.org/package=fitdistrplus
