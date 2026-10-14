# PGNN for binary state-of-charge classification from EIS

Code, data splits, and results for:

> Raju, A. P., Bandi, R., Jayaprakash, G. K., Ibrahim, A. B. M., and Naik, P.
> *A Physics-Guided Regularised Neural Network for Binary State-of-Charge Classification of
> Lithium-Ion Batteries Using Electrochemical Impedance Spectroscopy.* Manuscript under review,
> Frontiers in Chemistry.

The model is a feed-forward network (15-128-128-64-2, tanh, dropout 0.2) trained on
sliding-window EIS features (log10 f, Re(Z), Im(Z), |Z|, phase at three neighbouring frequencies).
Its loss adds a soft, Randles-motivated penalty to cross-entropy:

```
L_total = L_CE + λ · L_phys,    L_phys = mean[ p1 · z_norm · 1{z_norm > 0.5} ],    λ = 1.0
```

where `p1` is the predicted High-SoC probability and `z_norm` the training-set min-max-scaled
|Z| of the window's centre point. It is a physics-guided regulariser, not a PINN: no governing
equation is enforced, and the monotonic prior holds only below about 50% SoC (manuscript Section 4.5).

## Repository layout

```
notebooks/
  01_main_experiments.ipynb        sweep-grouped split, ablations, test-set λ sweep, activation
                                   counts, |Z| and R_ct proxy vs SoC, spectrum-level evaluation
  02_followup_experiments.ipynb    single run, ten-seed baselines and bootstrap CIs, A1 variants,
                                   consistency metric, tuned baselines, cutoffs, Randles fits,
                                   validation λ sweep
  03_auc_comparison.ipynb          ten-seed AUC-ROC comparison with RF and GB
  04_robustness_experiments.ipynb  matched control penalties, cell-disjoint split, ten-seed
                                   leave-one-cell-out, split audit and index export, Figures 4/5/7
splits/                            exact partition indices (Supplementary Files S1 to S3)
results/                           JSON outputs of notebook 04
figures/                           manuscript figures; figure3_architecture.py redraws Figure 3
data/                              place impedance.csv and frequencies.csv here (not included)
```

All notebooks are saved **with their outputs**; the printed numbers are the ones reported in the paper.

## Where each result comes from

| Manuscript item | Notebook, section |
|---|---|
| Table 1, Section 2.1 partition composition | 02 §2; 04 §2 |
| Table 3 PGNN row, Table 4, Figure 5A | 02, exact-ROC cell after §11; 04 §6 |
| Table 3 RF and GB rows (exact-ROC thresholds), Figure 4 | 04 §6 |
| Table 5 (multi-seed and tuned baselines, bootstrap CIs) | 02 §4, §7; 03 |
| Table 6, Figure 6 (ablations A2 to A4) | 01 §4 |
| Table 6 row A1 (matched-position) | 02 §5 |
| Table 7, Figure 5B (validation λ sweep) | 02, last cell |
| Section 4.3 test-set λ values | 01 §5 |
| Figure 7 | 04 §6 |
| Section 2.3.2 activation counts | 01 §6 |
| Figures 8A and 8B | 01 §7 |
| Figure 8C, fitted R_ct statistics | 02 §10 |
| Section 4.5 physical-consistency metric | 02 §6 |
| Table 8 (control penalties) | 04 §3 |
| Section 4.6 cutoff sensitivity | 02 §9 |
| Section 4.7 spectrum-level evaluation | 01 §8 |
| Section 4.8 leave-one-cell-out (10 seeds) and cell-disjoint split | 04 §5, §4 |
| Figure 3 | `figures/figure3_architecture.py` |

Figure 1 is a drawn schematic. Figure 2 (Nyquist plots) is included as an image only.

## Data

The dataset is not redistributed here. Please cite it as:

> Buchicchio, E., De Angelis, A., Santoni, F., and Carbone, P. (2022). Dataset on broadband
> Electrochemical Impedance Spectroscopy of Lithium-Ion Batteries for Different Values of the State
> of Charge. Mendeley Data, V3. doi: [10.17632/mbv3bx847g.3](https://doi.org/10.17632/mbv3bx847g.3)

It is described in Buchicchio et al. (2022), *Data in Brief* 45, 108589,
doi: [10.1016/j.dib.2022.108589](https://doi.org/10.1016/j.dib.2022.108589). The files used here were
obtained from the Kaggle mirror at <https://www.kaggle.com/datasets/orvile/eis-of-lithium-ion-batteries>.
Place two files in `data/`:

| File | Columns |
|---|---|
| `impedance.csv` | `MEASURE_ID, SOC, BATTERY_ID, FREQUENCY_ID, IMPEDANCE_VALUE` (complex, e.g. `(0.1109-0.0054j)`) |
| `frequencies.csv` | `FREQUENCY_ID, FREQUENCY_VALUE` (Hz) |

Expected: 3,360 rows, 4 batteries (2, 3, 5, 6), 240 sweeps of 14 frequencies. Notebook 04 checks
these and stops if the impedance column is not parsed correctly.

## Running

Colab: open a notebook, Runtime → Run all, and upload the two CSV files when prompted. Locally:

```bash
pip install -r requirements.txt
jupyter lab   # then run the notebooks with the CSVs in data/
```

Runtimes on a CPU: 01 about 25 min, 02 about 55 min, 03 about 3 min, 04 about 50 min.
Notebook 04 draws Figures 5B and 6 from earlier results if the result zips written by notebooks
01 and 02 are uploaded with the data; otherwise Figure 5B is recomputed and Figure 6 is skipped.

## Conventions

* **Splits.** Windows are concatenated sweep by sweep in file order (battery, then `MEASURE_ID`,
  then SoC from 100% to 10%). The principal split cuts at rows 2,016 and 2,448 of the 2,880 windows,
  which fall on sweep boundaries. `splits/` lists every window's partition for the principal,
  cell-disjoint (train 2+3, validate 5, test 6), and leave-one-cell-out splits.
* **Scaling.** Feature standardisation and the `z_norm` min-max scaling use training data only;
  `z_norm` is clipped to [0, 1].
* **Threshold.** Youden's J on the validation set. Tables 3, 4, 6, 7, and notebook 04 use exact ROC
  thresholds; Table 5, the tuned baselines, cutoffs, and consistency metric use a 500-point grid.
* **Seeds.** 42 for single runs; 0 to 9 for multi-seed results; 0 to 7 for the λ sweep.
* **Standard deviations.** Notebooks 02 and 04 use the sample SD (n − 1); notebook 01 uses the
  population SD (n).

## Environments

Notebooks 02, 03, and 04 were run with Python 3.13.16, PyTorch 2.11.0, scikit-learn 1.6.1,
NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, matplotlib 3.10.0 (`requirements.txt`). Notebook 01 was run
with Python 3.10, scikit-learn 1.8.0, and NumPy 2.4.2. Results can differ in the third or fourth
decimal across environments.

Only two code changes were made after the runs, neither affecting results: the input paths now
read `data/impedance.csv` and `data/frequencies.csv`, and sweep identifiers are split robustly
when `MEASURE_ID` itself contains an underscore.

## License and citation

Code is released under the MIT License (`LICENSE`). If you use it, please cite the paper
(`CITATION.cff`).
