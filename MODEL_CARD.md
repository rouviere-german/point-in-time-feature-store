# Model card: churn-lightgbm (leakage experiment)

Format based on Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). This model exists
to measure label leakage, not to predict real churn: it is trained on synthetic data, once with
point-in-time features and once with leaking joins, and the gap between offline and holdout AUC is
the quantity of interest. Numbers are copied from `bench/bench_baseline.py` output.

## Model details

- **Name:** `churn-lightgbm`, three arms per seed: `pit`, `naive_latest`, `event_time_asof`
- **Type:** LightGBM binary classifier (`objective=binary`, 120 rounds, learning rate 0.05,
  15 leaves, `min_data_in_leaf=40`, feature fraction 0.9, bagging 0.8, `deterministic=true`,
  2 threads)
- **Owner:** Rouviere German
- **Date:** 2026-09-13
- **License:** MIT (code). Data is synthetic and generated at run time.
- **Artifact format:** LightGBM text model (`model_to_string`), logged to MLflow as
  `models/model_<arm>.txt`; never pickle.
- **Lineage:** MLflow file-store run (params: seed, generator seed, customer count, every feature
  view version hash, per-source and combined Parquet digests, model SHA-256 per arm; metrics:
  offline AUC, holdout AUC and gap per arm; artifacts: models and a run manifest with package
  versions and dataset fingerprints) and the registry's `training_runs` table.

## Intended use

- **Primary use:** demonstrate and regression-test that point-in-time retrieval removes label
  leakage (CI fails if the mean point-in-time gap exceeds 0.01 or the naive gap drops below 0.05).
- **Intended users:** reviewers of this repository and engineers evaluating feature-store
  correctness.
- **Out-of-scope uses:** any decision about real customers. The data, base rates and signal
  strength are invented.

## Training data

- **Source:** `offline/synthetic.py` (`SyntheticConfig`, seed per run). Customers in steady state;
  churn hazard depends on a latent engagement; activity declines in the 45 days before churn.
- **Sources and defects:** CRM profiles (daily export lag, cancellation rows ingested 0.5 to
  3 days late), transactions (8% late by days, 1% by weeks, 2% duplicate deliveries, 0.5%
  restatements), sessions (zero ingestion lag, 1% exact duplicates).
- **Contract:** `ml/contracts.py` pandera schemas per source; the generator's output passes them.
- **Label spine:** every 14 days, active customers only; label = churn within 30 days.
- **Features:** plan tier, monthly fee, support tickets (90 days), cancellation flag, last payment
  amount, 30-day payment count, large-payment flag, last session minutes, 7-day session count,
  engaged flag, and the age of each view's value at label time.
- **Known gaps:** no seasonality, no pricing changes, no customer re-activation.

## Evaluation

- **Split:** customers are split by a hash of their id: 60% train, 20% validation, 20% holdout.
- **Offline AUC:** train customers' labels between day 60 and day 240 for fitting, validation
  customers' labels in the same period for scoring, both with the arm's join.
- **Holdout AUC:** holdout customers' labels between day 280 and day 380 with point-in-time
  features, which is what serving would have returned at those times.
- **Arms:** `naive_latest` attaches each customer's latest row from an extract taken on day 275;
  `event_time_asof` joins on event time only; `pit` uses the store's retrieval.
- **Results:**

  From `uv run python bench/bench_baseline.py --check` (benchmark `leakage_5_seeds`), 5 seeds
  (generator seeds 100 to 104, model seeds 0 to 4), 10,000 generated customers per seed, mean
  holdout size 7,064 label rows. Gap = offline AUC - holdout AUC. Run on 2026-09-13, Windows 10,
  6 logical CPUs, Python 3.12.14; two consecutive runs printed identical values.

  | Arm | Offline AUC (mean) | Holdout AUC (mean) | Gap (mean) | Gap per seed |
  |---|---:|---:|---:|---|
  | `naive_latest` | 1.0 | 0.4859 | 0.5141 | 0.6595, 0.4355, 0.4914, 0.4895, 0.4946 |
  | `event_time_asof` | 0.8072 | 0.8089 | -0.0017 | -0.0001, -0.0024, -0.0010, -0.0026, -0.0024 |
  | `pit` | 0.8001 | 0.8080 | -0.0079 | -0.0058, -0.0144, -0.0059, -0.0069, -0.0064 |

  The naive model reaches a perfect offline AUC because the cancellation row is in the extract,
  and scores worse than chance on the holdout, where no active customer is cancelled yet. The
  point-in-time gap is negative for every seed: the holdout period scores slightly better than
  the validation period. The retrieval is oracle-checked, so this is not leakage; it is most
  likely a small difference between the generated periods, which was not investigated further.
- **Slices:** not evaluated (synthetic data, no protected attributes).

## Robustness, drift and monitoring

- **Leakage robustness:** the retrieval used for `pit` is property-tested against an oracle on
  adversarial histories (`tests/unit/test_pit_properties.py`), and the mutation test shows the
  property rejects the leaking joins used by the other arms.
- **Determinism:** identical seeds give identical AUCs and model hashes
  (`tests/integration/test_training.py::test_training_is_deterministic_for_a_seed`).
- **Drift monitors:** PSI (threshold 0.2) and KS (alpha 0.01, minimum statistic 0.05) on numeric
  retrieval columns between the training and holdout periods, exported as `feature_psi`
  (`monitoring.py`); the demo reports the maximum PSI.
- **Serving parity:** the skew checker recomputes logged online reads offline and exports
  `parity_mismatch_ratio`.

## Ethical considerations

The data is synthetic; no personal data is used. The e-mail column exists only to exercise PII
controls (PII tags, `pii:read` scope, HMAC in the request log, erasure). The model is not trained
on it. A real churn model built on such features can steer retention offers unequally; that would
need its own fairness evaluation, which is out of scope here.

## Caveats and recommendations

- The size of the naive gap depends on how strongly the leaked signal (the cancellation row)
  predicts the label; real leaks are usually subtler and harder to see without an oracle.
- The `event_time_asof` gap is small on this data because most lag is hours, not days; data with
  longer or label-correlated ingestion lag would show a larger gap.
- Validation and holdout AUCs are estimated on a few thousand rows per seed, so single-seed gaps
  vary by about a percentage point; the benchmark averages five seeds.
