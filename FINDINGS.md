# FINDINGS — Transfer Learning for PM2.5 at Newly Deployed Stations (Abu Dhabi)

A concise record of what this project established and, just as importantly, what it
**refuted**. Every number in Section 4 is produced by the `aq/` pipeline under the
protocol in Section 3 and reproduced by `tests/`. For the full methodology see
[SPEC.md](SPEC.md); to run the code see [README.md](README.md).

> Several intuitively appealing results were tested and did not survive scrutiny —
> Section 5 lists them, and the guard rails in the code (see Section 6) exist to stop
> them recurring. This is a negative-results / evaluation study: pooled transfer beats
> persistence on ordinary days but fails on the extreme-pollution days that matter.

---

## 1. Goal

Next-day PM2.5 forecasting at *genuinely new* monitoring stations — real 2023+
deployments, not stations held out from an existing record — evaluated under a
strictly leak-free protocol. The question is what transfer learning buys a station on
its first days of operation, and where it fails.

---

## 2. Data facts (verified against the file — earlier drafts had these wrong)

Source file: `Abu_Dhabi_stations_2_.xlsx`, 27 sheets (one per station).
Columns: `ts, Station_name, city, Longitude, Latitude, PM2.5`.

| Fact | Correct value | Wrong value in old drafts |
|---|---|---|
| Stations | 27 | — |
| Total rows | 28,806 | cited as sample size |
| **Non-null PM2.5 records** | **25,577** | 28,806 |
| After daily reindex + ≤3-day gap interpolation | 26,191 | — |
| Temporal resolution | **daily** | "hourly" (wrong, 3 places) |
| PM2.5 range | 0.62 – 491.75 µg/m³ | — |
| Median / mean | 34.5 / 39.5 µg/m³ | — |
| Date span | first workbook row 2017-02-10; first non-null PM2.5 2017-02-17 → 2024-11-07 | "2017-01-01" (wrong) |
| Lag-1 autocorrelation | ~0.67–0.74 | — |

**Do not use** the figure "10.06% improvement" from the old drafts. It was inconsistent
across documents (85.7% vs 92.9% success; 7.5% vs 10.06%; 15.8% vs 28.26% max), carried
±10.61% so its CI included zero, and included a 74-record station.

### Station sets

**Rich sources (9)** — start ≤2018, ≥1500 records:
`station_8, station_6, station_2, station_4, station_3, station_7, station_25, station_1, station_5`
(Baniyas School, Mussafah, Al Maqtaa, Khalifah School, US Embassy, Khadeeja School,
Hamdan Street, Khalifa City A, etc.)

**Authentic targets (12)** — first observation ≥2023-01-01 **and** ≥60 usable test days:
`station_12` (CITIES@Saadiyat), `station_13` (Lycee Francais), `station_14` (West Yas),
`station_15` (Khalifa City 1), `station_16` (Khalifa City 2), `station_17` (Sas Al Nakhl),
`station_18` (Al Danah), `station_19` (Shakhbout), `station_20` (Al Muzoon),
`station_21` (Bateen-BWA), `station_22` (Mamsha Azure 3), `station_27` (Sheleila)

**Excluded as targets:** `station_23` (Al Raha Gardens, 74 records — test set hits zero
by K=60) and `station_24` (Saadiyat Residence, 7 records — no test set at any K).
They remain usable as *spatial context* for other targets. Note station_23 produced the
−10.07% outlier in the old drafts; it was test-set noise, not a transfer failure.

**Median nearest-source distance: 5.3 km** (range 0.2–26.8). This tight clustering is the
mechanistic explanation for why source selection does not matter (Section 5.2).

---

## 3. The protocol (definitive — do not deviate)

**Authentic transfer (primary):** for each target station *j*
- **Sources:** all rich stations, data strictly `≤ (first observation of j − 1 day)`
- **Adaptation:** target's first *K* days
- **Test:** strictly after the adaptation window, requires **≥60 usable days**

**Simulated transfer (secondary):** target held out entirely from sources; cut 2023-01-01.

**Non-negotiable rules**
1. Splits by absolute date, never random.
2. Scalers fit on source training data only; never refit on test.
3. Lag/rolling features past-only (`closed='left'`).
4. Similarity metrics computed **only** on the adaptation window.
5. Calibration fit **only** on adaptation, applied to test.
6. **All methods scored on the IDENTICAL test set.** See Section 6.3 — this is how the
   headline result was lost.

### Structural constraint: the 14-day lag window
With `SEQ_L=14`, the first predictable day of a station's life is **day 15**. Therefore:
- **K < 14 yields ZERO usable adaptation sequences.** K=14 gives zero; K=30 gives ~16.
- A true "zero-shot cold start" is **impossible** in this model class.
- Fine-tuning and calibration experiments must use **K ≥ 30**.

---

## 4. VERIFIED findings

All from the 12 authentic targets unless noted. Wilcoxon signed-rank, two-sided.

### 4.1 Transfer works overall
Numbers below are **post gap-interpolation fix** and post-LSTM-sweep (see SPEC.md
§7.4 and §8.2). The pre-fix values (persistence 12.291, ridge 11.529, LSTM 12.081,
+1.7%, 10/12, p=0.027) are superseded; the LSTM figures use the representative
configuration h64_d0.2 (the sweep found all nine configs within 0.5 RMSE and ridge
beating every one — see the LSTM-sweep note below).

| Method (K=90) | median RMSE | vs persistence | beats persistence |
|---|---|---|---|
| persistence | 12.330 | — | — |
| **ridge pooled** | **11.589** | **+6.0%** | **12/12, p=0.0005** |
| LSTM+attention direct | 11.824 | +4.1% | **12/12, p=0.0005** |

**Ridge beats LSTM: 10/12 at both K=30 (p=0.0068) and K=90 (p=0.0034); pooled 20/24.**
The LSTM is a **genuine competitor, not a near-failure**: it beats persistence
significantly at both K (+3.1% at K=30, +4.1% at K=90, 12/12 each). It is simply
outperformed by a linear model — and that result survives a leak-free 9-config
hyperparameter sweep (ridge wins under every configuration; SPEC.md §8.2).

### 4.2 The four proposed enhancements do not work
Tested against the LSTM+attention backbone (`stage3_lstm.py`):

| Component | Result | p (vs direct transfer) |
|---|---|---|
| Similarity weighting (geographic) | no benefit | 0.52 / 0.57 |
| Full fine-tuning | no benefit | 0.85 |
| Head-only fine-tuning | **harmful** | 0.0015 |
| Linear calibration (α, β) | **harmful**, −21% at K=30 | 0.0010 |

Proposed mechanism for calibration failure: at K=30 only ~16 adaptation sequences exist,
so the α/β fit is unstable and extrapolates badly. **This is a hypothesis — the fitted
α/β values were never inspected.** State it as such, or measure it.

### 4.3 Extreme-event failure (the central finding)
Extreme events are real and spiky: **6.4% of station-days ≥75 µg/m³**; **56% of episodes
last exactly one day**; max 491.8 µg/m³.

| Slice | persistence | ridge | gain |
|---|---|---|---|
| overall | 12.36 | 11.65 | **+5.8%** |
| normal days | 11.81 | 10.57 | **+10.5%** |
| extreme (≥75) | 26.85 | 32.35 | **−20.4%** |
| top 10% | 19.14 | 22.09 | −15.4% |
| onset days | 33.74 | 36.32 | −7.6% |

**The entire aggregate gain comes from normal days. On days that matter for health the
model is substantially worse than persistence.** Replicated independently in the
common-test-set analysis (Section 5.5): −7.4% on extremes.

Event warning skill (top-10% exceedance): ridge F1 **0.34** (recall 0.28) vs
persistence F1 **0.46** (recall 0.46). The model misses ~72% of events.

---

## 5. REFUTED — do not rebuild on these

### 5.1 Cold-start advantage curve
Died on the 14-day lag window (Section 3). K=0/7/14 are structurally identical.

### 5.2 Source selection / similarity-based source choice
Tested twice, both null. Best *oracle* single source beats pooling by only ~1%.
Deployable variants vs uniform pooling: geo-weighted p=0.27, top-5 p=0.52, top-3 p=0.23.
**Random-5 sources performs the same as geo-selected-5.** Geographic distance does predict
*individual* source quality (Spearman ρ≈0.44–0.49, p<0.05) but the signal is erased once
≥3 sources are pooled. Distributional (KS) similarity: **ρ=0.05, p=0.98 — no signal at all.**

### 5.3 "k=1 nearest source beats pooling all" — FALSE
An external ablation claimed k=1 (12.478) beat all-uniform (12.602), p=0.0425.
Under the leak-free protocol this **reverses**: k=1 = 11.749, which is 0.8% *worse* than
pooling all, winning at only 25% of targets (p=0.0522 trending the other way).
Cause: that script applied **no `train_end` cutoff**, so source training data spanned the
target's test period; it also scored persistence on n−1 points and models on n.

### 5.4 Distance-decay weighting of training sources
Null: τ=10 → +0.36% (p=0.42), τ=25 → +0.19% (p=0.27), τ=50 → +0.10% (p=0.15).
Also note: **at k=1 decay weighting is mathematically a no-op** (a single weight normalizes
away), so identical RMSE across τ values is expected, not evidence.

### 5.5 ⚠️ "Spatial neighbour features fix extreme events (+35.6%)" — ARTIFACT
This was the project's headline positive result and it **does not survive**.

The Stage 4 spatial design skips any day where a neighbour reading is missing, so the
spatial model was scored on ~140 days while the temporal model was scored on ~331 — and
the complete-coverage subset is *easier* (persistence scores 12.97 there vs 19.14 on the
full set). The apparent +35.6% was a difference between day-subsets, not a treatment effect.

On a **common test set** (`stage5b_common_ablation.csv`, 10 targets, ~140 shared days):

| k neighbour features | RMSE all | vs k=0 | RMSE extreme | vs k=0 | p |
|---|---|---|---|---|---|
| 0 | 9.673 | — | 13.871 | — | — |
| 1 | 9.806 | −1.4% | 14.218 | −2.5% | 0.32 |
| 5 | 9.898 | −2.3% | 13.692 | +1.3% | 0.16 |
| 8 | 9.893 | −2.3% | 13.551 | +2.3% | 0.43 |

**Corrected conclusion:** spatial features give ~+2.3% on extremes (null, p=0.43) and
slightly *hurt* overall (−1.4 to −2.3%). Best configuration is still **4.9% worse than
persistence** on extreme days.

---

## 6. Bugs found (fix these anywhere they appear)

### 6.1 LSTM target-scaling bug
Training on raw PM2.5 (mean ≈39) with standardized inputs forces the output layer to
converge on a large bias; with early stopping it badly underfits. Symptom: LSTM RMSE ≈15.4
versus ridge ≈11.5. **Fix:** standardize `y` during training, inverse-transform at predict,
and carry `(mu, sd)` through fine-tuning. After the fix the LSTM reached ≈11.4–12.1.

### 6.2 Seasonal routing bug (in all original scripts)
The original `SeasonalBackboneWrapper.predict()` ran both seasonal models on the full test
set and returned whichever produced more predictions — it **never routed by season**.
**Fix (`aq.models.SeasonalRouter`):** each prediction is taken from the model whose season
matches that day, with a documented fallback when one seasonal model has no output.
Covered by `tests/test_invariants.py`.

### 6.3 ⚠️ Test-set alignment (the most damaging class of error)
Three separate instances found in this project:
- Stage 4 spatial vs temporal: 140 vs 331 days → produced the false +36% (Section 5.5).
- External ablation: persistence on 507 points, models on 508.
- Any design that drops rows on missing data changes the denominator.

**Rule: build one common evaluation index first, then score every method on exactly it.**

---

## 7. Honest framing for the paper

> Pooled multi-source transfer beats persistence at all 12 newly deployed PM2.5 stations
> (+6.0%, p=0.0005) with no local history required. However, this aggregate gain arises
> entirely from ordinary days: on extreme-pollution days the same models are 15–20% *worse*
> than persistence and miss ~72% of exceedance events. Neither neural architecture,
> adaptation data, similarity-based source selection, output calibration, nor spatial
> neighbour features resolves this failure. Aggregate RMSE therefore systematically
> conceals extreme-event degradation in transfer-learning evaluations for air quality.

**Caveats to state explicitly:** n=12 targets (underpowered below ~1% effects — say
"no evidence of benefit," not "proof of no benefit"); single city; single pollutant;
no meteorological covariates; next-day horizon only.

---

