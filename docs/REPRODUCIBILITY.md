# Scientific reproduction status and implementation decisions

This repository is a reference implementation based on the supplied manuscript. It contains executable experiments, not reconstructed or fabricated paper measurements. Exact original checkpoints, split IDs, dataset preprocessing manifests, radio traces, energy traces, and board/phone code were unavailable. The scripts therefore reproduce the **declared reference protocol**, and do not certify the paper's numerical results or its theoretical claims.

## Definitions that affect results

1. **Deadline origin.** A 2 s acquisition window cannot meet a 100–300 ms deadline measured from its beginning. Here, latency starts when the window is complete. Add acquisition duration (and any streaming preprocessing delay) when reporting latency from window start. The figures and software metrics are post-window latency.
2. **Model shape.** The literal CNN is a 1→16 stem with a 3×5 kernel, then 16→32→64→96 depthwise/pointwise blocks, temporal strides 1, 2, 2, global average pooling, and a 96→128 projection. Inputs are `[batch,1,sensors,time]`. The stated approximate parameter/MAC totals in the manuscript are not forced into the implementation. `training.json` reports measured counts for the actual input and class count. Auxiliary heads affect counts.
3. **Data choice.** WISDM UCI 507 phone acceleration is the explicit default; PAMAP2 Protocol uses the hand; RealWorld uses paired chest sensors. Native label vocabularies are retained unless an explicit map is supplied. Sensor selection and taxonomy are not interchangeable with an unspecified original preprocessing pipeline.
4. **Train/validation/calibration/test.** Subjects are disjoint; overlap occurs only among windows belonging to the same split. Validation comes from the nominal training pool. Normalization and representative TFLite conversion samples use training only. Validation fits temperature, early stopping, and transmission policy. Final calibration labels are used only to calibrate final prediction sets. No test labels enter model fitting, policy decisions, debt, or threshold selection.
5. **Packet count.** `k` counts distinct useful application packets received by the inference deadline. Each packet carries `m` sparse projection equations. The actual measurement matrix has `m*k` rows, not `k` rows. Unknown coordinates are zero-filled. Seeds, packet IDs, and window IDs are checked.
6. **Rank limit.** With 128 features, `m=2`, and `k<=6`, rank is at most 12. These data cannot identify an arbitrary 128-dimensional vector. Peeling may reveal individual coordinates but is not full vector recovery. Support-union size is not a rank lower bound. `scripts/check_theory.py` illustrates this explicitly.
7. **Integer endpoints.** int8 includes −128. Projection range checks use magnitude 128 and a degree cap no larger than 255, so the signed int16 projection remains representable. A singleton value of 0, −1, or −128 is legitimate and is never discarded merely because its magnitude is small. The default residual tolerance of 2 applies to fully resolved consistency checks; malformed/out-of-range singletons are rejected transactionally.
8. **Wire format.** The versioned little-endian format in `PROTOCOL.md` is a concrete reference choice; the manuscript did not provide a complete interoperable byte layout. Each projection has its own degree and support. C and Python parity is tested. CRCs detect accidental corruption; they are not authentication. Sparse projections are not encryption, and degree-one equations reveal individual coordinates.

## Scores and conformal calibration

The default class-conditional nonconformity score is

\[
s_c(p) = \frac{1 + \max_{j\ne c}p_j - p_c}{2}.
\]

Q15 probabilities sum exactly to 32768. The score is rounded upward to Q15. A label-independent top-two uncertainty value alone does not specify a multiclass conformal set. It is used for the recent uncertainty sketch, while calibration uses each candidate label's score above.

Score sensitivities are explicit reference extensions: `probability` is `1-p_c`; `entropy` is `(H(p)/log(C)+1-p_c)/2`; `energy` is `sigmoid(-logit_c)` with temperature-scaled logits. These two latter extensions use floating point and are not advertised as the integer-only default. `direct_margin` uses the learned auxiliary margin's sigmoid for the winning class and its complement convention in `integer.py`. The auxiliary margin is trained by Smooth L1 against the full-logit top-two gap; it is not an original unpublished target inferred from the paper.

The 16-byte recent-uncertainty histogram travels in packets and is retained by the decoder. The manuscript does not uniquely specify how to fuse it with candidate-label phone scores. No unprovided fusion rule is invented here: the default calibrator uses the reconstructed-feature class scores. At zero received packets the phone has no current sketch or current local logits; it evaluates the zero feature vector. Local baselines have the full local feature by definition.

Histogram calibration uses unsigned 64-bit counts and the finite-sample rank `ceil((n+1)*(1-alpha))`. Its upper-bin threshold is conservative relative to the exact empirical order statistic. If the rank exceeds the sample size, or a group is empty, the threshold is the full score range, giving the full label set. Quantile rounding never removes this correction. The pooled ablation uses one realized score per calibration window; it does not inflate sample size with correlated prefixes.

Validation is divided into a pilot calibration half and a policy-selection half. The pilot defines per-prefix thresholds and selects the smallest `k_star` meeting both empirical coverage and set-size targets. If no prefix qualifies, `k_star` is `null`; the controller marks the target infeasible and attempts the maximum supported evidence subject to caps. It never claims the target was met merely because a budget was clipped.

Final held-out calibration replays the **same frozen policy** used for test, then calibrates at each window's realized packet count. Debt uses the previous window's pilot-set size, not final calibration thresholds or test labels. This avoids changing the policy after learning the thresholds for that policy. Policy selection itself is empirical, not a certificate of coverage or feasibility.

Ordinary split-conformal validity requires exchangeability. Held-out subjects, placement shift, serially overlapping windows, and adaptive history can violate relevant assumptions, including within packet-count groups. This code reports empirical coverage and group sample counts; it does not promise distribution-free conditional coverage under arbitrary shift or adaptive control. `AdaptiveHistogram` is an optional integer EWMA research primitive; it is not enabled in the main experiments and carries no finite-sample guarantee.

## Controller and transport

For independent Bernoulli packet arrivals with success probability `q`, the implemented Chernoff sizing uses

\[
P(S_n\ge k)\ge 1-\exp\!\left(-\frac{(nq-k)^2}{2nq}\right),\quad nq>k.
\]

Writing `a=log(1/beta)` gives a sufficient offered count

\[
n=\left\lceil\frac{k+a+\sqrt{a^2+2ka}}{q}\right\rceil.
\]

For `k=2,q=0.9,beta=0.05`, this requires 11 offered transmissions. It is incorrect to substitute an ACK-truncated average count such as 2.4 into this bound. The code records the offered budget and actual emitted count separately. The integer sizing helper rounds conservatively. Availability, energy, battery, and maximum-emission caps can make the request infeasible; metadata records that fact.

The default debt is a causal EWMA of previous pilot-set-size violations. A delayed true-label violation can be supplied to the primitive only if that label is actually available; benchmark evaluation does not do so. Coefficients and thresholds are in the configuration.

Link settings are declared assumptions: good/moderate/poor use 80/60/40 **attempt opportunities per second** and success probabilities .98/.90/.75. This is not already-loss-adjusted goodput. Bursty mode uses Gilbert–Elliott transitions .08 good→bad and .25 bad→good, with state success probabilities .98 and .25. The configured nominal .90 used by its controller is an intentionally imperfect estimate; the independent Bernoulli bound is only a sizing proxy for this correlated channel, not a bursty-channel guarantee.

Methods sharing a seed receive a common stream of potential link opportunities. ACKs stop future emissions, while transmissions that started before the ACK still consume airtime and energy. A packet is useful only if its arrival plus processing delay meets the deadline. Simulation operates on application frames; it does not implement BLE fragmentation, scheduling, pairing, reconnection, or operating-system queues. Larger frames consume proportionally more service time under the declared fixed per-frame loss model. Trace replay refuses missing windows, insufficient opportunities, and mismatched frame sizes.

Energy values and processing latencies in configuration files are a **cost model**, not hardware measurements. Phone-only and edge-assisted variants change these timing assumptions; they do not launch an Android app or edge server. Full-offload is a full feature vector, not a raw 2 s sensor upload. Fixed-chunk sends an importance-ranked fixed subset once. MC dropout uses 20 stochastic passes through a separately trained dropout classification head; it is not 20 reruns of a stochastic backbone.

## Mathematical limitations

A Lipschitz bound on absolute score error does not establish monotonic signed score improvement as reconstruction error shrinks. For the identity score at ideal point 0, errors −2→−1 have decreasing norms but scores −2→−1 increase. The runner does not artificially force calibration thresholds, set sizes, or coverage curves to be monotone. Results may reveal failure modes and infeasible settings.

Seed-level confidence intervals use Student's t over independent seed summaries. Overlapping windows are not treated as independent replicates for these intervals. With only three seeds, intervals can be wide. Paired method comparisons align the same scenario and seed. Placement AUROC is omitted/null when one placement group is absent.

## What remains necessary for exact reproduction

Supply the original preprocessing and taxonomy, subject IDs, initialization/checkpoints, calibration data, full model specification, fusion/decoder/controller details where they differ, per-condition packet and current traces, and target hardware/firmware. Reconcile the deadline origin, rank interpretation, and monotonicity claim before calling this an exact reproduction of the paper. The provided implementation can serve as a transparent starting artifact for that reconciliation.
