# MTS DP Conference Houston 2026 — Abstract

**Submission target:** MTS DP Conference Houston, abstract deadline 1 June 2026 (extended).
**Word limit:** 200 words (hard, per online submission form).

---

## Title

Motion-Based Consequence Analysis: From Pre-Operation Risk Assessment to Live Operational Monitoring

## Abstract (200 words)

Current DP rules and guidance specify capability analysis in detail but treat consequence analysis (CQA) as little more than a periodic alarm. The prevailing requirement — recompute every five minutes, raise a flag only once a limit is breached — leaves the operator with no continuous indication of the probability of exceeding operational limits, in the intact condition or following a worst-case failure (WCF), and no view of the motion consequences of such a failure before it happens. The risk that CQA is meant to manage is inherently dynamic; the rules treat it as static.

This work develops a motion-based CQA framework that closes that gap end-to-end. A pre-operation risk assessment extends the conventional capability analysis with an estimate of vessel response and post-WCF excursion from the planned sea state, allowing the worst-case failure design intent (WCFDI) to be evaluated in motion terms. The same model is reformulated for online execution, producing a continuously updated CQA panel that estimates intact and post-WCF station-keeping footprint directly from live measurements and the DP model, without a runtime sea-state input. The panel is further extended to motion-sensitive mission equipment such as walk-to-work gangways.

Validation against a nonlinear DP simulator is complete; validation against logged operations is in progress.

---

## Notes on framing (not part of submission)

- **WCF** = Worst Case Failure (the event). **WCFDI** = Worst Case Failure Design Intent (what capability/consequence analyses are evaluated against). Used precisely in the abstract.
- **"DP model"** is the industry term for the observer/Kalman estimator carried by the DP controller. Avoids the term "observer" which is academic jargon for this audience.
- **"Capability analysis"** (not "capability polytope") is the standard term for the classical pre-operation deliverable.
- **No vendor or product names.** No marketing adjectives.
- **No outcome promised for the real-vessel validation** — framed as in-progress work that will be reported in the full paper.
- **Critique of current rules** is technically specific (5-minute recompute, binary alarm) rather than rhetorical, so it lands with a technical committee rather than reading as marketing.
