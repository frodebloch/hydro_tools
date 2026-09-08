# Speaker notes - MTS DP 2026

Motion-Based Consequence Analysis: From Pre-Operation Risk Assessment to Live Operational Monitoring

Frode Bloch, Brunvoll AS - 2026-10-13, Houston.

Generated from `build_deck.py`. These are intentionally kept out
of the pptx, which is the file submitted to the conference.

---

## Slide 1 - Motion-Based Consequence Analysis

MTS DP Conference, Houston, 13-14 October 2026.

## Slide 3 - Capability is not the whole answer

Set up the gap. Classical CQA is a capacity check. The limits that actually stop a W2W operation are motion limits. Both matter; the capacity check is necessary but not sufficient.

## Slide 4 - Goal: a motion layer on top of the classical CQA

Emphasise additive, not a replacement. Nothing here suppresses a rule-mandated alarm. That matters to this audience.

## Slide 6 - From forecast sea state to motion envelope

Keep the statistics light out loud. Rice and Vanmarcke are in the paper if anyone asks. Stress that the sea state comes from the forecast and the vessel model is the same one the DP uses.

## Slide 7 - Forecast-conditioned polar

The planning artefact. Explain the polar convention here - radius is wind speed, angle is the heading you would choose. Where the ring sits relative to the coloured band is the go / no-go read. Note the gangway is often the binding constraint, not the vessel. The dashed contour is the post-WCF boundary.

## Slide 8 - A 24-hour storm to plan against

Short slide. Only here so the matrix that follows can be read.

## Slide 9 - WCF decision matrix

THE key pre-op slide - the three panels are three different questions at the same cell. Classical CQA: force vs residual capability, shows yellow caution and a small red spot at 12:00 in beam sectors. Intact motion adds nothing - the vessel never nears the alarm radius. Post-WCF motion changes the picture: saturated red in a narrow 2-hour window at the storm peak in beam headings, where the classical panel says only caution. So the plan changes from 'operate through, ease off around 12:00' to 'avoid the beam sector 11:00-13:00, or take one of the two green heading windows'. That is the added operational content: the WHEN and WHICH HEADING, not just the WHETHER.

## Slide 11 - The live panel needs no forecast

Stress: no vessel-specific fudge factor. The motion layer consumes what the onboard classical CQA already publishes.

## Slide 12 - Post-WCF motion: an ensemble, not one run

Illustrative figure, no numbers on purpose. The point is the spread, which is what the WCF could catch you in.

## Slide 13 - Two independent channels

Two channels means a cell has to pass two independent gates. Defensible to a sceptic: one channel is statistical, one is not.

## Slide 14 - The live operator display

SPEND TIME HERE. This is the centre of the talk. Walk the top bar, then the bottom bar, then the thresholds, then the headline probabilities. IMCA M254 approach / connection decision aid: do I connect, do I stay, do I abort and recover the gangway.

## Slide 15 - Validation: intact motion

The intact axis is the part that can be validated against a real vessel later. Prediction tracks the simulator cell by cell and sits a little above it - that is the controller conservatism, kept as margin rather than tuned away.

## Slide 16 - Validation: post-WCF, and a deliberate limit

LEAD WITH THE DESIGN INTENT, not the miss. Own it: on the three cells where the survivor allocator saturates, the linear closed-loop ODE is no longer sufficient and the motion P95 drifts from truth. bf8p5_q10_w45: simulator LF peak P95 is 29.9 m, driven by seeds where the vessel loses station outright; we predict 5.4 m. An 82% underprediction. Then the recovery: the operator is not misled, because 5.4 m already exceeds the 4 m alarm (red on 28 of 30 seeds) AND the saturation channel reads p_sat 0.14 mean, 0.26 max against a 0.10 red threshold (Large Transient on 21 of 30, amber on the rest). Two independent gates, both red, before the simulator ever runs the WCF. That is the design intent of the two-channel layout: where the linear ODE breaks, the trust indicator triggers. WHY NOT JUST FIX THE ODE: three refinements were investigated and rejected. Online QTF-gradient estimation needs a runtime wave-field model, which breaks in mixed sea, cross-current and shielding - and breaks the measurement-only design goal. Pre-trip variance and pre-trip PSD shape were both empirically falsified: per-cell correlations run from -0.7 to +0.7 with no consistent sign. The post-WCF peak depends on the wave phase at the instant of the trip, which is not in the past record. So this is a demonstrated limit, not a shortcut: the two-channel design is the tight envelope for this vessel and instrumentation.

## Slide 17 - Conclusion

Land on: the deliverables are buildable today from what a modern DP system already knows. NOTE - first session, and there is no JDP talk in the programme, so nobody will have introduced it before you. Say in one sentence what the JDP DP Consequence Analysis (CA) is and that this work is offered as input, otherwise that bullet means nothing to much of the room.
