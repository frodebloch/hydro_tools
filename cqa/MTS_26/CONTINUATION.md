# MTS DP Houston 2026 paper — continuation notes

Working notes for resuming work on the conference paper. Written at the
end of a long session; everything described here is committed.

## What this is

Conference paper for the MTS Dynamic Positioning Conference, Houston,
13–14 October 2026. Author Frode Bloch, Brunvoll AS. Title:
*Motion-Based Consequence Analysis: From Pre-Operation Risk Assessment
to Live Operational Monitoring*.

Subject: a motion-based extension to the classical DP consequence
analysis. Two halves — a pre-operation layer driven by forecast sea
states, and a live panel driven by measurements only.

## Where things live

- `cqa/MTS_26/paper.org` — the source of truth. `~/Documents/MTS_26` is
  a symlink to this directory, so the old editing path still works.
- `cqa/MTS_26/paper.pdf` — tracked deliberately, so each build is
  versioned.
- `cqa/MTS_26/mts.sty` — venue template, with three local bug fixes
  (see Gotchas).
- Figure sources are `cqa/scripts/run_*.py` plus
  `cqa/scripts/p7_brucon_validation/roll_up_live_operator_panel.py`.
  Figures are written straight into `MTS_26/images/`.

## Build

```
cd ~/Documents/MTS_26 && rm -f paper.aux paper.bbl paper.blg && \
  emacs --batch --eval "(progn (require 'org) (require 'ox-latex) \
    (find-file \"paper.org\") (org-latex-export-to-latex))" && \
  pdflatex -interaction=nonstopmode paper.tex && bibtex paper && \
  pdflatex -interaction=nonstopmode paper.tex && \
  pdflatex -interaction=nonstopmode paper.tex
```

`pdflatex` returns 1 even on success (a harmless template warning), so
chain with `;` not `&&`. Check for real problems with:

```
grep -E "Warning: (Citation|Reference)|Undefined" paper.log
```

Current state: 24 pages, 10 figures, 10 references, no unresolved
citations or cross-references.

## Structure

Abstract · Abbreviations · Introduction · Goal · Motion-Based
Capability Analysis · Consequence Analysis (Live) · Simulation Studies
· Conclusion · Acknowledgements

Pre-op deliverables are the two operability polars and the WCF decision
matrix. The operator display belongs to the live section — it was moved
there because `cqa/operator_view.py` documents it as an IMCA M254
approach/connection decision aid ("if WCF now", "abort and recover the
gangway"), which is not a planning artefact.

## Open items

1. **Acknowledgements is still `(TODO: whoever helped)`.**
2. **Check the LF/WF combination in `cqa/excursion.py`** — does the
   quadrature happen on the σ's before the running-max step, or on the
   two peak values after it? The assumptions bullet now only claims
   independence, so nothing is wrong in the paper, but it would be good
   to know which the intact path does. The gangway end-stop check adds
   a peak to a level and that is deliberate.
3. **Real-vessel validation** was dropped from the paper and moved to
   future work. Data is reportedly available (vessels operating west of
   Florø to west of Stavanger). If it returns: NORA3 is the right
   hindcast for that area, not ERA5. The intact side is what can be
   validated; post-WCF cannot be, since nobody triggers a WCF for a
   paper.
4. **Captions are worth a general pass.** Two had grown into second
   copies of the body text and were cut hard; the roll-up captions may
   have the same problem.

## Conventions settled this session

Do not undo these without reason — several were corrections of real
errors.

- **CQA**, not CA, for Consequence Analysis. "Classical CQA" for the
  rules-mandated function, "motion-based CQA" for the extension.
- **WCF** only. WCFDI has been removed entirely.
- **IMCA M254** is *Guidelines for Walk to Work Operations*, August
  2025. The 2 m / 4 m radii in it are examples showing how to build the
  procedure, not fixed limits — the paper says so.
- **Percentiles in plain language.** No bare P90 in captions; "exceeded
  about one time in ten". Notation is introduced once in method bullet
  c and defined in the abbreviations.
- **"limits" not "radii"** wherever the threshold spans vessel and
  mission equipment, since the gangway limit is a stroke, not a circle.
  Genuine position circles keep "radius".
- **No internal parameters in operator-facing text** — α, k_σ and the
  thruster-group scenario detail have been removed. `T_realloc` stays,
  being a real modelling parameter.
- **Statistics are deliberately understated.** Extreme-value detail is
  deferred to Rice and Vanmarcke. There are vocal sceptics of
  statistical methods in the DP community and the paper avoids giving
  them a target.

## Gotchas

- **`mts.sty` has three local fixes**: `p{}` column spec, footer page
  column widened so "Page N" fits on one line, and outer `@{}` on the
  header tabular which was overrunning the right margin by 12 pt on
  every page. Re-apply if the template is ever refreshed.
- **`mathptmx` must not be loaded.** It was in the paper header and
  silently broke every equation — `+`, parentheses and `\ddot`/`\dot`
  accents vanished, because it overrode the template's `newtxmath`.
  Eq. 1 rendered as `Mη Dη = τenv τc` for who knows how long.
- **`#+OPTIONS: ^:nil` is required** or org eats underscores as
  subscripts, mangling cell tags like `bf8p5_q10_w45`.
- **Keep each inline `\(...\)` on one `#+CAPTION:` line.** Splitting one
  across two lines prints raw LaTeX.
- **A line starting with `+ `** is parsed as a list item and will break
  a sentence into a stray bullet.
- **`operability_polar()` and `wcfdi_operability_overlay()` share
  several identical anchor lines.** Scripted edits leaked into the wrong
  function once. Split the file at the function boundary before doing
  text substitution.
- **Do not use `git stash` in this repo** — several hundred unrelated
  modified files live alongside, and a rebuild touching `paper.pdf`
  makes the pop fail.

## Recent substantive changes

- Added the forecast-conditioned operability polar: wave condition and
  current pinned to a forecast, wind swept, per-heading limiting wind
  speed with the forecast wind drawn as a ring. Needed threading
  `Hs_fixed`/`Tp_fixed` and non-collinear wind/current angles through
  `wcfdi_transient` and the overlay.
- Fixed a real bug: the bistability gate forced both motion metrics to
  `+inf` together, so the vessel and gangway post-WCF boundaries came
  out bit-identical and the plotted "no-go boundary" was actually the
  bistability boundary. Now bisected separately as `bist_Vw` and
  labelled *possible large transient*, matching the live panel's Large
  Transient alert.
- Merged the two design polars into one figure and section.
- Added the synthetic storm profile figure so the decision matrix can
  be interpreted.
- Gangway bar now plots the combined slow+wave demand as its primary
  marker, which is what the utilisation figure and end-stop probability
  are computed from.
