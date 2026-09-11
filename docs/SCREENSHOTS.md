# Demo capture guide

Four screenshots to take from the live Hugging Face Space, in this order. Save them to
`docs/screenshots/` using the filenames below so the README can reference them.

Take them at a browser width of roughly 1280px so both columns are visible without scrolling.

---

## 1. `01-landing.png` — the landing state

Open the Space and capture before uploading anything.

**Must be visible:** the title, the MFL/ultrasonic disclaimer, and the six example thumbnails.

This is the credibility shot. A reviewer sees the scope limitation stated up front, before any
prediction is on screen, which is the opposite of overselling.

## 2. `02-correct-prediction.png` — a confident, correct prediction

Click the **crazing** example (or upload `app/examples/crazing.jpg`).

**Expect:** `crazing` at roughly 97% confidence, with the Grad-CAM overlay concentrated on the
crack network rather than on background texture.

**Must be visible:** the top-3 probability bars, the Grad-CAM overlay, and the plain-English
interpretation panel together in one frame.

The point is not that it is correct — it is that the saliency map shows the model attending to
the defect itself. Correctness alone proves nothing about *why*.

## 3. `03-failure-case.png` — a failure or low-confidence case

Try the **pitted_surface** example, then the **inclusion** example.

The held-out test set contains exactly one error in 270 images: a `pitted_surface` image
predicted as `inclusion` (see `reports/results.csv` — `pitted_surface` recall 0.9778,
`inclusion` precision 0.9783). If the bundled example does not reproduce it, capture instead
whichever example yields the **lowest top-1 confidence**, and caption it accurately as a
low-confidence case rather than implying it is the known error.

Include this one. Showing a failure signals the evaluation was honest rather than curated, and
it is usually the slide that earns the most trust.

## 4. `04-corrupted-input.png` — degraded field conditions

Upload `docs/demo_inputs/crazing_motion_blur_moderate.png`.

These files were generated with the **same corruption functions used in the robustness study**
(`scripts/robustness_test.py`), so what is on screen corresponds directly to the numbers in
README Section 6 — moderate motion blur sits at 0.4630 accuracy against 0.9963 clean.

**Expect:** confidence to drop sharply, and quite possibly a wrong top-1 class.

Alternatives in `docs/demo_inputs/`, in rough order of severity:

| File | Corresponds to |
|---|---|
| `crazing_defocus_moderate.png` | defocus blur, moderate — 0.7481 |
| `crazing_gaussian_noise_moderate.png` | gaussian noise, moderate — 0.4296 |
| `crazing_motion_blur_moderate.png` | motion blur, moderate — 0.4630 |
| `crazing_low_light_severe.png` | low light, severe — 0.5963 |
| `crazing_motion_blur_severe.png` | motion blur, severe — 0.3037 |

This is the most important of the four. It turns the project's central finding — a 31.80
percentage-point mean accuracy drop under simulated field conditions — from a table into
something a viewer can see happening.

---

## After capturing

Send the files and they can be added to the README: view 1 near the top beside the demo link,
views 2 and 3 in Section 7 (Explainability), and view 4 in Section 6 (Robustness) where the
corruption numbers are reported.

Caption view 4 explicitly as a simulated corruption of a laboratory image, not field imagery —
the same distinction Section 8 draws.
