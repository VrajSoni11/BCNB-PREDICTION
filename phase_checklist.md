# Phase Checklist (living document)

## Phase 0 — Foundations ✅ (this scaffold)
- [x] Project structure set up
- [x] Free-GPU strategy decided (Kaggle primary)
- [ ] BCNB access requested (**do this first — approval can take days**)
- [ ] Kaggle account + GPU quota verified

## Phase 1 — Data Pipeline
- [ ] Run `notebooks/01_data_exploration.md` checklist on real data
- [ ] Adjust `load_tumor_mask()` in `tiling.py` to match actual BCNB annotation format
- [ ] Tile a 20-30 patient subset, manually verify patch quality
- [ ] Tile full/larger dataset once pipeline is verified
- [ ] Upload tiled patches as a Kaggle Dataset (private) for fast reuse across sessions

## Phase 2 — Baseline Model
- [ ] Run `train_baseline.py` on the subset, confirm training loop works end-to-end
- [ ] Check for class imbalance issues — consider weighted loss or oversampling if
      tumor patches are a small minority
- [ ] Establish baseline AUROC/F1 — this is your reference point for everything after
- [ ] Scale to larger patient set once baseline is stable

## Phase 3 — MIL / Slide-Level Model (not yet built — next milestone)
- [ ] Implement attention-based Multiple Instance Learning (MIL) to aggregate
      patch predictions into a slide-level verdict
- [ ] Add attention heatmap visualization (this is your core product feature)
- [ ] Compare MIL results against simple patch-averaging baseline

## Phase 4 — MVP Product Layer (not yet built)
- [ ] Simple web UI: upload → heatmap + confidence output
- [ ] Prominent "research tool, not diagnostic" disclaimer
- [ ] (Optional) ER/PR/HER2 prediction as secondary exploratory feature

## Phase 5 — Validation & Reporting (not yet built)
- [ ] Write up methodology + metrics + limitations, compare to published BCNB benchmarks

## Phase 6 — Go-to-Market (not yet built)
- [ ] Identify 2-3 pathology labs/research groups for demo feedback
- [ ] Research regulatory pathway (SaMD / local equivalent) even if early

---
**Update this file as you complete steps — it's your single source of truth for progress.**
