# 01 — Data Exploration Checklist

Once BCNB access is approved, run through this before writing more pipeline code.

## 1. Inspect a single WSI
```python
import openslide
slide = openslide.OpenSlide("path/to/slide.svs")
print("Dimensions (level 0):", slide.dimensions)
print("Level count:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)
print("Level downsamples:", slide.level_downsamples)
```
Note the typical file size and dimensions — this tells you which pyramid `level`
to tile at in `tiling.py` (level 0 is usually too large/slow to tile directly;
level 1 or 2 is typical for 512px patches).

## 2. Inspect the annotation format
Open one annotation file and check:
- Is it JSON, XML, or a mask image?
- What coordinate system (level 0 pixel coords, or normalized 0-1)?
- **Adjust `load_tumor_mask()` in `tiling.py` to match** — the current implementation
  assumes a JSON list of polygon point lists in level-0 pixel coordinates. This is
  a starting assumption, not a guarantee — verify against real BCNB files.

## 3. Check clinical metadata
Load the clinical CSV/spreadsheet (ER/PR/HER2/molecular subtype/lymph node status)
and check:
- Class balance for each label (tumor subtype prediction will likely be imbalanced)
- Missing values — decide whether to drop or impute
- How patient IDs map to WSI filenames (this mapping is what `slide_id` in the
  manifest needs to match)

## 4. Run tiling.py on ONE slide first
```bash
python src/tiling.py --wsi_path sample_slide.svs \
                      --annotation_path sample_annot.json \
                      --out_dir test_patches/ \
                      --patch_size 512 --level 1
```
Manually open 10-15 saved patches and visually confirm:
- Tissue patches actually contain tissue (not background)
- Tumor-labeled patches actually overlap visible abnormal tissue

Only after this manual sanity check should you run tiling across the full dataset.

## 5. Estimate storage/compute budget
- Patch count per slide × 1,058 patients = your total dataset size
- At 512×512 PNG (~200-400KB each), estimate total GB — make sure it fits your
  Kaggle Dataset storage quota (currently ~100GB private, verify current limits)
- If too large: increase `stride` (fewer, non-overlapping patches) or start with
  a 100-200 patient subset for Phase 2 baseline training
