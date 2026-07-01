# AI-Assisted Breast Cancer Histopathology Triage Tool

**Status:** Phase 0–2 (Foundations, Data Pipeline, Baseline Model)
**Positioning:** Research / second-opinion aid for pathologists. NOT a diagnostic device.

---

## 1. Dataset

**BCNB — Early Breast Cancer Core-Needle Biopsy WSI Dataset**
Source: https://bupt-ai-cz.github.io/BCNB/

- 1,058 patients' whole-slide images (WSIs)
- Tumor region annotations (contours)
- Clinical metadata: ER, PR, HER2 status, molecular subtype, lymph node metastasis

**Download note:** BCNB is distributed via a request form + Baidu/Google Drive links from the
official site (link above). Because of file size (WSIs are gigapixel), you won't download
the full raw set locally — instead:

1. Request access via the official BCNB page.
2. Upload the WSIs (or a patient subset to start) directly to a **Kaggle Dataset** (private) —
   this avoids re-uploading every session and lets Kaggle notebooks mount it instantly.
3. Start with 20–30 patients for pipeline development before scaling to the full 1,058.

---

## 2. Environment Setup

### Option A: Kaggle Notebook (recommended)
1. Create a new Kaggle Notebook, enable GPU (Settings → Accelerator → GPU P100 or T4 x2)
2. Attach your BCNB Kaggle Dataset
3. Install extra deps in the first cell:
```bash
pip install -q openslide-python opencv-python-headless albumentations timm
apt-get install -y openslide-tools
```

### Option B: Local (CPU-only, for pipeline dev/testing on small samples only)
```bash
pip install -r requirements.txt
```
(Full training will NOT run on CPU in reasonable time — local is for testing tiling logic only.)

---

## 3. Project Structure

```
histopath-project/
├── README.md
├── requirements.txt
├── src/
│   ├── tiling.py          # Phase 1: WSI → patch extraction pipeline
│   ├── dataset.py         # Phase 1/2: PyTorch Dataset + DataLoader
│   ├── train_baseline.py  # Phase 2: baseline CNN training script
│   └── utils.py           # shared helpers (tissue detection, patient-level splits)
├── notebooks/
│   └── 01_data_exploration.md   # what to check first when you get the data
└── docs/
    └── phase_checklist.md       # full phase-by-phase checklist (living doc)
```

---

## 4. Roadmap (see docs/phase_checklist.md for detail)

- [x] Phase 0: Foundations & environment
- [ ] Phase 1: Data pipeline (tiling.py, dataset.py)
- [ ] Phase 2: Baseline patch classifier (train_baseline.py)
- [ ] Phase 3: MIL / attention-based slide-level model
- [ ] Phase 4: Web MVP with heatmap visualization
- [ ] Phase 5: Validation report
- [ ] Phase 6: Go-to-market groundwork

---

## 5. Next Immediate Steps

1. Request BCNB access (do this today — approval can take time)
2. While waiting: run `src/tiling.py` against any public sample WSI (e.g., a CAMELYON16
   sample slide) just to test the pipeline mechanics
3. Set up your Kaggle account + verify GPU quota (Settings → Accelerator)
