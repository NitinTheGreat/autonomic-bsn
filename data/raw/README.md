# Raw datasets

Two datasets are used. Neither is committed: `.gitignore` uses
`data/raw/*` + `!data/raw/README.md`, so the data stays out of the repo while
this file is tracked. Download them locally with the commands below.

| Dataset | Phase | Subjects | Rate | Columns | Nodes |
|---|---|---|---|---|---|
| [PAMAP2](#pamap2-physical-activity-monitoring) | 1 | 9 | 100 Hz | 54 | wrist / chest / ankle, full 9-axis IMU each |
| [MHEALTH](#mhealth-dataset) | 2 | 10 | 50 Hz | 24 | chest (**accel + ECG only**) / ankle / wrist |

> **The two schemas are not interchangeable.** MHEALTH's chest node has no gyro
> and no magnetometer, while PAMAP2's does. See the MHEALTH schema warning.

## PAMAP2 Physical Activity Monitoring

UCI Machine Learning Repository, dataset 231.
9 subjects, 3 Colibri wireless IMUs (wrist/hand, chest, ankle) + HR monitor,
sampled at **100 Hz**.

### Download

```bash
# from the repo root
mkdir -p data/raw && cd data/raw

curl -L -o pamap2.zip \
  "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"

unzip pamap2.zip                 # yields PAMAP2_Dataset.zip (a nested archive)
unzip PAMAP2_Dataset.zip         # yields PAMAP2_Dataset/{Protocol,Optional,...}

mv PAMAP2_Dataset pamap2         # Phase 1 expects data/raw/pamap2/Protocol/
```

## if using Powershell 

```
# from the repo root
New-Item -ItemType Directory -Force data\raw | Out-Null
Set-Location data\raw

c

```

Expected result:

```
data/raw/pamap2/Protocol/subject101.dat
data/raw/pamap2/Protocol/subject102.dat
...
data/raw/pamap2/Protocol/subject109.dat
```

Verify:

```bash
ls data/raw/pamap2/Protocol/subject10*.dat | wc -l    # -> 9
awk '{print NF; exit}' data/raw/pamap2/Protocol/subject101.dat   # -> 54
```

If your copy unpacks to a different layout, point the scripts at it with
`--data-dir`, or edit `data.pamap2_protocol_dirs` in `configs/models.yaml`.

### File format

Whitespace-separated, **no header**, **54 columns**, 0-indexed, `NaN` for
missing values.

| Column(s) | Meaning |
|---|---|
| 0 | timestamp (s) |
| 1 | activityID |
| 2 | heart rate (bpm) |
| 3–19 | IMU **hand/wrist** (17 cols) |
| 20–36 | IMU **chest** (17 cols) |
| 37–53 | IMU **ankle** (17 cols) |

Each 17-column IMU block, relative to its start:

| Offset | Meaning |
|---|---|
| +0 | temperature (°C) |
| +1 … +3 | **3D acceleration, ±16 g scale (m/s²)** |
| +4 … +6 | 3D acceleration, ±6 g scale (m/s²) |
| +7 … +9 | 3D gyroscope (rad/s) |
| +10 … +12 | 3D magnetometer (µT) |
| +13 … +16 | orientation — **invalid in this collection, do not use** |

So the ±16 g accelerometer columns Phase 1 uses are:

| Node | Block start | accel16 x,y,z |
|---|---|---|
| wrist (hand) | 3 | **4, 5, 6** |
| chest | 20 | **21, 22, 23** |
| ankle | 37 | **38, 39, 40** |

These indices are the only part of Phase 1's parsing that carries forward;
Phase 2 rebuilds the full 54-column parser behind `DataSource`.

### Activity IDs

Phase 1 uses these 8:

| ID | Activity |
|---|---|
| 1 | lying |
| 2 | sitting |
| 3 | standing |
| 4 | walking |
| 5 | running |
| 6 | cycling |
| 12 | ascending_stairs |
| 13 | descending_stairs |

ID `0` is *transient* activity between protocol items and is always dropped.

The Protocol files **also legitimately contain** IDs 7 (nordic_walking),
16 (vacuum_cleaning), 17 (ironing) and 24 (rope_jumping). These are excluded
from the 8-class set by design — `check_baseline_accuracy.py` reports them as
INFO, not as a warning. A *prominent* warning is raised only when an expected
ID is missing or an ID appears that is not in the documented PAMAP2 activity
list at all, which is the real signal that the label map is wrong.

Note that coverage is uneven across subjects: subject109 in particular performed
only a small subset of the protocol, so it is not used as a test subject.


---

# MHEALTH Dataset

UCI Machine Learning Repository, dataset 319.
10 subjects, 3 sensors (chest, right wrist, left ankle) + 2-lead ECG,
sampled at **50 Hz**.

### Download

```bash
# from the repo root
mkdir -p data/raw && cd data/raw

curl -L -o mhealth.zip   "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"

unzip mhealth.zip            # yields MHEALTHDATASET/
mv MHEALTHDATASET mhealth    # scripts expect data/raw/mhealth/
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force dataaw | Out-Null
Set-Location dataaw
curl.exe -L -o mhealth.zip "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
Expand-Archive mhealth.zip -DestinationPath .
Rename-Item MHEALTHDATASET mhealth
```

Expected result:

```
data/raw/mhealth/mHealth_subject1.log
...
data/raw/mhealth/mHealth_subject10.log
```

Verify with the shipped checker (verification only — no parser):

```bash
python scripts/verify_mhealth.py
```

### File format

Whitespace-separated (tabs in the published files), **no header**,
**24 columns**, 0-indexed.

| Column(s) | Meaning |
|---|---|
| 0, 1, 2 | **chest** acceleration x, y, z (m/s²) |
| 3, 4 | ECG lead 1, ECG lead 2 (mV) |
| 5, 6, 7 | **ankle** acceleration x, y, z (m/s²) |
| 8, 9, 10 | ankle gyroscope x, y, z (deg/s) |
| 11, 12, 13 | ankle magnetometer x, y, z (local) |
| 14, 15, 16 | **wrist** acceleration x, y, z (m/s²) |
| 17, 18, 19 | wrist gyroscope x, y, z (deg/s) |
| 20, 21, 22 | wrist magnetometer x, y, z (local) |
| 23 | activity label |

### ⚠️ Schema warning — the nodes are NOT uniform

| Node | accel | gyro | magnetometer | ECG |
|---|:--:|:--:|:--:|:--:|
| chest | ✅ | ❌ | ❌ | ✅ |
| ankle | ✅ | ✅ | ✅ | ❌ |
| wrist | ✅ | ✅ | ✅ | ❌ |

**MHEALTH's chest node carries accelerometer + ECG only — no gyroscope and no
magnetometer** — unlike PAMAP2, where all three nodes are full 9-axis IMUs.

Phase 2's feature extractor **must handle per-node channel availability rather
than assume a uniform schema, and must not zero-fill the missing channels**. A
zero gyro reading is not the same as an absent sensor: conflating the two would
inject a fake "sensor reads zero" signal into precisely the degradation study
this project exists to run.

### Activity labels (column 23)

Our canonical 6-class set, chosen to align with the PAMAP2 classes:

| ID | Activity |
|---|---|
| 1 | standing |
| 2 | sitting |
| 3 | lying |
| 4 | walking |
| 9 | cycling |
| 11 | running |

ID `0` is the null/transient class between protocol items and is always
dropped. It is the majority of every file (~78 % of rows).

Documented MHEALTH activities **excluded by design** — reported as INFO by
`verify_mhealth.py`, not as warnings:

| ID | Activity | Why excluded |
|---|---|---|
| 5 | climbing_stairs | **No ascending/descending distinction**, unlike PAMAP2's separate 12/13. It cannot be mapped onto the PAMAP2 stair classes. |
| 10 | jogging | Deliberately excluded to avoid conflating it with running (11); they are distinct labels here. |
| 6, 7, 8, 12 | waist bends, frontal arm elevation, knee bends, jumping | Gym exercises with no PAMAP2 counterpart. |

### Verified profile

All 10 files present at 24 columns. 1,215,745 rows, 6.8 hours total.
The 6 canonical classes are **exactly balanced at 30,720 rows each**
(3,072 rows = ~61 s per subject per class), which is unusual and worth
remembering: unlike PAMAP2, MHEALTH needs no class-balancing when sampling.
