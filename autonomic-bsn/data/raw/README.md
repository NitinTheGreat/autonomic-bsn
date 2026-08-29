# Raw datasets

Nothing in this directory is committed — `data/raw/` is in `.gitignore`
(PAMAP2 is ~1.6 GB unpacked). Download it locally with the commands below.

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
