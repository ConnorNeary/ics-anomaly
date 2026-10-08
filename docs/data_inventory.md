# SWaT data inventory (Phase 0)

Produced with `scripts/inventory_swat.py`. Updated 2026-10-08. Only
descriptions and summary statistics are recorded here; the data itself
stays out of the repo (iTrust licence).

## Releases inventoried so far

| Release | Files | Span | Labels | Notes |
|---|---|---|---|---|
| Dec 2015 (A1 & A2) | `SWaT_Dataset_Normal_v1.xlsx`, `SWaT_Dataset_Attack_v0.xlsx` | 495,000 s normal + 449,919 s attack | Label column, known to be wrong in places; corrected ranges in `src/icsad/data/swat2015.py` | The dissertation release. 51 tags. |
| Jul 2019 (A4 & A5) | `SWaT_dataset_Jul 19.xlsx` (v1), `... v2.xlsx`, `readme.docx`, attack PDF | 4.17 h (14,996 rows) on 20 Jul 2019 | **No label column.** Six attacks listed in the PDF with start/end times | Normal run 2 h 15 min, then attacks 15:08–16:16 SGT. |
| Jun 2020 (A7) | 4 workbooks: 22 Jun 09:00–10:00 and 10:00–14:01; 29 Jun 10:00–12:00 and 13:30–15:30 | 9 h total | **No label column and no attack log in the zip** | 29 Jun sheet is named "XS2020 Forensics Historian", which suggests an exercise or event. |
| Jun 2021 (A8) | not yet downloaded | | | |
| Jun 2017 (A3), Dec 2019 (A6), 2022, 2023, 2026 | not yet inventoried (large, or not yet located) | | | |

## Format differences the loaders must handle

| | 2015 | Jul 2019 | Jun 2020 |
|---|---|---|---|
| Header | junk row, then tags | 3 rows: stage, tag, "value" | 1 row |
| Tag names | `LIT101` | `LIT 101`, `P101 Status`, `P1_STATE` | `LIT101.Pv`, `MV101.Status`, `LSH602.Alarm` |
| Time zone | local (SGT) | **UTC** (`Z` suffix); the attack PDF is in SGT (+8 h) | not stated |
| Sampling | 1 s | 1 s, one gap of 5.5 s | 22 Jun 10:00 file: rows ~1.005 s apart (14,400 rows over 4 h 1 min); others 1 s |
| Extra columns | none | +26: stage states `P1_STATE`..`P6_STATE`, level switches (`LS201`, `LSH601`...), P207/P208, AIT301-303, MV501-504 | +30 on 22 Jun (adds PSH301, DPSH301, PSH501, PSL501); 29 Jun has only 60 columns and **lacks P603** |
| Text columns | none | level switches: `Active`/`Inactive` in v2; Python dict strings in v1 | level switches `Active`/`Inactive`; 29 Jun file has a second time column `d_stamp` |

All 51 of the 2015 features are present in 2019 and 2020 once names are
normalised (`normalise_tag` in the inventory script), apart from P603 in
the 29 June 2020 files. The 51 shared tags are identical between the 2019
v1 and v2 files; v2 only fixes the level-switch columns.

## Finding: the same tags read on different scales across years

Minimum to maximum per file, selected tags:

| Tag | Jul 2019 | 22 Jun 2020 (10:00-14:00) | 29 Jun 2020 (10:00-12:00) |
|---|---|---|---|
| AIT201 | 113.8 - 146.8 | 14.4 - 110.4 | 38.8 - 117.4 |
| AIT203 | 198.1 - 272.3 | 13.9 - 158.7 | 142.5 - 177.1 |
| AIT402 | 3.3 - 88.0 | 0 (constant) | 0 (constant) |
| AIT503 | 1016 (constant) | 41.3 - 55.2 | 57.4 - 79.9 |
| PIT502 | 2.5 - 192.4 | 0.16 - 4.7 | 2.9 - 3.3 |
| FIT504 | 0.20 - 0.28 | 0 (constant) | ~0 |
| AIT401 | 0 (constant) | 0 (constant) | 0 (constant) |

Some of this may be attacks inside the windows (2020 has no attack log yet),
but constant or zero readings across whole files (AIT402, FIT504, AIT401,
AIT503 in 2019) are not attacks. They are sensors that are dead, saturated,
recalibrated, or reading a stage that was not running.

**Implications**

- A model trained on 2015 cannot be applied to later years as-is. Each
  release needs its own normal-period training or at least recalibration,
  and the brief's cross-release drift test has to separate genuine drift
  from these step changes.
- Constant tags must be detected and dropped per release; they carry no
  signal and break per-feature normalisation.
- For the product: this is direct evidence that a detector calibrated once
  will not survive changes at the plant it watches. It supports building
  calibration from each site's own normal data, and monitoring for
  sensors going flat.

## What each release can be used for

| Release | Train a model? | Evaluate? |
|---|---|---|
| Dec 2015 | Yes (5.7 days normal) | Yes (corrected labels) |
| Jul 2019 | Barely: ~2 h 15 min normal (~8,100 rows). Classical baselines or calibration only | Yes, once labels are built from the PDF (6 attacks, two with minute-level times only) |
| Jun 2020 | Not without knowing which periods are normal | No, until iTrust provides the attack log |

## Jul 2019 attack log (from the PDF; data timestamps are UTC)

| # | Target | Action | SGT (GMT+8) | UTC |
|---|---|---|---|---|
| 1 | FIT401 | spoof 0.8 -> 0.5 | 15:08:46 - 15:10:31 | 07:08:46 - 07:10:31 |
| 2 | LIT301 | spoof 835 -> 1024 | 15:15 - 15:19:32 | 07:15:00 - 07:19:32 (start to the minute) |
| 3 | P601 | OFF -> ON | 15:26:57 - 15:30:48 | 07:26:57 - 07:30:48 |
| 4 | MV201 + P101 | CLOSE -> OPEN, OFF -> ON | 15:38:50 - 15:46:20 | 07:38:50 - 07:46:20 |
| 5 | MV501 | OPEN -> CLOSE | 15:54 - 15:56 | 07:54 - 07:56 (minutes only) |
| 6 | P301 | ON -> OFF | 16:02:56 - 16:16:18 | 08:02:56 - 08:16:18 |

Normal operation 12:35-14:50 SGT (04:35-06:50 UTC). The file starts at
04:30 UTC, five minutes before the stated plant start, so the first rows
are start-up.

## Open items

1. Ask iTrust for the June 2020 attack log, and what "XS2020" was.
2. Download Jun 2021 (A8) and inventory it.
3. On the desktop: inventory Jun 2017 (A3), Dec 2019 (A6), and the
   2022 / 2023 / 2026 releases.
4. Build the 2019 loader with labels from the table above, and check the
   labels against the data (does each target tag move when the log says?).
