"""
Phase 0 inventory of SWaT releases: what each workbook actually contains,
before any loader is designed for it.

    python -I scripts/inventory_swat.py <folder> [<folder> ...] [--out inventory.csv]

For every sheet of every .xlsx/.csv under the folders: header layout, row
count, time span, sampling interval and gaps, timestamp format and zone,
tag list (normalised to 2015-style names, e.g. 'FIT 101' / 'LIT101.Pv' ->
'FIT101'), text-valued columns, any label column, and how the tag set
compares with the 51 features of the December 2015 release.

Read-only: never modifies the source files. Release layouts differ, so the
header detection is heuristic -- check the printed header rows.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SWAT2015_TAGS = [
    'FIT101', 'LIT101', 'MV101', 'P101', 'P102',
    'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',
    'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401',
    'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'P501', 'P502', 'PIT501', 'PIT502', 'PIT503',
    'FIT601', 'P601', 'P602', 'P603',
]
TAG_RE = re.compile(r'^(P\d_?STATE|[A-Z]{1,6}\s?_?\d{3}\w*)', re.I)
LABEL_RE = re.compile(r'attack|label|normal', re.I)
TIME_RE = re.compile(r'time|t_stamp|stamp|gmt', re.I)


def normalise_tag(name):
    """'FIT 101' / 'LIT101.Pv' / 'P101 Status' / 'P1 STATE' -> 'FIT101' / 'LIT101' / 'P101' / 'P1_STATE'."""
    s = str(name).strip()
    s = re.sub(r'\.(Pv|Status|Alarm)$', '', s, flags=re.I)
    s = re.sub(r'\s+Status$', '', s, flags=re.I)
    s = re.sub(r'^(P\d)[\s_]*STATE$', r'\1_STATE', s, flags=re.I)
    return s.replace(' ', '').upper()


def _looks_like_time(v):
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, 'year'):
        return True
    if isinstance(v, str) and re.match(r'^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}[ T]\d', v):
        return True
    return False


def detect_layout(top):
    """top: first rows read with header=None. Returns (tag_row, data_start)."""
    data_start = next(i for i in range(len(top)) if _looks_like_time(top.iat[i, 0]))
    header_rows = range(data_start)
    tag_row = max(header_rows, key=lambda r: sum(bool(TAG_RE.match(str(v))) for v in top.iloc[r]))
    return tag_row, data_start


def parse_times(col):
    s = col.astype(str).str.strip()
    utc = bool(s.str.endswith('Z').mean() > 0.5)
    t = pd.to_datetime(s, utc=utc, format='mixed', dayfirst=bool(s.str.match(r'^\d{1,2}/').mean() > 0.5))
    return t, ('UTC (Z suffix)' if utc else 'naive / unstated')


def inventory_sheet(path, sheet, raw):
    tag_row, data_start = detect_layout(raw.head(10))
    header = [str(v) for v in raw.iloc[tag_row]]
    data = raw.iloc[data_start:].reset_index(drop=True)
    data.columns = range(data.shape[1])
    times, zone = parse_times(data[0])
    dt = times.diff().dt.total_seconds().dropna()

    cols = {}
    text_cols, label_cols = {}, []
    for j, name in enumerate(header[1:], start=1):
        if name in ('nan', ''):
            continue
        if LABEL_RE.search(name) and not TAG_RE.match(name):
            label_cols.append(name)
        values = data[j]
        numeric = pd.to_numeric(values, errors='coerce')
        if numeric.notna().mean() < 0.99:
            uniq = values.dropna().astype(str).unique()
            text_cols[name] = sorted(uniq)[:6]
        cols[normalise_tag(name)] = name

    tags = [c for c in cols if TAG_RE.match(c)]
    in_2015 = [t for t in SWAT2015_TAGS if t in tags]
    return {
        'file': path.name, 'sheet': sheet,
        'header_rows': data_start, 'tag_row': tag_row,
        'time_col': str(raw.iat[tag_row, 0]), 'time_zone': zone,
        'rows': len(data), 'start': times.iloc[0], 'end': times.iloc[-1],
        'hours': round((times.iloc[-1] - times.iloc[0]).total_seconds() / 3600, 2),
        'median_dt_s': round(float(dt.median()), 4) if len(dt) else None,
        'gaps_over_2s': int((dt > 2).sum()), 'max_gap_s': round(float(dt.max()), 1) if len(dt) else None,
        'n_columns': len(cols), 'n_tags': len(tags),
        'n_2015_tags_present': len(in_2015),
        'missing_2015_tags': [t for t in SWAT2015_TAGS if t not in tags],
        'extra_tags': [t for t in tags if t not in SWAT2015_TAGS],
        'text_columns': text_cols, 'label_columns': label_cols,
        'header_sample': header[:6],
    }


def inventory(paths):
    rows = []
    for folder in paths:
        for path in sorted(Path(folder).rglob('*')):
            if path.suffix.lower() == '.xlsx':
                sheets = pd.read_excel(path, sheet_name=None, header=None, engine='openpyxl')
            elif path.suffix.lower() == '.csv':
                sheets = {'csv': pd.read_csv(path, header=None, low_memory=False)}
            else:
                continue
            for sheet, raw in sheets.items():
                try:
                    rows.append({'folder': Path(folder).name, **inventory_sheet(path, sheet, raw)})
                except Exception as exc:  # report and carry on: layouts vary
                    rows.append({'folder': Path(folder).name, 'file': path.name, 'sheet': sheet,
                                 'error': f'{type(exc).__name__}: {exc}'})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--out', help='write the full table to this CSV')
    args = ap.parse_args()

    df = inventory(args.folders)
    for _, r in df.iterrows():
        print(f"\n=== {r['folder']} / {r['file']} [{r['sheet']}]")
        if isinstance(r.get('error'), str):
            print('  ERROR', r['error'])
            continue
        print(f"  header: {r['header_rows']} rows (tags on row {r['tag_row']}), time column "
              f"{r['time_col']!r}, {r['time_zone']}; first headers {r['header_sample']}")
        print(f"  {r['rows']:,} rows, {r['start']} -> {r['end']} ({r['hours']} h), median step "
              f"{r['median_dt_s']} s, {r['gaps_over_2s']} gaps > 2 s (max {r['max_gap_s']} s)")
        print(f"  {r['n_columns']} columns, {r['n_tags']} tags; {r['n_2015_tags_present']}/51 of the 2015 "
              f"features present")
        print(f"  missing vs 2015: {r['missing_2015_tags']}")
        print(f"  extra vs 2015:   {r['extra_tags']}")
        print(f"  text-valued columns: {r['text_columns'] or 'none'}")
        print(f"  label column: {r['label_columns'] or 'NONE -- labels must come from an attack log'}")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f'\nwrote {args.out}')
    sys.exit(1 if 'error' in df and df['error'].notna().any() else 0)


if __name__ == '__main__':
    main()
