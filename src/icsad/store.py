"""
Results store: one SQLite file holding every run and every evaluation of it.

  runs         one row per trained model: config, seed, provenance, timing
  evaluations  one row per (run, score aggregation, tuning metric): the
               selected tau/w and the tuning score on attack-val
  metrics      test-partition metric values for each evaluation

Retuning a cached run against another metric or aggregation adds an
evaluation; it never retrains a model or overwrites another evaluation.

summary() reports distributions across seeds (n, mean, std, min, max),
never a best run.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    experiment    TEXT NOT NULL,
    model         TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    split_seed    INTEGER NOT NULL,
    run_key       TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    status        TEXT NOT NULL,          -- running | done | failed
    artifact_dir  TEXT NOT NULL,
    git_commit    TEXT,
    git_dirty     INTEGER,
    host          TEXT,
    device        TEXT,
    deterministic INTEGER,
    versions_json TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    train_time_s  REAL,
    n_params      INTEGER,
    epochs_run    INTEGER,
    best_val_loss REAL,
    extra_json    TEXT,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
    eval_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    aggregation    TEXT NOT NULL,
    tuning_metric  TEXT NOT NULL,
    tau            REAL,
    tau_percentile REAL,
    w              INTEGER,
    tuning_score   REAL,
    created_at     TEXT,
    UNIQUE (run_id, aggregation, tuning_metric)
);
CREATE TABLE IF NOT EXISTS metrics (
    eval_id INTEGER NOT NULL REFERENCES evaluations(eval_id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    value   REAL,
    PRIMARY KEY (eval_id, name)
);
"""

RUN_COLUMNS = ('run_id', 'experiment', 'model', 'seed', 'split_seed', 'train_time_s',
               'n_params', 'epochs_run', 'device', 'git_commit')


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class ResultsStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys = ON')
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # ---- runs ---------------------------------------------------------------

    def get_run(self, run_id):
        row = self.conn.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        return dict(row) if row else None

    def start_run(self, *, run_id, experiment, model, seed, split_seed, run_key, config,
                  artifact_dir, provenance):
        """Register a run as running. A previous failed or interrupted attempt
        with the same run_id is replaced, along with its evaluations."""
        with self.conn:
            self.conn.execute('DELETE FROM runs WHERE run_id = ?', (run_id,))
            self.conn.execute(
                'INSERT INTO runs (run_id, experiment, model, seed, split_seed, run_key, config_json, '
                'status, artifact_dir, git_commit, git_dirty, host, device, deterministic, '
                'versions_json, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (run_id, experiment, model, seed, split_seed, run_key, json.dumps(config, sort_keys=True),
                 'running', str(artifact_dir), provenance.get('git_commit'), provenance.get('git_dirty'),
                 provenance.get('host'), provenance.get('device'), provenance.get('deterministic'),
                 json.dumps(provenance.get('versions', {}), sort_keys=True), _now()))

    def finish_run(self, run_id, *, train_time_s, n_params=None, epochs_run=None,
                   best_val_loss=None, extra=None):
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = 'done', finished_at = ?, train_time_s = ?, n_params = ?, "
                'epochs_run = ?, best_val_loss = ?, extra_json = ? WHERE run_id = ?',
                (_now(), train_time_s, n_params, epochs_run, best_val_loss,
                 json.dumps(extra or {}, sort_keys=True, default=str), run_id))

    def fail_run(self, run_id, error):
        with self.conn:
            self.conn.execute("UPDATE runs SET status = 'failed', finished_at = ?, error = ? "
                              'WHERE run_id = ?', (_now(), error, run_id))

    def runs(self, experiment=None, status='done'):
        q, args = 'SELECT * FROM runs WHERE status = ?', [status]
        if experiment:
            q += ' AND experiment = ?'
            args.append(experiment)
        return [dict(r) for r in self.conn.execute(q + ' ORDER BY model, seed', args)]

    # ---- evaluations --------------------------------------------------------

    def add_evaluation(self, run_id, aggregation, tuning_metric, tau, tau_percentile, w,
                       tuning_score, metrics):
        with self.conn:
            self.conn.execute('DELETE FROM evaluations WHERE run_id = ? AND aggregation = ? '
                              'AND tuning_metric = ?', (run_id, aggregation, tuning_metric))
            cur = self.conn.execute(
                'INSERT INTO evaluations (run_id, aggregation, tuning_metric, tau, tau_percentile, w, '
                'tuning_score, created_at) VALUES (?,?,?,?,?,?,?,?)',
                (run_id, aggregation, tuning_metric, float(tau), float(tau_percentile), int(w),
                 float(tuning_score), _now()))
            self.conn.executemany('INSERT INTO metrics (eval_id, name, value) VALUES (?,?,?)',
                                  [(cur.lastrowid, k, float(v)) for k, v in metrics.items()])

    def results(self, experiment=None, aggregation='mean', tuning_metric='range_f1'):
        """One row per finished run: run info, chosen tau/w, and every test metric."""
        q = ('SELECT r.run_id, r.experiment, r.model, r.seed, r.split_seed, r.train_time_s, '
             'r.n_params, r.epochs_run, r.device, r.git_commit, e.eval_id, e.tau, '
             'e.tau_percentile, e.w, e.tuning_score '
             'FROM runs r JOIN evaluations e ON e.run_id = r.run_id '
             "WHERE r.status = 'done' AND e.aggregation = ? AND e.tuning_metric = ?")
        args = [aggregation, tuning_metric]
        if experiment:
            q += ' AND r.experiment = ?'
            args.append(experiment)
        runs = pd.read_sql_query(q, self.conn, params=args)
        if runs.empty:
            return runs
        m = pd.read_sql_query(
            f'SELECT eval_id, name, value FROM metrics WHERE eval_id IN '
            f'({",".join("?" * len(runs))})', self.conn, params=runs['eval_id'].tolist())
        wide = m.pivot(index='eval_id', columns='name', values='value').reset_index()
        return runs.merge(wide, on='eval_id').drop(columns='eval_id').sort_values(['model', 'seed'])

    def summary(self, metric_names, experiment=None, aggregation='mean', tuning_metric='range_f1'):
        """Distribution across seeds per (experiment, model): n, mean, std, min, max."""
        df = self.results(experiment, aggregation, tuning_metric)
        if df.empty:
            return df
        return df.groupby(['experiment', 'model'])[list(metric_names)].agg(
            ['count', 'mean', 'std', 'min', 'max'])
