"""Small SQLite index for completed run summaries."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SQLiteRunIndex:
    """Index run summaries for traceability without becoming the source of raw data."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def record_run(self, summary: Mapping[str, Any]) -> None:
        """Insert or replace one run summary row."""

        run_id = str(summary["run_id"])
        sample_info = _mapping(summary.get("sample_info"))
        recipe_info = _mapping(summary.get("recipe"))
        calibration_info = _mapping(summary.get("calibration"))
        output_files = _mapping(summary.get("output_files"))
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                insert or replace into runs (
                    run_id,
                    sample_id,
                    recipe_id,
                    calibration_id,
                    final_judgement,
                    run_dir,
                    report_html,
                    summary_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sample_info.get("sample_id"),
                    recipe_info.get("recipe_id"),
                    calibration_info.get("calibration_id"),
                    summary.get("final_judgement"),
                    summary.get("run_dir"),
                    output_files.get("report_html"),
                    json.dumps(summary, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one indexed run summary by run_id."""

        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "select summary_json from runs where run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(json.loads(row[0]))

    def list_runs(self) -> list[dict[str, Any]]:
        """Return compact run rows ordered by newest insertion/update."""

        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                select run_id, sample_id, recipe_id, calibration_id, final_judgement,
                       run_dir, report_html
                from runs
                order by rowid desc
                """
            ).fetchall()
        return [
            {
                "run_id": row[0],
                "sample_id": row[1],
                "recipe_id": row[2],
                "calibration_id": row[3],
                "final_judgement": row[4],
                "run_dir": row[5],
                "report_html": row[6],
            }
            for row in rows
        ]

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                create table if not exists runs (
                    run_id text primary key,
                    sample_id text not null,
                    recipe_id text,
                    calibration_id text,
                    final_judgement text,
                    run_dir text,
                    report_html text,
                    summary_json text not null
                )
                """
            )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["SQLiteRunIndex"]
