"""Optimization tracking and experiment history for GEPA-Karenina integration.

Provides persistent storage of optimization runs using SQLite for experiment
tracking, comparison, and reproducibility.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class OptimizationRun(BaseModel):
    """Record of a single optimization run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    benchmark_name: str

    # Configuration
    targets: list[str]  # What was optimized
    seed_prompts: dict[str, str]  # Initial prompts

    # Results
    optimized_prompts: dict[str, str]
    train_score: float
    val_score: float
    test_score: float | None = None
    improvement: float  # vs baseline (relative improvement)

    # GEPA params
    reflection_model: str
    metric_calls: int

    # Trajectory summary
    best_generation: int
    total_generations: int

    # Per-model scores (for Pareto analysis)
    model_scores: dict[str, float] | None = None


class OptimizationTracker:
    """Track optimization experiments over time using SQLite storage.

    This tracker persists optimization runs for:
    - Experiment reproducibility and auditing
    - Comparing optimization strategies
    - Finding best-performing configurations
    - Analyzing improvement trends over time

    Example:
        >>> tracker = OptimizationTracker("~/.karenina/optimization_history.db")
        >>> tracker.log_run(run)
        >>> best = tracker.get_best_run("my_benchmark")
        >>> history = tracker.list_runs("my_benchmark", limit=10)
    """

    def __init__(self, storage_path: Path | str):
        """Initialize tracker with SQLite storage.

        Args:
            storage_path: Path to SQLite database file. Will be created if
                         it doesn't exist. Parent directories will be created.
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if not exists."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    targets TEXT NOT NULL,
                    seed_prompts TEXT NOT NULL,
                    optimized_prompts TEXT NOT NULL,
                    train_score REAL NOT NULL,
                    val_score REAL NOT NULL,
                    test_score REAL,
                    improvement REAL NOT NULL,
                    reflection_model TEXT NOT NULL,
                    metric_calls INTEGER NOT NULL,
                    best_generation INTEGER NOT NULL,
                    total_generations INTEGER NOT NULL,
                    model_scores TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_benchmark_name
                ON optimization_runs(benchmark_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON optimization_runs(timestamp DESC)
            """)
            conn.commit()

    def log_run(self, run: OptimizationRun) -> str:
        """Log a completed optimization run.

        Args:
            run: OptimizationRun record to store

        Returns:
            The run_id of the logged run
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute(
                """
                INSERT INTO optimization_runs (
                    run_id, timestamp, benchmark_name, targets, seed_prompts,
                    optimized_prompts, train_score, val_score, test_score,
                    improvement, reflection_model, metric_calls,
                    best_generation, total_generations, model_scores
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.timestamp.isoformat(),
                    run.benchmark_name,
                    json.dumps(run.targets),
                    json.dumps(run.seed_prompts),
                    json.dumps(run.optimized_prompts),
                    run.train_score,
                    run.val_score,
                    run.test_score,
                    run.improvement,
                    run.reflection_model,
                    run.metric_calls,
                    run.best_generation,
                    run.total_generations,
                    json.dumps(run.model_scores) if run.model_scores else None,
                ),
            )
            conn.commit()
        return run.run_id

    def get_run(self, run_id: str) -> OptimizationRun | None:
        """Get a specific run by ID.

        Args:
            run_id: The run_id to retrieve

        Returns:
            OptimizationRun if found, None otherwise
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM optimization_runs WHERE run_id = ?",
                (run_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_run(row)
            return None

    def get_best_run(
        self,
        benchmark_name: str,
        metric: Literal["val_score", "improvement"] = "val_score",
    ) -> OptimizationRun | None:
        """Get best performing run for a benchmark.

        Args:
            benchmark_name: Name of the benchmark to filter by
            metric: Which metric to use for ranking (val_score or improvement)

        Returns:
            Best OptimizationRun if any exist, None otherwise
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f"""
                SELECT * FROM optimization_runs
                WHERE benchmark_name = ?
                ORDER BY {metric} DESC
                LIMIT 1
                """,
                (benchmark_name,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_run(row)
            return None

    def list_runs(
        self,
        benchmark_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OptimizationRun]:
        """List optimization runs, optionally filtered by benchmark.

        Args:
            benchmark_name: Optional benchmark name filter
            limit: Maximum number of runs to return
            offset: Number of runs to skip (for pagination)

        Returns:
            List of OptimizationRun records, ordered by timestamp descending
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            if benchmark_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM optimization_runs
                    WHERE benchmark_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (benchmark_name, limit, offset),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM optimization_runs
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
            return [self._row_to_run(row) for row in cursor.fetchall()]

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Compare multiple optimization runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dict with comparison data including:
            - runs: Dict mapping run_id to run data
            - metrics: Comparison of key metrics
            - best: Which run is best for each metric
        """
        runs = {}
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs[run_id] = run

        if not runs:
            return {"runs": {}, "metrics": {}, "best": {}}

        # Find best for each metric
        best_val = max(runs.items(), key=lambda x: x[1].val_score)
        best_improvement = max(runs.items(), key=lambda x: x[1].improvement)

        return {
            "runs": {
                run_id: {
                    "benchmark_name": run.benchmark_name,
                    "timestamp": run.timestamp.isoformat(),
                    "targets": run.targets,
                    "train_score": run.train_score,
                    "val_score": run.val_score,
                    "test_score": run.test_score,
                    "improvement": run.improvement,
                    "reflection_model": run.reflection_model,
                    "metric_calls": run.metric_calls,
                    "model_scores": run.model_scores,
                }
                for run_id, run in runs.items()
            },
            "metrics": {
                "val_score": {run_id: run.val_score for run_id, run in runs.items()},
                "improvement": {run_id: run.improvement for run_id, run in runs.items()},
                "metric_calls": {run_id: run.metric_calls for run_id, run in runs.items()},
            },
            "best": {
                "val_score": best_val[0],
                "improvement": best_improvement[0],
            },
        }

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID.

        Args:
            run_id: The run_id to delete

        Returns:
            True if a run was deleted, False if not found
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute(
                "DELETE FROM optimization_runs WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def export_history(
        self,
        format: Literal["json", "csv"] = "json",
        benchmark_name: str | None = None,
    ) -> str:
        """Export optimization history.

        Args:
            format: Output format ("json" or "csv")
            benchmark_name: Optional filter by benchmark name

        Returns:
            Formatted string with optimization history
        """
        runs = self.list_runs(benchmark_name=benchmark_name, limit=1000)

        if format == "json":
            return json.dumps(
                [
                    {
                        "run_id": run.run_id,
                        "timestamp": run.timestamp.isoformat(),
                        "benchmark_name": run.benchmark_name,
                        "targets": run.targets,
                        "train_score": run.train_score,
                        "val_score": run.val_score,
                        "test_score": run.test_score,
                        "improvement": run.improvement,
                        "reflection_model": run.reflection_model,
                        "metric_calls": run.metric_calls,
                        "best_generation": run.best_generation,
                        "total_generations": run.total_generations,
                        "model_scores": run.model_scores,
                    }
                    for run in runs
                ],
                indent=2,
            )
        else:  # csv
            lines = [
                "run_id,timestamp,benchmark_name,targets,train_score,val_score,"
                "test_score,improvement,reflection_model,metric_calls,"
                "best_generation,total_generations"
            ]
            for run in runs:
                lines.append(
                    f"{run.run_id},{run.timestamp.isoformat()},"
                    f'{run.benchmark_name},"{";".join(run.targets)}",'
                    f"{run.train_score},{run.val_score},"
                    f"{run.test_score if run.test_score is not None else ''},"
                    f"{run.improvement},{run.reflection_model},"
                    f"{run.metric_calls},{run.best_generation},"
                    f"{run.total_generations}"
                )
            return "\n".join(lines)

    def get_improvement_trend(
        self,
        benchmark_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get improvement trend over time for a benchmark.

        Args:
            benchmark_name: Benchmark to analyze
            limit: Number of most recent runs to include

        Returns:
            List of dicts with timestamp, val_score, improvement
        """
        runs = self.list_runs(benchmark_name=benchmark_name, limit=limit)
        return [
            {
                "run_id": run.run_id,
                "timestamp": run.timestamp.isoformat(),
                "val_score": run.val_score,
                "improvement": run.improvement,
                "targets": run.targets,
            }
            for run in reversed(runs)  # Chronological order
        ]

    def _row_to_run(self, row: sqlite3.Row) -> OptimizationRun:
        """Convert a database row to OptimizationRun."""
        return OptimizationRun(
            run_id=row["run_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            benchmark_name=row["benchmark_name"],
            targets=json.loads(row["targets"]),
            seed_prompts=json.loads(row["seed_prompts"]),
            optimized_prompts=json.loads(row["optimized_prompts"]),
            train_score=row["train_score"],
            val_score=row["val_score"],
            test_score=row["test_score"],
            improvement=row["improvement"],
            reflection_model=row["reflection_model"],
            metric_calls=row["metric_calls"],
            best_generation=row["best_generation"],
            total_generations=row["total_generations"],
            model_scores=(json.loads(row["model_scores"]) if row["model_scores"] else None),
        )
