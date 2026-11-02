"""Durable local job queue. SQLite transactions contain no generation or file I/O."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .job_requests import decode_request, encode_request, request_fingerprints
from .workflows.base import SavedImage, SavedVideo


STATES = ("queued", "running", "completed", "failed", "interrupted")
TERMINAL_STATES = {"completed", "failed", "interrupted"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class JobOutput:
    id: str
    kind: str
    file_path: Path
    metadata_path: Path
    seed: int
    conditioning_path: Path | None


@dataclass(frozen=True)
class JobEvent:
    stage: str
    current: int | None
    total: int | None
    created_at: str


@dataclass(frozen=True)
class Job:
    id: str
    workflow: str
    status: str
    request: dict
    fingerprints: dict
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    worker_id: str | None
    stage: str
    current: int | None
    total: int | None
    error: str | None
    outputs: tuple[JobOutput, ...]
    events: tuple[JobEvent, ...] = ()


class SubmissionConflict(ValueError):
    """A form's submission key was reused with different values."""


class JobDeletionConflict(ValueError):
    """Queued and running jobs must finish before their files can be deleted."""


class JobStore:
    """Persist resolved requests, lifecycle states, progress, and typed output references."""

    def __init__(self, database: str | Path):
        self.database = Path(database).expanduser().resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise ValueError(f"Unsupported job database schema version {version}")
            connection.execute("PRAGMA journal_mode=WAL")
        with self._transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                workflow TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN
                    ('queued','running','completed','failed','interrupted')),
                request_json TEXT NOT NULL, fingerprints_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT, worker_id TEXT,
                stage TEXT NOT NULL, current INTEGER, total INTEGER, error TEXT)""")
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(status, sequence)")
            connection.execute("""CREATE TABLE IF NOT EXISTS job_outputs (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                position INTEGER NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('image','video')),
                file_path TEXT NOT NULL, metadata_path TEXT NOT NULL, seed INTEGER NOT NULL,
                conditioning_path TEXT, UNIQUE(job_id, position))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS worker_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT NOT NULL,
                pid INTEGER NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL, stopped_at TEXT)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS job_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
                stage TEXT NOT NULL, current INTEGER, total INTEGER, created_at TEXT NOT NULL)""")
            connection.execute("CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id,sequence)")
            connection.execute("""CREATE TABLE IF NOT EXISTS job_submissions (
                key TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), request_hash TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS model_unload_requests (
                id TEXT PRIMARY KEY, worker_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','completed','failed')),
                created_at TEXT NOT NULL, finished_at TEXT, error TEXT)""")
            connection.execute("PRAGMA user_version=2")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.database), timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _read_job(connection, identifier):
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        outputs = tuple(JobOutput(item["id"], item["kind"], Path(item["file_path"]),
                                  Path(item["metadata_path"]), item["seed"],
                                  Path(item["conditioning_path"]) if item["conditioning_path"] else None)
                        for item in connection.execute("SELECT * FROM job_outputs WHERE job_id=? ORDER BY position", (identifier,)))
        events = tuple(JobEvent(item["stage"], item["current"], item["total"], item["created_at"])
                       for item in connection.execute("SELECT * FROM job_events WHERE job_id=? ORDER BY sequence", (identifier,)))
        return Job(row["id"], row["workflow"], row["status"], json.loads(row["request_json"]),
                   json.loads(row["fingerprints_json"]), row["created_at"], row["updated_at"],
                   row["started_at"], row["finished_at"], row["worker_id"], row["stage"],
                   row["current"], row["total"], row["error"], outputs, events)

    def enqueue(self, request, *, submission_key: str | None = None) -> Job:
        """Persist a previously prepared request. This does not import a model runtime."""
        document = encode_request(request)
        request_hash = hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if submission_key is not None:
            if not isinstance(submission_key, str) or not re.fullmatch(r"[0-9a-f]{32}", submission_key):
                raise ValueError("Invalid submission key")
            with self._connection() as connection:
                existing = self._submitted_job(connection, submission_key, request_hash)
                if existing is not None:
                    return existing
        frozen = decode_request(document)
        workflow = frozen.model.defaults["workflow"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", workflow):
            raise ValueError("Workflow names must contain only letters, numbers, underscores, or hyphens")
        fingerprints = request_fingerprints(frozen)
        identifier, now = uuid4().hex, _now()
        with self._transaction() as connection:
            if submission_key is not None:
                existing = self._submitted_job(connection, submission_key, request_hash)
                if existing is not None:
                    return existing
            connection.execute("""INSERT INTO jobs
                (id,workflow,status,request_json,fingerprints_json,created_at,updated_at,stage)
                VALUES (?,?,'queued',?,?,?,?,'queued')""",
                (identifier, workflow, _json(document), _json(fingerprints), now, now))
            self._event(connection, identifier, "queued")
            if submission_key is not None:
                connection.execute("INSERT INTO job_submissions VALUES (?,?,?)", (submission_key, identifier, request_hash))
            return self._read_job(connection, identifier)

    def _submitted_job(self, connection, key, request_hash):
        row = connection.execute("SELECT * FROM job_submissions WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise SubmissionConflict("This form was already submitted with different values. Reload to create another job.")
        return self._read_job(connection, row["job_id"])

    @staticmethod
    def _event(connection, job_id, stage, current=None, total=None):
        connection.execute("INSERT INTO job_events(job_id,stage,current,total,created_at) VALUES (?,?,?,?,?)",
                           (job_id, stage, current, total, _now()))

    def get(self, identifier: str) -> Job:
        with self._connection() as connection:
            connection.execute("BEGIN")
            return self._read_job(connection, identifier)

    def list_jobs(self, *, status: str | None = None, limit=100) -> tuple[Job, ...]:
        if status is not None and status not in STATES:
            raise ValueError("Unknown job status")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        with self._connection() as connection:
            connection.execute("BEGIN")
            rows = connection.execute("SELECT id FROM jobs " + ("WHERE status=? " if status else "")
                                      + "ORDER BY sequence DESC LIMIT ?", (status, limit) if status else (limit,)).fetchall()
            return tuple(self._read_job(connection, row["id"]) for row in rows)

    def worker_status(self) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM worker_state WHERE singleton=1").fetchone()
            return dict(row) if row else None

    def submission_keys(self, identifier: str) -> tuple[str, ...]:
        with self._connection() as connection:
            return tuple(row[0] for row in connection.execute(
                "SELECT key FROM job_submissions WHERE job_id=?", (identifier,)))

    def delete_finished(self, identifier: str) -> None:
        """Delete a terminal job's database records after its files are removed."""
        with self._transaction() as connection:
            job = self._read_job(connection, identifier)
            if job.status not in TERMINAL_STATES:
                raise JobDeletionConflict("Wait for this job to finish before deleting it.")
            for table in ("job_outputs", "job_events", "job_submissions"):
                connection.execute(f"DELETE FROM {table} WHERE job_id=?", (identifier,))
            connection.execute("DELETE FROM jobs WHERE id=?", (identifier,))

    def request_model_unload(self) -> dict:
        """Ask the current worker to release models on its owner thread."""
        with self._transaction() as connection:
            worker = connection.execute("SELECT * FROM worker_state WHERE singleton=1").fetchone()
            if worker is None or worker["status"] != "running" or (
                datetime.now(timezone.utc) - datetime.fromisoformat(worker["heartbeat_at"])
            ).total_seconds() > 15:
                raise ValueError("The generation worker is unavailable. Start it before clearing models.")
            pending = connection.execute(
                "SELECT * FROM model_unload_requests WHERE worker_id=? AND status='pending'",
                (worker["id"],),
            ).fetchone()
            if pending is not None:
                return dict(pending)
            identifier = uuid4().hex
            connection.execute("""INSERT INTO model_unload_requests(id,worker_id,status,created_at)
                VALUES (?,?,'pending',?)""", (identifier, worker["id"], _now()))
            return dict(connection.execute("SELECT * FROM model_unload_requests WHERE id=?", (identifier,)).fetchone())

    def model_unload_status(self, identifier) -> dict:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM model_unload_requests WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            return dict(row)

    def _pending_model_unloads(self, worker_id):
        with self._connection() as connection:
            return [row[0] for row in connection.execute(
                "SELECT id FROM model_unload_requests WHERE worker_id=? AND status='pending'", (worker_id,))]

    def _finish_model_unloads(self, worker_id, identifiers, error=None):
        with self._transaction() as connection:
            connection.executemany("""UPDATE model_unload_requests SET status=?,finished_at=?,error=?
                WHERE id=? AND worker_id=? AND status='pending'""",
                [("failed" if error else "completed", _now(), error, identifier, worker_id) for identifier in identifiers])

    # Worker mutations are private: the Worker holds the OS lock throughout them.
    def _start_worker(self, identifier, pid):
        now = _now()
        with self._transaction() as connection:
            connection.execute("""UPDATE model_unload_requests SET status='failed',finished_at=?,
                error='The previous worker exited before confirming model cleanup. Try again.' WHERE status='pending'""", (now,))
            for row in connection.execute("SELECT id FROM jobs WHERE status='running'").fetchall():
                self._event(connection, row["id"], "interrupted")
            connection.execute("""UPDATE jobs SET status='interrupted', stage='interrupted',
                error='Previous worker exited before completion; job was not retried.',
                updated_at=?, finished_at=? WHERE status='running'""", (now, now))
            connection.execute("""INSERT INTO worker_state
                (singleton,id,pid,status,started_at,heartbeat_at,stopped_at) VALUES (1,?,?,'running',?,?,NULL)
                ON CONFLICT(singleton) DO UPDATE SET id=excluded.id,pid=excluded.pid,status='running',
                started_at=excluded.started_at,heartbeat_at=excluded.heartbeat_at,stopped_at=NULL""",
                (identifier, pid, now, now))

    def _heartbeat(self, identifier):
        with self._transaction() as connection:
            count = connection.execute("UPDATE worker_state SET heartbeat_at=? WHERE id=? AND status='running'",
                                       (_now(), identifier)).rowcount
            if count != 1:
                raise RuntimeError("Worker no longer owns the queue")

    def _stop_worker(self, identifier):
        with self._transaction() as connection:
            connection.execute("""UPDATE model_unload_requests SET status='failed',finished_at=?,
                error='The worker stopped before confirming model cleanup. Restart it and try again.'
                WHERE worker_id=? AND status='pending'""", (_now(), identifier))
            connection.execute("UPDATE worker_state SET status='stopped', stopped_at=?, heartbeat_at=? WHERE id=?",
                               (_now(), _now(), identifier))

    def _claim_next(self, worker_id) -> Job | None:
        with self._transaction() as connection:
            row = connection.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY sequence LIMIT 1").fetchone()
            if row is None:
                return None
            now = _now()
            connection.execute("""UPDATE jobs SET status='running', stage='validating', worker_id=?,
                started_at=?, updated_at=? WHERE id=? AND status='queued'""", (worker_id, now, now, row["id"]))
            self._event(connection, row["id"], "validating")
            return self._read_job(connection, row["id"])

    @staticmethod
    def _require_owner(connection, job_id, worker_id):
        row = connection.execute("SELECT status,worker_id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["status"] != "running" or row["worker_id"] != worker_id:
            raise RuntimeError("Job is not running under this worker")

    def _progress(self, job_id, worker_id, stage, current=None, total=None):
        if not isinstance(stage, str) or not stage:
            raise ValueError("Progress needs a stage")
        if any(value is not None and (type(value) is not int or value < 0) for value in (current, total)):
            raise ValueError("Progress counts must be nonnegative integers or null")
        if current is not None and total is not None and current > total:
            raise ValueError("Progress current must not exceed total")
        with self._transaction() as connection:
            self._require_owner(connection, job_id, worker_id)
            previous = connection.execute("SELECT stage FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            if previous != stage:
                self._event(connection, job_id, stage, current, total)
            else:
                connection.execute("""UPDATE job_events SET current=?,total=? WHERE sequence=
                    (SELECT max(sequence) FROM job_events WHERE job_id=?)""", (current, total, job_id))
            connection.execute("UPDATE jobs SET stage=?,current=?,total=?,updated_at=? WHERE id=?",
                               (stage, current, total, _now(), job_id))

    def _record_outputs(self, job_id, worker_id, saved):
        with self._transaction() as connection:
            self._require_owner(connection, job_id, worker_id)
            position = connection.execute("SELECT count(*) FROM job_outputs WHERE job_id=?", (job_id,)).fetchone()[0]
            for output in saved:
                if not isinstance(output, (SavedImage, SavedVideo)):
                    raise ValueError("Unknown saved output type")
                video = isinstance(output, SavedVideo)
                connection.execute("INSERT INTO job_outputs VALUES (?,?,?,?,?,?,?,?)", (
                    uuid4().hex, job_id, position, "video" if video else "image",
                    str(output.video_path if video else output.image_path), str(output.metadata_path), output.seed,
                    str(output.conditioning_path) if video else None,
                ))
                position += 1

    def _finish(self, job_id, worker_id, status, error=None):
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("Invalid terminal status")
        with self._transaction() as connection:
            self._require_owner(connection, job_id, worker_id)
            now = _now()
            count = connection.execute("SELECT count(*) FROM job_outputs WHERE job_id=?", (job_id,)).fetchone()[0]
            if status == "completed" and not count:
                raise ValueError("A completed job must have at least one output")
            self._event(connection, job_id, status, count, count if status == "completed" else None)
            connection.execute("""UPDATE jobs SET status=?,stage=?,error=?,updated_at=?,finished_at=?,
                current=?,total=? WHERE id=?""", (status, status, error, now, now, count,
                                               count if status == "completed" else None, job_id))
            # Return the final snapshot while still holding the transaction:
            # the history UI may delete this record immediately after commit.
            return self._read_job(connection, job_id)
