"""
core/database.py
===================
Offline-first persistence layer. Stores generated campaign briefs
(including both static and Seedance motion prompts) in a local SQLite
database so the Streamlit app can offer a "History" tab with zero
external infrastructure -- satisfying the "Offline-first architecture
running on a local SQLite/JSON database" requirement.

The schema intentionally stores the full concept payload as JSON text in
a single column per campaign rather than fully normalizing across many
tables: campaigns are read far more often as a whole document than
queried by individual sub-field, and this keeps the module dependency-free
(stdlib `sqlite3` + `json` only) and trivially portable across Windows,
macOS, Linux, and the Docker container.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "visionforge.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    brand_name TEXT NOT NULL,
    product_category TEXT,
    brief_idea TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_campaigns_brand ON campaigns (brand_name);
CREATE INDEX IF NOT EXISTS idx_campaigns_created ON campaigns (created_at DESC);
"""


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class CampaignDatabase:
    """Thin, dependency-free SQLite wrapper for campaign persistence."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def save_campaign(
        self,
        brand_name: str,
        product_category: str,
        brief_idea: str,
        payload: Dict[str, Any],
        campaign_id: Optional[str] = None,
    ) -> str:
        """Persists a full campaign payload (concepts + static prompts +
        seedance motion prompts, already rendered to plain dicts) and
        returns the generated campaign id."""
        campaign_id = campaign_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, default=_json_default, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO campaigns (id, created_at, brand_name, product_category, brief_idea, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (campaign_id, created_at, brand_name, product_category, brief_idea, payload_json),
            )
        return campaign_id

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def list_campaigns(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, brand_name, product_category, brief_idea
                FROM campaigns
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        return record

    def delete_campaign(self, campaign_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
        return cur.rowcount > 0

    def clear_all(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM campaigns")
        return cur.rowcount
