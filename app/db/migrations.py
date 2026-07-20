"""Filesystem SQL migrations, applied by ``PgVectorStore.ensure_schema``.

Migrations live in ``migrations/NNN_name.sql`` at the repo root and run in
filename order. Applied versions are tracked in a ``schema_migrations`` table,
and every file is written to be idempotent (``IF NOT EXISTS`` everywhere), so
re-running against a database bootstrapped by an older build is safe.

The files are plain SQL with one convention: the literal ``{dim}`` marker is
replaced with the embedding dimension at apply time (plain string replacement,
so ``'{}'::jsonb`` and friends need no escaping).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

_FILENAME_RE = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")

CREATE_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""

IS_APPLIED_SQL = "SELECT 1 FROM schema_migrations WHERE version = $1"

# ON CONFLICT DO NOTHING keeps the tracking insert idempotent even if two
# instances race to apply the same migration.
RECORD_APPLIED_SQL = """
INSERT INTO schema_migrations (version) VALUES ($1)
ON CONFLICT (version) DO NOTHING
"""


@dataclass(frozen=True)
class Migration:
    version: str  # file stem, e.g. "001_init"; also the tracking-table key
    sql: str

    def sql_for(self, dim: int) -> str:
        """Return the SQL with the ``{dim}`` marker substituted."""
        return self.sql.replace("{dim}", str(int(dim)))


def load_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Load all migrations from ``directory``, sorted by filename.

    Every non-hidden file must match ``NNN_name.sql`` and numbers must be
    unique — anything else raises a clear ``ValueError`` instead of being
    silently skipped, so a typo cannot leave the schema half-applied.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"migrations directory not found: {directory}")
    migrations: list[Migration] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue  # editor/OS droppings (.DS_Store etc.)
        match = _FILENAME_RE.match(path.name)
        if not match:
            raise ValueError(
                f"unrecognized file in migrations dir (expected NNN_name.sql): {path.name!r}"
            )
        number = match.group(1)
        if number in seen:
            raise ValueError(
                f"duplicate migration number {number}: {seen[number]!r} vs {path.name!r}"
            )
        seen[number] = path.name
        migrations.append(Migration(version=path.stem, sql=path.read_text(encoding="utf-8")))
    return migrations
