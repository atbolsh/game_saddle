"""Disk + Neo4j storage of game snapshots.

Best practice for storing images in Neo4j is **not** to put large blobs in
node properties (overflow record chains -> many extra disk I/Os). We adopt
the recommended hybrid:

  * the full-resolution PNG lives on the filesystem under
    ``<image_dir>/<session_id>/<snapshot_id>.png``;
  * a small base64-encoded PNG thumbnail (default 64x64) is stored on the
    ``GameSnapshot`` node's ``thumbnail_b64`` property, small enough to
    avoid the BLOB penalty and big enough to preview in Neo4j Browser;
  * the filesystem ``path``, ``width``, ``height`` and the full game
    ``settings_json`` are stored as ordinary properties on the same node.

``GameSnapshot`` nodes are linked to the corresponding ``Message`` node via
a ``(:Message)-[:CAPTURED_STATE]->(:GameSnapshot)`` relationship (written
through :func:`link_snapshot_to_message`).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from .config import AgentConfig, CONFIG
from .game_io import render_frame_png

logger = logging.getLogger(__name__)


def make_thumbnail_b64(src_path: str | Path, size: int = 64) -> str:
    """Read ``src_path`` and return a base64-encoded PNG thumbnail
    (nearest-power square thumbnail, preserving aspect via thumbnail())."""
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def snapshot_id() -> str:
    return uuid.uuid4().hex


async def store_snapshot(
    client: Any,
    session_id: str,
    sid: str,
    game: Any,
    settings_dict: dict[str, Any],
    cfg: AgentConfig | None = None,
    label: str = "step",
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render the current game frame to disk, write a ``GameSnapshot`` node
    to Neo4j, and return ``(path, node_dict)``.

    ``client`` is a NAMS ``MemoryClient`` with a bolt backend (so
    ``client.graph.execute_write`` is available).
    """
    cfg = cfg or CONFIG
    img_root = Path(cfg.image_dir)
    rel_path = img_root / session_id / f"{sid}.png"
    abs_path = rel_path.resolve()
    width, height = render_frame_png(game, abs_path)
    thumb_b64 = make_thumbnail_b64(abs_path, size=64)
    # Store a relative-ish path so the DB is portable across hosts.
    path_str = str(rel_path)

    props: dict[str, Any] = {
        "id": sid,
        "session_id": session_id,
        "path": path_str,
        "width": width,
        "height": height,
        "thumbnail_b64": thumb_b64,
        "settings_json": json.dumps(settings_dict),
        "label": label,
    }
    if extra:
        props.update(extra)

    await client.graph.execute_write(
        "MERGE (n:GameSnapshot {id: $id}) "
        "SET n.session_id = $session_id, n.path = $path, "
        "    n.width = $width, n.height = $height, "
        "    n.thumbnail_b64 = $thumbnail_b64, "
        "    n.settings_json = $settings_json, "
        "    n.label = $label, "
        "    n.created_at = datetime()",
        props,
    )
    return str(abs_path), props


async def link_snapshot_to_message(
    client: Any, message_id: str, snapshot_id: str, role: str = "ASSOCIATED"
) -> None:
    """Create ``(:Message {id: message_id})-[:CAPTURED_STATE {role:$role}]
    ->(:GameSnapshot {id: snapshot_id})``.

    The relationship type is ``CAPTURED_STATE``; ``role`` distinguishes
    'before' / 'after' / 'observation' snapshots.

    NAMS returns message ids as ``uuid.UUID`` objects, which the Neo4j bolt
    driver cannot serialize as query parameters. The ``Message.id`` node
    property is stored as the (dashed) string form, so we cast to ``str`` --
    this both fixes the ``ValueError: Values of type <class 'uuid.UUID'> are
    not supported`` and matches the stored id.
    """
    await client.graph.execute_write(
        "MATCH (m:Message {id: $mid}), (s:GameSnapshot {id: $sid}) "
        "MERGE (m)-[r:CAPTURED_STATE]->(s) "
        "SET r.role = $role",
        {"mid": str(message_id), "sid": str(snapshot_id), "role": role},
    )


async def fetch_session_snapshots(client: Any, session_id: str) -> list[dict[str, Any]]:
    """Return all GameSnapshot nodes for a session, ordered by creation time."""
    rows = await client.query.cypher(
        "MATCH (s:GameSnapshot {session_id: $sid}) "
        "RETURN s ORDER BY s.created_at ASC",
        {"sid": session_id},
    )
    return [dict(r["s"]) for r in rows if "s" in r]


async def fetch_messages_with_snapshots(client: Any, session_id: str) -> list[dict[str, Any]]:
    """Return messages for a session with their linked snapshots (used by
    mode 3 self-evaluation). Each row: {message, snapshots: [GameSnapshot...]}.

    NAMS does not store ``session_id`` on ``Message`` nodes; it links messages
    to a ``Conversation`` node (which carries ``session_id``) via
    ``(:Conversation)-[:HAS_MESSAGE]->(:Message)``. We therefore reach the
    session's messages through the Conversation.
    """
    rows = await client.query.cypher(
        "MATCH (c:Conversation {session_id: $sid})-[:HAS_MESSAGE]->(m:Message) "
        "OPTIONAL MATCH (m)-[:CAPTURED_STATE]->(s:GameSnapshot) "
        "RETURN m, collect(s) AS snaps "
        "ORDER BY m.created_at ASC",
        {"sid": session_id},
    )
    out = []
    for r in rows:
        m = dict(r.get("m") or {})
        snaps = [dict(s) for s in (r.get("snaps") or []) if s is not None]
        out.append({"message": m, "snapshots": snaps})
    return out


def load_image(path: str | Path) -> str:
    """Return the absolute path for an image stored by :func:`store_snapshot`.
    The Gemma processor accepts local paths directly."""
    return str(Path(path).resolve())
