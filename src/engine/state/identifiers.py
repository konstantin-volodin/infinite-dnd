"""Shared normalization for generated world-state identifiers."""

import re
from datetime import datetime
from uuid import uuid4


def new_run_id() -> str:
    """Return a sortable, collision-resistant id for a campaign run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return f"{timestamp}_{uuid4().hex[:8]}"


def slugify(text: str) -> str:
    text = text.strip().lower().replace("_", "-")
    text = re.sub(r"[^a-z0-9\- ]+", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text.strip("-")
