"""Read-only access to packaged interface templates and static assets."""

from pathlib import Path


_ROOT = Path(__file__).parent
_STATIC = _ROOT / "static"

CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def read_asset(relative_path: str) -> str:
    """Read a packaged interface asset as UTF-8 text."""
    return (_ROOT / relative_path).read_text(encoding="utf-8")


def static_asset(name: str) -> tuple[bytes, str] | None:
    """Return a whitelisted top-level static asset and its content type."""
    if Path(name).name != name:
        return None
    path = _STATIC / name
    content_type = CONTENT_TYPES.get(path.suffix)
    if content_type is None or not path.is_file():
        return None
    return path.read_bytes(), content_type
