"""Integrity checks on the notebooks under ``docs/``.

The case-study notebooks ship their stored outputs (``docs/conf.py`` sets
``nb_execution_mode = "off"``, and ``.pre-commit-config.yaml`` excludes them
from ``nbstripout``), so their base64 image payloads are source-controlled
content that the Sphinx build decodes.  A corrupted payload fails the docs
build with a bare ``binascii.Error``, pointing at the notebook but not at the
cause -- and ``binascii`` *silently ignores* invalid characters unless
``strict_mode=True``, so a payload can decode to garbage without raising at
all.

These tests exist because a repo-wide identifier rename once rewrote `P0` to
`period_ref` inside a PNG payload: in base64, `+` and `/` are not word
characters, so a `\\bP0\\b` pattern matched `+P0+`.  Any tool that rewrites
``.ipynb`` files as text can do this again.
"""

import base64
import binascii
import json
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"

# Standard base64 alphabet plus the whitespace notebooks may wrap payloads with.
# Notably absent: "_" and "-", which belong to the *URL-safe* alphabet and are
# the tell-tale of a text substitution having landed inside a payload.
B64_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
)

MAGIC = {
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
    "image/gif": b"GIF8",
}


def _image_payloads():
    """Yield ``(notebook, cell_index, mime, payload)`` for every stored image."""
    for path in sorted(DOCS.rglob("*.ipynb")):
        if "_build" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook.get("cells", [])):
            for output in cell.get("outputs") or []:
                for mime, value in (output.get("data") or {}).items():
                    if not mime.startswith("image/") or mime.endswith("+xml"):
                        continue
                    payload = "".join(value) if isinstance(value, list) else value
                    yield path, index, mime, payload


PAYLOADS = list(_image_payloads())
IDS = [f"{p.name}::cell{i}::{m}" for p, i, m, _ in PAYLOADS]


def test_docs_contain_notebooks_with_stored_images():
    """Guard the guard: these tests are vacuous if the walk finds nothing."""
    assert PAYLOADS, f"no stored notebook images found under {DOCS}"


@pytest.mark.parametrize(("path", "index", "mime", "payload"), PAYLOADS, ids=IDS)
def test_stored_image_is_valid_base64(path, index, mime, payload):
    """Every stored image must be strictly-valid base64 of a real image.

    ``strict_mode=True`` matters: the lenient default discards characters
    outside the alphabet, so a corrupted payload can decode to the wrong bytes
    without raising.
    """
    intruders = sorted(set(payload) - B64_CHARS)
    assert not intruders, (
        f"{path.name} cell {index} ({mime}) has non-base64 character(s) "
        f"{intruders} in its payload -- most likely a text substitution that "
        f"matched inside the payload. '_' or '-' means exactly that."
    )

    try:
        blob = binascii.a2b_base64(payload, strict_mode=True)
    except binascii.Error as exc:  # pragma: no cover -- only on a corrupt file
        pytest.fail(f"{path.name} cell {index} ({mime}) is not valid base64: {exc}")

    expected = MAGIC.get(mime)
    if expected is not None:
        assert blob.startswith(expected), (
            f"{path.name} cell {index} declares {mime} but the decoded bytes "
            f"start {blob[:8]!r}, not {expected!r}"
        )

    # Round-trip: re-encoding must reproduce the payload, so the check also
    # catches a payload that decodes but was silently truncated or re-padded.
    assert base64.b64encode(blob).decode() == "".join(payload.split())
