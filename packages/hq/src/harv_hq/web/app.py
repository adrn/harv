"""``create_app``: the FastAPI app behind ``hq serve``.

See ``packages/hq/docs/spec.md``, "Web viewer". Pages are Jinja2 templates
whose scripts fetch the JSON API; Plotly.js comes from a CDN.
"""

__all__ = ("create_app",)

import hashlib
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, PositiveInt

from harv_hq._parquet_io import read_parquet
from harv_hq.prepare import CATALOG_FILE
from harv_hq.run import STAGES, Run
from harv_hq.summarize import SUMMARY_FILE
from harv_hq.web.payload import (
    _numeric,
    column_info,
    filter_mask,
    jsonable,
    source_payload,
)

_HERE = Path(__file__).parent


@final
class RowsRequest(BaseModel):
    """Body of ``POST /api/rows``.

    The rows of ``source_ids``, sorted by ``sort`` (nulls last) when given,
    and only the first ``limit``.
    """

    source_ids: list[Any]
    columns: list[str]
    sort: str | None = None
    descending: bool = False
    limit: PositiveInt | None = None


Stamp = tuple[int, int] | None  # the summary's (inode, mtime in ns), or None


def _stamp(path: Path) -> Stamp:
    """What identifies one version of ``path``; ``None`` when it is absent.

    The summary is written to a temporary file and renamed, so every write
    gives a new inode, even within the filesystem's timestamp resolution.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_mtime_ns


@final
@dataclass(frozen=True)
class _Snapshot:
    """One version of the run's table and results index, never changed.

    ``summary.parquet`` already holds the catalog columns; before
    ``hq summarize`` has run, the table is ``catalog.parquet`` alone.
    """

    stamp: Stamp
    table: pa.Table
    units: dict[str, str]
    positions: dict[Any, int]
    int_ids: bool
    run: Run

    @classmethod
    def load(cls, run: Run) -> "_Snapshot":
        """Read the current table, and index each stage's results now.

        Indexing here, not on first use, keeps the table and the results of
        one snapshot from the same moment.
        """
        summary = run.run_dir / SUMMARY_FILE
        stamp = _stamp(summary)
        try:
            contents = read_parquet(
                summary if stamp is not None else run.run_dir / CATALOG_FILE
            )
        except FileNotFoundError:
            if stamp is None:
                raise
            # Deleted since the stamp: the catalog alone, as before summarize.
            stamp, contents = None, read_parquet(run.run_dir / CATALOG_FILE)
        table = contents.table
        fresh = run._fresh()
        for stage in STAGES:
            fresh._index(stage)
        return cls(
            stamp=stamp,
            table=table,
            units=contents.units,
            positions={sid: i for i, sid in enumerate(table["source_id"].to_pylist())},
            int_ids=pa.types.is_integer(table.schema.field("source_id").type),
            run=fresh,
        )

    def source_id(self, text: str) -> Any:
        """The ID as stored, from its URL form; 404 if not in the run."""
        try:
            source_id = int(text) if self.int_ids else text
        except ValueError:
            source_id = None
        if source_id not in self.positions:
            raise HTTPException(404, f"no source {text!r} in this run")
        return source_id

    def columns(self, names: list[str], *, numeric: bool = False) -> list[str]:
        """``names`` checked against the table.

        400 for an unknown column, or with ``numeric``, for one that cannot
        be plotted.
        """
        if unknown := [n for n in names if n not in self.table.column_names]:
            raise HTTPException(400, f"unknown columns {unknown}")
        schema = self.table.schema
        if numeric and (
            text := [n for n in names if not _numeric(schema.field(n).type)]
        ):
            raise HTTPException(400, f"columns {text} are not numeric")
        return names

    @property
    def version(self) -> str:
        """Names this snapshot, so the browser can tell when its IDs are stale."""
        return "catalog" if self.stamp is None else "{}-{}".format(*self.stamp)

    def summary_row(self, source_id: Any) -> dict[str, Any] | None:
        """The source's row of this snapshot's summary (``None`` before one)."""
        if self.stamp is None:
            return None
        return self.table.slice(self.positions[source_id], 1).to_pylist()[0]


@final
class _Viewer:
    """The run as the app sees it.

    ``hq.toml`` and the model file are loaded once, so the curves always use
    the model the server started with. The table and results index are
    reloaded when the summary changes: a new :class:`_Snapshot` is built
    under a lock and swapped in whole, so a request (FastAPI runs them on a
    thread pool) never sees a half-updated one.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run = Run(run_dir)
        self.model = self.run.model_file.setup(self.run.config.run.kind)[1]
        self._lock = threading.Lock()
        self._snapshot = _Snapshot.load(self.run)

    def current(self) -> _Snapshot:
        """The snapshot for the summary as it is now."""
        snapshot = self._snapshot
        if _stamp(self.run.run_dir / SUMMARY_FILE) == snapshot.stamp:
            return snapshot
        with self._lock:
            # Another request may have reloaded while this one waited.
            if _stamp(self.run.run_dir / SUMMARY_FILE) != self._snapshot.stamp:
                self._snapshot = _Snapshot.load(self.run)
            return self._snapshot


def _columns_json(rows: pa.Table) -> dict[str, list[Any]]:
    """Columns as JSON arrays, with IDs as strings."""
    # 19-digit Gaia IDs exceed the integers a browser represents exactly.
    return {
        name: jsonable(column.cast(pa.string()) if name == "source_id" else column)
        for name, column in zip(rows.column_names, rows.columns, strict=True)
    }


def _sort_indices(column: pa.ChunkedArray, *, descending: bool) -> pa.Array:
    """The order that sorts ``column``, nulls last either way."""
    return pc.array_sort_indices(  # ty: ignore[unresolved-attribute]
        column,
        order="descending" if descending else "ascending",
        null_placement="at_end",
    )


def _plot_columns(rows: pa.Table, positions: np.ndarray) -> bytes:
    """The plot's columns as float32, then each row's table position as int32.

    Little-endian, one array after another, so the browser reads them with
    typed arrays instead of parsing JSON. Nulls and non-finite values are NaN,
    which Plotly leaves out. float32 halves the transfer and still resolves
    far more than a screen's pixels.

    Examples
    --------
    >>> import numpy as np, pyarrow as pa
    >>> raw = _plot_columns(pa.table({"g": [1.5, None]}), np.array([0, 4]))
    >>> np.frombuffer(raw[:8], "<f4").tolist(), np.frombuffer(raw[8:], "<i4").tolist()
    ([1.5, nan], [0, 4])
    """
    arrays = []
    for column in rows.columns:
        values = column.cast(pa.float64()).to_numpy(zero_copy_only=False)
        arrays.append(np.where(np.isfinite(values), values, np.nan).astype("<f4"))
    arrays.append(positions.astype("<i4"))
    return b"".join(a.tobytes() for a in arrays)


def _static_version(directory: Path) -> str:
    """A short hash of every file under ``directory``, names and contents."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def create_app(run_dir: str | os.PathLike) -> FastAPI:
    """The web viewer for one run directory.

    Parameters
    ----------
    run_dir
        The run directory (holding ``hq.toml`` and ``catalog.parquet``).

    Returns
    -------
        The app; ``hq serve`` runs it with uvicorn.

    Raises
    ------
    FileNotFoundError
        If the run has not been prepared (no ``catalog.parquet``).
    """
    viewer = _Viewer(Path(run_dir))
    app = FastAPI(title=f"hq: {viewer.run.config.run.name}")
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    # Asset URLs carry a hash of the static files, so a browser fetches them
    # again after an hq update instead of running a cached, older script
    # against the new API.
    assets = {"static_version": _static_version(_HERE / "static")}
    templates = Jinja2Templates(
        directory=_HERE / "templates", context_processors=[lambda _: assets]
    )

    # Responses are built with JSONResponse, which refuses NaN, rather than
    # through FastAPI's encoder, which walks every element of large arrays.

    state = viewer.current

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "index.html", {"run_name": viewer.run.config.run.name}
        )

    # ":path" because a "/" in an ID is decoded before routing.
    @app.get("/source/{source_id:path}", response_class=HTMLResponse)
    def source_page(request: Request, source_id: str) -> HTMLResponse:
        state().source_id(source_id)
        return templates.TemplateResponse(
            request,
            "source.html",
            {"run_name": viewer.run.config.run.name, "source_id": source_id},
        )

    @app.get("/api/columns")
    def api_columns() -> list[dict[str, Any]]:
        current = state()
        return column_info(current.table, current.units)

    @app.get("/api/ids")
    def api_ids() -> JSONResponse:
        current = state()
        ids = jsonable(current.table["source_id"].cast(pa.string()))
        return JSONResponse(ids, headers={"X-HQ-Version": current.version})

    @app.get("/api/catalog")
    def api_catalog(
        columns: str = "",
        filter_: Annotated[str, Query(alias="filter")] = "",
        status: str = "",
    ) -> Response:
        current = state()
        names = current.columns([c for c in columns.split(",") if c], numeric=True)
        try:
            mask = filter_mask(current.table, filter_, status)
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        positions = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        rows = current.table.select(names).take(positions)
        return Response(
            _plot_columns(rows, positions),
            media_type="application/octet-stream",
            headers={"X-HQ-Version": current.version},
        )

    @app.post("/api/rows")
    def api_rows(body: RowsRequest) -> JSONResponse:
        current = state()
        names = current.columns(body.columns)
        sort = [] if body.sort is None else current.columns([body.sort])
        ids = [current.source_id(str(sid)) for sid in body.source_ids]
        rows = current.table.select(list(dict.fromkeys([*names, *sort])))
        rows = rows.take([current.positions[i] for i in ids])
        if sort:
            rows = rows.take(_sort_indices(rows[sort[0]], descending=body.descending))
        return JSONResponse(_columns_json(rows.select(names).slice(0, body.limit)))

    @app.get("/api/source/{source_id:path}")
    def api_source(source_id: str) -> JSONResponse:
        current = state()
        key = current.source_id(source_id)
        # The summary row of this snapshot, not one reread from the file,
        # which may have been replaced since.
        result = replace(current.run.load_source(key), summary=current.summary_row(key))
        return JSONResponse(source_payload(result, viewer.model))

    return app
