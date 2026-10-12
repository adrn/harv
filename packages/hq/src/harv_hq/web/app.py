"""``create_app``: the FastAPI app behind ``hq serve``.

See ``packages/hq/docs/spec.md``, "Web viewer". Pages are Jinja2 templates
whose scripts fetch the JSON API; Plotly.js comes from a CDN.
"""

__all__ = ("create_app",)

import os
from pathlib import Path
from typing import Annotated, Any, final

import pyarrow as pa
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from harv_hq._parquet_io import read_parquet
from harv_hq.prepare import CATALOG_FILE
from harv_hq.run import Run
from harv_hq.summarize import SUMMARY_FILE
from harv_hq.web.payload import column_info, filter_mask, jsonable, source_payload

_HERE = Path(__file__).parent


class RowsRequest(BaseModel):
    """Body of ``POST /api/rows``."""

    source_ids: list[Any]
    columns: list[str]


@final
class _Viewer:
    """The run as the app sees it: the joined table, reloaded with the summary.

    ``summary.parquet`` already holds the catalog columns; before
    ``hq summarize`` has run, the table is ``catalog.parquet`` alone.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._mtime: float | None = -1.0
        self.reload()

    def reload(self) -> None:
        """Reread the table, and start a fresh ``Run`` (and results index)."""
        summary = self.run_dir / SUMMARY_FILE
        mtime = summary.stat().st_mtime if summary.exists() else None
        if mtime == self._mtime:
            return
        contents = read_parquet(summary if mtime else self.run_dir / CATALOG_FILE)
        self.table: pa.Table = contents.table
        self.units = contents.units
        self.positions = {
            sid: i for i, sid in enumerate(self.table["source_id"].to_pylist())
        }
        self._int_ids = pa.types.is_integer(self.table.schema.field("source_id").type)
        self.run = Run(self.run_dir)
        self.model = self.run.model_file.setup(self.run.config.run.kind)[1]
        self._mtime = mtime

    def source_id(self, text: str) -> Any:
        """The ID as stored, from its URL form; 404 if not in the run."""
        try:
            source_id = int(text) if self._int_ids else text
        except ValueError:
            source_id = None
        if source_id not in self.positions:
            raise HTTPException(404, f"no source {text!r} in this run")
        return source_id

    def columns(self, names: list[str]) -> list[str]:
        """``names`` checked against the table; 400 for an unknown one."""
        if unknown := [n for n in names if n not in self.table.column_names]:
            raise HTTPException(400, f"unknown columns {unknown}")
        return names


def _columns_json(rows: pa.Table) -> dict[str, list[Any]]:
    """Columns as JSON arrays, with IDs as strings."""
    # 19-digit Gaia IDs exceed the integers a browser represents exactly.
    return {
        name: jsonable(column.cast(pa.string()) if name == "source_id" else column)
        for name, column in zip(rows.column_names, rows.columns, strict=True)
    }


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
    templates = Jinja2Templates(directory=_HERE / "templates")

    # Responses are built with JSONResponse, which refuses NaN, rather than
    # through FastAPI's encoder, which walks every element of large arrays.

    def state() -> _Viewer:
        viewer.reload()
        return viewer

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

    @app.get("/api/catalog")
    def api_catalog(
        columns: str = "source_id",
        filter_: Annotated[str, Query(alias="filter")] = "",
        status: str = "",
    ) -> JSONResponse:
        current = state()
        names = current.columns([c for c in columns.split(",") if c])
        try:
            mask = filter_mask(current.table, filter_, status)
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        rows = current.table.select(names).filter(mask)
        return JSONResponse(_columns_json(rows))

    @app.post("/api/rows")
    def api_rows(body: RowsRequest) -> JSONResponse:
        current = state()
        names = current.columns(body.columns)
        ids = [current.source_id(str(sid)) for sid in body.source_ids]
        rows = current.table.select(names).take([current.positions[i] for i in ids])
        return JSONResponse(_columns_json(rows))

    @app.get("/api/source/{source_id:path}")
    def api_source(source_id: str) -> JSONResponse:
        current = state()
        result = current.run.load_source(current.source_id(source_id))
        return JSONResponse(source_payload(result, current.model))

    return app
