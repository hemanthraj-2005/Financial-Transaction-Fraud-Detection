"""Cached data access for the dashboard. The cache refreshes automatically when a file changes."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import DB_PATH, GRAPH_METRICS_STEM
from dashboard import loaders
from src.utils.io_utils import table_paths


def file_version(path) -> float:
    path = Path(path)
    return path.stat().st_mtime if path.exists() else 0.0


def table_version(stem) -> float:
    return max(file_version(p) for p in table_paths(stem))


def status() -> tuple[bool, str]:
    return loaders.db_status(DB_PATH)


@st.cache_data(show_spinner=False, max_entries=256)
def _run(name: str, version: float, args: tuple, kwargs_items: tuple):
    return loaders.run_query(DB_PATH, name, *args, **dict(kwargs_items))


def q(name: str, *args, **kwargs):
    """Run a registered database query (cached until the database file changes)."""
    return _run(name, file_version(DB_PATH), tuple(args), tuple(sorted(kwargs.items())))


@st.cache_data(show_spinner=False)
def _read_json(path: str, version: float):
    return loaders.read_json(path)


def read_json(path):
    return _read_json(str(path), file_version(path))


@st.cache_data(show_spinner=False)
def _read_csv(path: str, version: float):
    return loaders.read_csv(path)


def read_csv(path):
    return _read_csv(str(path), file_version(path))


@st.cache_data(show_spinner=False)
def _graph_metrics(version: float):
    return loaders.load_optional_table(GRAPH_METRICS_STEM)


def graph_metrics():
    return _graph_metrics(table_version(GRAPH_METRICS_STEM))