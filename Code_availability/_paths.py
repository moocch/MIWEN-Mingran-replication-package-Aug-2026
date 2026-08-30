"""Path resolution for this replication package.

Data and code live in two sibling trees whose directory structures mirror
each other:

    Code_availability/figure_scripts/<item>/...  <->  Data_availability/<item>/...
    Code_availability/pipelines/<campaign>/...   <->  Data_availability/raw/<campaign>/...

`data_dir(__file__)` returns the data directory that belongs to the calling
script, so every script reads its inputs from — and writes its outputs to —
the matching folder in Data_availability/.
"""

from pathlib import Path

__all__ = ["code_root", "repo_root", "data_root", "data_dir", "code_dir"]


def code_root(start):
    """The Code_availability/ directory containing `start`."""
    p = Path(start).resolve()
    for q in (p, *p.parents):
        if q.name == "Code_availability":
            return q
    raise RuntimeError(f"{start} is not inside Code_availability/")


def repo_root(start):
    return code_root(start).parent


def data_root(start):
    return repo_root(start) / "Data_availability"


def data_dir(start):
    """The Data_availability/ directory mirroring the caller's own folder."""
    root = data_root(start)
    parts = Path(start).resolve().parent.relative_to(code_root(start)).parts
    if not parts:
        return root
    if parts[0] == "figure_scripts":
        return root.joinpath(*parts[1:])
    if parts[0] == "pipelines":
        return root.joinpath("raw", *parts[1:])
    return root.joinpath(*parts)


def code_dir(start):
    """The caller's own directory inside Code_availability/ (for sibling code)."""
    return Path(start).resolve().parent
