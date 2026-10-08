# harv-hq specification

This is the authoritative spec for `harv-hq` (import name `harv_hq`). It plays
the same role for hq that `docs/spec.md` plays for harv: code, docstrings, and
tests follow it, and any public API must be documented here first.

## Scope

hq handles the parts of a harv analysis that sit around a single fit:

1. Pipelining harv runs over large samples, driven by a config file.
1. Visualizing the outputs.
1. Preparing input catalogs into a form harv can read.

Single-source modeling and sampling stay in harv.

## Packaging

- PyPI distribution `harv-hq`, import package `harv_hq`.
- Lives at `packages/hq/` in the harv repository as a uv workspace member.
- Released in lockstep with harv from the same git tag.

## Public API

None yet.
