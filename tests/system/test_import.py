# MIT License
#
# Copyright (c) 2025 adrn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Test that the package can be imported and that ``__all__`` is honest."""

import types

import pytest

import harv


def test_import():
    """Test that the package can be imported."""
    assert harv.__version__


@pytest.mark.parametrize("name", harv.__all__)
def test_every_all_entry_resolves(name):
    """Nothing in ``__all__`` may be a name the package does not actually bind."""
    assert hasattr(harv, name), f"harv.__all__ lists {name!r}, which does not exist"


def test_every_public_name_is_declared():
    """The reverse direction: no public name may be exported without declaring it.

    Submodules are exempt -- Python binds a submodule on its parent package as a
    side effect of any ``from harv.x import y``, so their presence in ``vars()``
    is not a curation decision.  ``harv.__all__`` lists the submodules that *are*
    part of the public surface explicitly (``data``, ``periodogram``, ``plot``).
    """
    undeclared = {
        name
        for name, value in vars(harv).items()
        if not name.startswith("_")
        and name not in harv.__all__
        and not isinstance(value, types.ModuleType)
    }
    assert not undeclared, (
        f"public names missing from harv.__all__: {sorted(undeclared)}"
    )


def test_plotting_and_io_are_not_top_level():
    """Plotting and serialization live in their own modules, not on ``harv``.

    ``docs/spec.md`` used to show ``harv.plot_rv`` / ``harv.save_sampler``; the
    supported spellings are ``harv.plot.plot_rv`` and
    ``from harv.io import save_sampler``.
    """
    for name in ("plot_rv", "plot_gaia_astrometry", "save_sampler", "load_sampler"):
        assert not hasattr(harv, name), f"harv.{name} should not be top-level"

    from harv.io import load_sampler, save_sampler  # noqa: F401, PLC0415

    assert callable(harv.plot.plot_rv)
