"""Unit tests for extensions.base (ParamInfo + AbstractExtension ABC)."""

import jax.numpy as jnp
import pytest

from harv.extensions.base import AbstractExtension, ParamInfo


class TestParamInfo:
    def test_create_nonlinear(self):
        p = ParamInfo("period", "time")
        assert p.name == "period"
        assert p.unit == "time"
        assert not p.linear

    def test_create_linear(self):
        p = ParamInfo("rv_semiamp", "speed", linear=True)
        assert p.name == "rv_semiamp"
        assert p.linear

    def test_frozen(self):
        p = ParamInfo("period", "time")
        with pytest.raises(AttributeError):
            p.name = "other"  # type: ignore[misc]

    def test_dot_in_name_rejected(self):
        with pytest.raises(ValueError, match="must not contain"):
            ParamInfo("comp.period", "time")

    def test_dimensionless_unit(self):
        p = ParamInfo("eccentricity", "")
        assert p.unit == ""


class TestAbstractExtension:
    """Verify that subclassing AbstractExtension works correctly."""

    def test_subclass_with_extra_params(self):
        class _Dummy(AbstractExtension):
            def extra_params(self):
                return ()

        d = _Dummy()
        assert isinstance(d, AbstractExtension)
        assert d.extra_params() == ()

    def test_default_modify_design_matrix(self):
        class _Dummy(AbstractExtension):
            def extra_params(self):
                return ()

        d = _Dummy()
        X = jnp.ones((3, 2))
        assert d.modify_design_matrix(X, None, {}) is X

    def test_default_modify_covariance(self):
        class _Dummy(AbstractExtension):
            def extra_params(self):
                return ()

        d = _Dummy()
        cov = jnp.ones(5)
        assert d.modify_covariance(cov, None, {}) is cov

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractExtension()
