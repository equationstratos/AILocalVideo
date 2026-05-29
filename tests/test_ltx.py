"""Tests des contraintes de dimensions/frames du backend LTX."""

from backend.generation.ltx_backend import _snap_dim, _snap_frames


def test_snap_frames_is_8k_plus_1():
    for n in (1, 9, 50, 120, 121, 200, 257):
        snapped = _snap_frames(n)
        assert (snapped - 1) % 8 == 0
        assert snapped >= 9


def test_snap_dim_multiple_of_32():
    for x in (10, 64, 500, 768, 769):
        snapped = _snap_dim(x)
        assert snapped % 32 == 0
        assert snapped >= 32


def test_ltx_registered():
    from backend.generation.registry import list_backends

    assert "ltx" in list_backends()
