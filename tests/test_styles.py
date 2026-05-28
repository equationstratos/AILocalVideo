"""Tests du chargement et de la fusion des styles."""

import pytest

from backend.styles import load_styles


def test_load_styles_from_config():
    styles = load_styles()  # utilise config/styles.yaml par défaut
    assert "animation" in styles
    assert "realistic" in styles
    # Chaque style a un backend et des défauts complets.
    for style in styles.values():
        assert style.backend
        for key in ("num_frames", "fps", "steps", "guidance", "width", "height"):
            assert key in style.defaults


def test_build_prompt_merges_prefix_suffix():
    styles = load_styles()
    style = styles["anime"]
    prompt = style.build_prompt("a cat playing")
    assert "a cat playing" in prompt
    assert prompt.endswith(style.prompt_suffix)


def test_parse_rejects_missing_defaults(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "styles:\n  broken:\n    backend: animatediff\n    defaults:\n      fps: 8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_styles(bad)


def test_parse_rejects_missing_backend(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("styles:\n  broken:\n    label: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_styles(bad)
