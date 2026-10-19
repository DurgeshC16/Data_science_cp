"""Smoke test for the Streamlit demo's missing-artifact landing state."""

from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT


def test_demo_renders_without_flow_files_or_model_artifacts() -> None:
    """The demo should explain prerequisites instead of crashing or faking data."""
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "demo.py")).run(timeout=30)

    assert not app.exception
    assert any(element.value == "Cascade-XID — analyst review demo" for element in app.title)
    assert any("No curated sample flow records" in element.value for element in app.info)
