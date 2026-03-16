import pytest


def pytest_collection_modifyitems(config, items):
    """Register custom markers to suppress warnings."""
    for item in items:
        if "api" in item.keywords:
            item.add_marker(pytest.mark.api)
