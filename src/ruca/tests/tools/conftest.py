"""
Shared fixtures and configuration for tools tests
"""

import pytest


@pytest.fixture
def sample_location():
    """Sample location for testing"""
    return "Moscow"


@pytest.fixture
def sample_date():
    """Sample date for testing"""
    return "2026-01-15"


@pytest.fixture
def sample_amount():
    """Sample amount for testing"""
    return 100.0


@pytest.fixture
def sample_currency():
    """Sample currency pair"""
    return {"from": "USD", "to": "EUR", "amount": 100.0}
