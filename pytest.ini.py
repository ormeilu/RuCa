"""
pytest Configuration and Best Practices
This file contains pytest configuration used across the tests
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path so we can import ruca modules
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ============== MARKERS ==============
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ============== FIXTURES ==============
@pytest.fixture(scope="session")
def mock_datetime_fixture():
    """Provides mocked datetime for consistent testing"""
    with patch("datetime.datetime") as mock:
        mock.now.return_value.__str__.return_value = "2026-01-15 12:00:00"
        yield mock


@pytest.fixture
def sample_api_response():
    """Sample API response for mocking"""
    return {
        "status": "success",
        "data": {"id": "123", "value": "test"},
        "code": 200
    }


@pytest.fixture
def mock_external_api(mocker):
    """Mock external API calls"""
    return mocker.patch("requests.get")


# ============== PYTEST HOOKS ==============
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to add custom reporting"""
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call":
        # Custom reporting logic here
        pass


# ============== PARAMETRIZED TEST HELPERS ==============
def pytest_generate_tests(metafunc):
    """Generate tests with parameters if needed"""
    if "test_param" in metafunc.fixturenames:
        metafunc.parametrize("test_param", ["value1", "value2", "value3"])


# ============== COLLECTION HOOKS ==============
def pytest_collection_modifyitems(config, items):
    """Modify collected items"""
    for item in items:
        # Mark all tests as unit by default
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)


# ============== ASSERTION HELPERS ==============
class TestAssertions:
    """Helper class for common test assertions"""
    
    @staticmethod
    def assert_valid_response(response):
        """Assert response has required fields"""
        assert isinstance(response, dict)
        assert "success" in response
        assert isinstance(response["success"], bool)
    
    @staticmethod
    def assert_has_error(response):
        """Assert response contains error"""
        assert response.get("success") is False
        assert "error" in response
    
    @staticmethod
    def assert_is_dict_with_keys(obj, required_keys):
        """Assert object is dict with required keys"""
        assert isinstance(obj, dict)
        for key in required_keys:
            assert key in obj


# ============== MOCK HELPERS ==============
class MockHelpers:
    """Helper functions for mocking"""
    
    @staticmethod
    def create_mock_tool(name, return_value=None):
        """Create a mock tool with default return value"""
        mock = MagicMock(name=name)
        mock.return_value = return_value or {"success": True}
        return mock
    
    @staticmethod
    def create_side_effect_mock(*return_values):
        """Create mock with side effects (multiple return values)"""
        mock = MagicMock()
        mock.side_effect = return_values
        return mock
    
    @staticmethod
    @patch('random.random')
    def mock_random_value(mock_random, value=0.5):
        """Mock random.random() to return specific value"""
        mock_random.return_value = value
        return mock_random
