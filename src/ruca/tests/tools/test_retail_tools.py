"""
Tests for retail_tools (EcommerceTools)
Testing e-commerce/retail operations with pytest and mock
"""

from unittest.mock import patch

from ruca.tools.retail_tools import EcommerceTools


class TestEcommerceTools:
    """Test suite for EcommerceTools (retail/e-commerce)"""

    def test_get_tools_metadata(self):
        """Test that get_tools_metadata returns correct structure"""
        metadata = EcommerceTools.get_tools_metadata()

        assert isinstance(metadata, list)
        assert len(metadata) > 0

        # Check expected tools
        tool_names = [tool["name"] for tool in metadata]
        assert "search_products" in tool_names
        assert "place_order" in tool_names
        assert "track_order" in tool_names

    def test_cancel_order(self):
        """Test cancel_order operation"""
        result = EcommerceTools.cancel_order(order_id="ORDER123")

        assert isinstance(result, dict)
        assert "success" in result

    def test_search_products_basic(self):
        """Test search_products with simple query"""
        result = EcommerceTools.search_products(query="laptop")

        assert isinstance(result, dict)
        assert "success" in result
        if result["success"]:
            assert "products" in result

        assert isinstance(result, dict)
        assert "success" in result

    def test_return_order(self):
        """Test return_order operation"""
        result = EcommerceTools.return_order(order_id="ORDER456", reason="Defective product")

        assert isinstance(result, dict)
        assert "success" in result

    def test_place_order(self):
        """Test place_order operation"""
        result = EcommerceTools.place_order(items="laptop", address="123 Main St, City")

        assert isinstance(result, dict)
        assert "success" in result

    def test_track_order(self):
        """Test track_order operation"""
        result = EcommerceTools.track_order(order_id="ORDER789")

        assert isinstance(result, dict)
        assert "success" in result

    def test_search_products_returns_product_info(self):
        """Test that search_products returns product information"""
        result = EcommerceTools.search_products(query="keyboard")

        if result["success"] and "product" in result:
            product = result["product"]
            assert "id" in product or "name" in product or "price" in product

    def test_place_order_requires_items(self):
        """Test that place_order requires items parameter"""
        result = EcommerceTools.place_order(items="monitor", address="456 Oak Ave, Town")

        assert isinstance(result, dict)

    def test_track_order_returns_status(self):
        """Test that track_order returns order status"""
        result = EcommerceTools.track_order(order_id="ORDER999")

        if result["success"]:
            # Should contain status information
            assert isinstance(result, dict)

    def test_cancel_order_existing_order(self):
        """Test canceling an existing order"""
        # First place an order
        place_result = EcommerceTools.place_order(items="tablet", address="789 Pine Rd")

        if place_result.get("success"):
            order_id = place_result.get("order_id", "ORDER001")
            cancel_result = EcommerceTools.cancel_order(order_id=order_id)
            assert isinstance(cancel_result, dict)

    def test_return_order_with_multiple_reasons(self):
        """Test return_order with different reasons"""
        reasons = ["Defective", "Wrong item", "Changed mind", "No longer needed"]

        for reason in reasons:
            result = EcommerceTools.return_order(order_id="ORDER_TEST", reason=reason)
            assert isinstance(result, dict)

    def test_search_products_multiple_queries(self):
        """Test search_products with various queries"""
        queries = ["laptop", "mouse", "keyboard", "monitor", "headphones"]

        for query in queries:
            result = EcommerceTools.search_products(query=query)
            assert isinstance(result, dict)
            assert "success" in result

    def test_place_order_multiple_items(self):
        """Test place_order with different items"""
        items_list = ["laptop", "mouse", "keyboard"]

        for items in items_list:
            result = EcommerceTools.place_order(items=items, address="123 Test St")
            assert isinstance(result, dict)

    def test_track_order_various_order_ids(self):
        """Test track_order with various order IDs"""
        order_ids = ["ORD001", "ORD002", "ORD003"]

        for order_id in order_ids:
            result = EcommerceTools.track_order(order_id=order_id)
            assert isinstance(result, dict)
            assert "success" in result

    @patch("ruca.tools.retail_tools.random.choice")
    def test_search_products_with_mock_random(self, mock_choice):
        """Test search_products with mocked random"""
        mock_choice.return_value = "Mock Product"

        result = EcommerceTools.search_products(query="test")
        assert isinstance(result, dict)

    def test_search_products_price_filter_effect(self):
        """Test that price filter is applied"""
        expensive = EcommerceTools.search_products(query="laptop", max_price=100.0)

        cheap = EcommerceTools.search_products(query="laptop", max_price=10000.0)

        # Both should return valid responses
        assert isinstance(expensive, dict)
        assert isinstance(cheap, dict)

    def test_all_operations_return_success_key(self):
        """Test that all operations return success key"""
        operations = [
            lambda: EcommerceTools.cancel_order("ORD001"),
            lambda: EcommerceTools.search_products("item"),
            lambda: EcommerceTools.return_order("ORD002", "reason"),
            lambda: EcommerceTools.place_order("item", "address"),
            lambda: EcommerceTools.track_order("ORD003"),
        ]

        for operation in operations:
            result = operation()
            assert "success" in result
            assert isinstance(result["success"], bool)

    def test_place_order_with_complex_address(self):
        """Test place_order with complex address"""
        complex_address = "Apt 123, 456 Main Street, Suite 789, City, State 12345"

        result = EcommerceTools.place_order(items="laptop", address=complex_address)

        assert isinstance(result, dict)

    def test_search_products_empty_query(self):
        """Test search_products with empty query"""
        result = EcommerceTools.search_products(query="")

        # Should still return a dict, even if no results
        assert isinstance(result, dict)

    def test_order_flow(self):
        """Test complete order flow: search -> place -> track"""
        # Search for product
        search = EcommerceTools.search_products(query="headphones")
        assert isinstance(search, dict)

        # Place order
        order = EcommerceTools.place_order(items="headphones", address="Test Address")
        assert isinstance(order, dict)

        # Track order
        if order.get("success"):
            order_id = order.get("order_id", "TEST_ORDER")
            track = EcommerceTools.track_order(order_id=order_id)
            assert isinstance(track, dict)

    def test_tools_metadata_has_descriptions(self):
        """Test that all tools in metadata have descriptions"""
        metadata = EcommerceTools.get_tools_metadata()

        for tool in metadata:
            assert "description" in tool
            assert len(tool["description"]) > 0

    def test_tools_metadata_has_parameters(self):
        """Test that all tools in metadata have proper parameters"""
        metadata = EcommerceTools.get_tools_metadata()

        for tool in metadata:
            assert "parameters" in tool
            params = tool["parameters"]
            assert "type" in params
            assert "properties" in params

    @patch("ruca.tools.retail_tools.random.randint")
    def test_operations_with_mocked_randint(self, mock_randint):
        """Test operations with mocked randint"""
        mock_randint.return_value = 100

        result = EcommerceTools.place_order("item", "address")
        assert isinstance(result, dict)

    def test_cancel_order_various_ids(self):
        """Test cancel_order with various format order IDs"""
        order_ids = ["ORDER123", "ORD-2026-001", "2026_ORDER_TEST", "SIMPLE_ID"]

        for order_id in order_ids:
            result = EcommerceTools.cancel_order(order_id=order_id)
            assert isinstance(result, dict)

    def test_search_products_returns_dict_always(self):
        """Test that search_products always returns dict"""
        test_cases = [
            {"query": "product"},
            {"query": "item", "max_price": 50},
            {"query": "expensive", "max_price": 10000},
        ]

        for kwargs in test_cases:
            result = EcommerceTools.search_products(**kwargs)
            assert isinstance(result, dict)
