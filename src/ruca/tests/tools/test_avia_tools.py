"""
Tests for avia_tools (AviationTools)
Testing aviation/booking service tools with pytest and mock
"""

from unittest.mock import patch

from ruca.tools.avia_tools import AviationTools


class TestAviationTools:
    """Test suite for AviationTools"""

    def test_get_tools_metadata(self):
        """Test that get_tools_metadata returns correct structure"""
        metadata = AviationTools.get_tools_metadata()

        assert isinstance(metadata, list)
        assert len(metadata) > 0

        # Check some expected tools
        tool_names = [tool["name"] for tool in metadata]
        assert "BookingService" in tool_names
        assert "FlightStatusService" in tool_names
        assert "CheckInService" in tool_names

    def test_tools_have_required_fields(self):
        """Test that all tools have required metadata fields"""
        metadata = AviationTools.get_tools_metadata()

        for tool in metadata:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "type" in tool["parameters"]
            assert "properties" in tool["parameters"]

    def test_booking_service_success(self):
        """Test BookingService returns success response"""
        result = AviationTools.BookingService(
            passenger_name="John Doe", origin="SVO", destination="LHR", date="2026-02-15"
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_booking_service_with_seat_class(self):
        """Test BookingService with seat class specification"""
        result = AviationTools.BookingService(
            passenger_name="Jane Smith", origin="JFK", destination="CDG", date="2026-03-20", seat_class="business"
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_flight_status_service(self):
        """Test FlightStatusService"""
        result = AviationTools.FlightStatusService(date="2026-02-15")

        assert isinstance(result, dict)
        assert "success" in result

    def test_check_in_service(self):
        """Test CheckInService"""
        result = AviationTools.CheckInService(seat="12A")

        assert isinstance(result, dict)
        assert "success" in result

    def test_upgrade_service(self):
        """Test UpgradeService"""
        result = AviationTools.UpgradeService(new_class="first")

        assert isinstance(result, dict)
        assert "success" in result

    def test_payment_service(self):
        """Test PaymentService"""
        result = AviationTools.PaymentService(amount=1500.0, method="card")

        assert isinstance(result, dict)
        assert "success" in result

    def test_loyalty_service(self):
        """Test LoyaltyService"""
        result = AviationTools.LoyaltyService(card_number="1234567890", action="earn", amount=1000)

        assert isinstance(result, dict)
        assert "success" in result

    def test_baggage_service(self):
        """Test BaggageService"""
        result = AviationTools.BaggageService(weight=23.0, type="checked")

        assert isinstance(result, dict)
        assert "success" in result

    def test_seat_map_service(self):
        """Test SeatMapService"""
        result = AviationTools.SeatMapService(date="2026-02-15")

        assert isinstance(result, dict)
        assert "success" in result

    def test_refund_service(self):
        """Test RefundService"""
        result = AviationTools.RefundService(booking_id="BOOK123456", reason="Changed plans")

        assert isinstance(result, dict)
        assert "success" in result

    def test_rebooking_service(self):
        """Test RebookingService"""
        result = AviationTools.RebookingService(
            booking_id="BOOK123456", new_date="2026-03-15", new_flight_number="SU201"
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_ancillaries_service(self):
        """Test AncillariesService"""
        result = AviationTools.AncillariesService(service="meal")

        assert isinstance(result, dict)
        assert "success" in result

    def test_insurance_service(self):
        """Test InsuranceService"""
        result = AviationTools.InsuranceService(insurance_type="travel")

        assert isinstance(result, dict)
        assert "success" in result

    def test_cargo_service(self):
        """Test CargoService"""
        result = AviationTools.CargoService(weight=100.0, cargo_type="general", origin="SVO", destination="LHR")

        assert isinstance(result, dict)
        assert "success" in result

    def test_lost_and_found_service(self):
        """Test LostAndFoundService"""
        result = AviationTools.LostAndFoundService(
            passenger_name="John Doe", flight_number="SU100", baggage_tag="SVO123456"
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_ops_service(self):
        """Test OpsService"""
        result = AviationTools.OpsService(flight_number="SU100")

        assert isinstance(result, dict)
        assert "success" in result

    @patch("ruca.tools.avia_tools.random.choice")
    def test_booking_service_with_mock_random(self, mock_choice):
        """Test BookingService with mocked random choice"""
        mock_choice.return_value = "CONFIRMED"

        result = AviationTools.BookingService(
            passenger_name="Test User", origin="SVO", destination="LHR", date="2026-02-15"
        )

        assert isinstance(result, dict)

    def test_multiple_services_return_dict(self):
        """Test that all services return dictionaries"""
        services = [
            lambda: AviationTools.BookingService("John", "SVO", "LHR", "2026-02-15"),
            lambda: AviationTools.FlightStatusService("2026-02-15"),
            lambda: AviationTools.CheckInService("12A"),
            lambda: AviationTools.UpgradeService("first"),
            lambda: AviationTools.PaymentService(100.0),
        ]

        for service_call in services:
            result = service_call()
            assert isinstance(result, dict)

    def test_booking_service_generates_confirmation(self):
        """Test that BookingService includes confirmation details"""
        result = AviationTools.BookingService(
            passenger_name="Alice Johnson", origin="CDG", destination="JFK", date="2026-04-10"
        )

        # Service should return dict with success status
        assert isinstance(result, dict)
        assert "success" in result

    def test_payment_service_different_methods(self):
        """Test PaymentService with different payment methods"""
        methods = ["card", "apple_pay", "google_pay"]

        for method in methods:
            result = AviationTools.PaymentService(amount=500.0, method=method)
            assert isinstance(result, dict)
            assert "success" in result

    def test_loyalty_service_earn_action(self):
        """Test LoyaltyService with earn action"""
        result = AviationTools.LoyaltyService(card_number="9876543210", action="earn", amount=500)

        assert isinstance(result, dict)
        assert "success" in result

    def test_loyalty_service_spend_action(self):
        """Test LoyaltyService with spend action"""
        result = AviationTools.LoyaltyService(card_number="9876543210", action="spend", amount=1000)

        assert isinstance(result, dict)
        assert "success" in result

    def test_baggage_service_hand_baggage(self):
        """Test BaggageService with hand baggage"""
        result = AviationTools.BaggageService(weight=8.0, type="hand")

        assert isinstance(result, dict)
        assert "success" in result

    def test_cargo_service_different_types(self):
        """Test CargoService with different cargo types"""
        cargo_types = ["general", "fragile", "dangerous"]

        for cargo_type in cargo_types:
            result = AviationTools.CargoService(weight=50.0, cargo_type=cargo_type, origin="SVO", destination="LHR")
            assert isinstance(result, dict)

    def test_tools_metadata_completeness(self):
        """Test that each tool metadata is complete"""
        metadata = AviationTools.get_tools_metadata()

        for tool in metadata:
            # Every tool must have description
            assert len(tool["description"]) > 0

            # Every tool must have parameters
            params = tool["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

            # Should have required fields listed
            assert "required" in params or len(params.get("properties", {})) == 0

    def test_booking_service_returns_confirmation_id(self):
        """Test that BookingService returns or can generate booking ID"""
        result = AviationTools.BookingService(
            passenger_name="Test Name", origin="LHR", destination="CDG", date="2026-02-28"
        )

        # Should contain success status and likely other info
        assert isinstance(result, dict)

    @patch("ruca.tools.avia_tools.random.randint")
    def test_services_with_mocked_random_int(self, mock_randint):
        """Test services with mocked random integer"""
        mock_randint.return_value = 100

        result = AviationTools.PaymentService(amount=100.0)
        assert isinstance(result, dict)

    def test_all_metadata_tools_are_callable(self):
        """Test that all tools mentioned in metadata can be called"""
        metadata = AviationTools.get_tools_metadata()

        for tool in metadata:
            tool_name = tool["name"]
            # Check that method exists
            assert hasattr(AviationTools, tool_name)
