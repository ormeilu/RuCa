"""
Набор НЕРАБОЧИХ (null) инструментов для бенчмарка.
Каждый инструмент имеет имя с суффиксом "_T" и всегда возвращает единую "Null" ошибку/ответ.
Используется для тестирования устойчивости LLM-агентов к мусорным / неработающим интерфейсам.
"""

from typing import Dict, Any, List


class NullTools:
    @staticmethod
    def get_tools_metadata() -> List[Dict[str, Any]]:
        """Метаданные всех null-инструментов (идентично именам, но с суффиксом _T)."""
        tool_names = [
            "cancel_order_T",
            "search_products_T",
            "return_order_T",
            "place_order_T",
            "track_order_T",
            "update_address_T",
            "add_to_cart_T",
            "remove_from_cart_T",
            "update_payment_method_T",
            "apply_discount_code_T",
            "get_order_history_T",
            "schedule_delivery_T",
            "update_profile_T",
            "contact_support_T",
            # дополнительные доменные утилиты
            "calculator_T",
            "get_weather_T",
            "translate_T",
            "currency_converter_T",
            "get_time_T",
            "get_date_T",
            # мои дополнительные "шумовые" null-инструменты
            "ping_T",
            "noop_T",
            "faulty_gateway_T",
            # авиа и смежное
            "flight_status_T",
            "baggage_service_T",
            "rebooking_T",
            "seat_map_T",
            "refund_T",
            "loyalty_T",
            # доп. утилиты
            "timezone_convert_T",
            "wishlist_T",
            "price_alert_T",
            "parsing_error_T",
        ]

        metadata = []
        for name in tool_names:
            metadata.append({
                "name": name,
                "description": f"NON-WORKING mock tool — always returns Null response ({name}).",
                # Параметры можно задать пустыми — это намеренно неработающие моки.
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            })
        return metadata

    # ========== Нерабочие исполнители (все ведут себя одинаково) ==========
    @staticmethod
    def _null_response(tool_name: str, **_kwargs) -> Dict[str, Any]:
        """Единый ответ для всех нерабочих инструментов."""
        return {
            "success": False,
            "error": "null_response",
            "message": "Null",
            "tool": tool_name
        }

    # Для удобства — явные обёртки (они просто вызывают _null_response)
    @staticmethod
    def cancel_order_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("cancel_order_T", **kwargs)

    @staticmethod
    def search_products_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("search_products_T", **kwargs)

    @staticmethod
    def return_order_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("return_order_T", **kwargs)

    @staticmethod
    def place_order_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("place_order_T", **kwargs)

    @staticmethod
    def track_order_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("track_order_T", **kwargs)

    @staticmethod
    def update_address_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("update_address_T", **kwargs)

    @staticmethod
    def add_to_cart_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("add_to_cart_T", **kwargs)

    @staticmethod
    def remove_from_cart_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("remove_from_cart_T", **kwargs)

    @staticmethod
    def update_payment_method_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("update_payment_method_T", **kwargs)

    @staticmethod
    def apply_discount_code_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("apply_discount_code_T", **kwargs)

    @staticmethod
    def get_order_history_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("get_order_history_T", **kwargs)

    @staticmethod
    def schedule_delivery_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("schedule_delivery_T", **kwargs)

    @staticmethod
    def update_profile_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("update_profile_T", **kwargs)

    @staticmethod
    def contact_support_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("contact_support_T", **kwargs)

    @staticmethod
    def calculator_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("calculator_T", **kwargs)

    @staticmethod
    def get_weather_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("get_weather_T", **kwargs)

    @staticmethod
    def translate_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("translate_T", **kwargs)

    @staticmethod
    def currency_converter_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("currency_converter_T", **kwargs)

    @staticmethod
    def get_time_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("get_time_T", **kwargs)

    @staticmethod
    def get_date_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("get_date_T", **kwargs)

    # Дополнительные шумовые null-инструменты
    @staticmethod
    def ping_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("ping_T", **kwargs)

    @staticmethod
    def noop_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("noop_T", **kwargs)

    @staticmethod
    def faulty_gateway_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("faulty_gateway_T", **kwargs)

    @staticmethod
    def flight_status_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("flight_status_T", **kwargs)

    @staticmethod
    def baggage_service_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("baggage_service_T", **kwargs)

    @staticmethod
    def rebooking_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("rebooking_T", **kwargs)

    @staticmethod
    def seat_map_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("seat_map_T", **kwargs)

    @staticmethod
    def refund_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("refund_T", **kwargs)

    @staticmethod
    def loyalty_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("loyalty_T", **kwargs)

    @staticmethod
    def timezone_convert_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("timezone_convert_T", **kwargs)

    @staticmethod
    def wishlist_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("wishlist_T", **kwargs)

    @staticmethod
    def price_alert_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("price_alert_T", **kwargs)

    @staticmethod
    def parsing_error_T(**kwargs) -> Dict[str, Any]:
        return NullTools._null_response("parsing_error_T", **kwargs)


def register_null_tools(tool_registry):
    """
    Регистрирует все null-инструменты в переданном registry.
    Использование: register_null_tools(tool_registry)
    """
    metas = NullTools.get_tools_metadata()

    # маппинг имён в исполнители
    executors = {
        "cancel_order_T": NullTools.cancel_order_T,
        "search_products_T": NullTools.search_products_T,
        "return_order_T": NullTools.return_order_T,
        "place_order_T": NullTools.place_order_T,
        "track_order_T": NullTools.track_order_T,
        "update_address_T": NullTools.update_address_T,
        "add_to_cart_T": NullTools.add_to_cart_T,
        "remove_from_cart_T": NullTools.remove_from_cart_T,
        "update_payment_method_T": NullTools.update_payment_method_T,
        "apply_discount_code_T": NullTools.apply_discount_code_T,
        "get_order_history_T": NullTools.get_order_history_T,
        "schedule_delivery_T": NullTools.schedule_delivery_T,
        "update_profile_T": NullTools.update_profile_T,
        "contact_support_T": NullTools.contact_support_T,
        "calculator_T": NullTools.calculator_T,
        "get_weather_T": NullTools.get_weather_T,
        "translate_T": NullTools.translate_T,
        "currency_converter_T": NullTools.currency_converter_T,
        "get_time_T": NullTools.get_time_T,
        "get_date_T": NullTools.get_date_T,
        "ping_T": NullTools.ping_T,
        "noop_T": NullTools.noop_T,
        "faulty_gateway_T": NullTools.faulty_gateway_T,
        "flight_status_T": NullTools.flight_status_T,
        "baggage_service_T": NullTools.baggage_service_T,
        "rebooking_T": NullTools.rebooking_T,
        "seat_map_T": NullTools.seat_map_T,
        "refund_T": NullTools.refund_T,
        "loyalty_T": NullTools.loyalty_T,
        "timezone_convert_T": NullTools.timezone_convert_T,
        "wishlist_T": NullTools.wishlist_T,
        "price_alert_T": NullTools.price_alert_T,
        "parsing_error_T": NullTools.parsing_error_T,
    }

    for meta in metas:
        name = meta["name"]
        # регистрируем, даже если в executors нет — тогда просто пропускаем
        if name in executors:
            tool_registry.register_tool(name, meta, executors[name])

    print(f"✅ Зарегистрировано {len(executors)} null-инструментов (suffix _T)")
