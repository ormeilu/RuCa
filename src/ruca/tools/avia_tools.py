"""
Aviation инструменты для бенчмарка LLM агентов
Все инструменты являются моками с предопределенными ответами
"""

import random


class AviationTools:
    """Набор mock инструментов для aviation-бенчмарка"""

    @staticmethod
    def get_tools_metadata():
        """Метаданные всех авиа-инструментов"""
        return [
            {
                "name": "BookingService",
                "description": "Оформляет бронирование авиабилета",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passenger_name": {"type": "string", "description": "Имя пассажира"},
                        "origin": {"type": "string", "description": "Аэропорт вылета "},
                        "destination": {"type": "string", "description": "Аэропорт прилета "},
                        "date": {"type": "string", "description": "Дата рейса YYYY-MM-DD"},
                        "seat_class": {"type": "string", "description": "Класс обслуживания (economy/business/first)"},
                    },
                    "required": ["passenger_name", "origin", "destination", "date"],
                },
            },
            {
                "name": "FlightStatusService",
                "description": "Проверяет статус рейса",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string", "description": "Дата вылета YYYY-MM-DD"}},
                    "required": ["date"],
                },
            },
            {
                "name": "CheckInService",
                "description": "Выполняет онлайн-регистрацию на рейс",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seat": {"type": "string", "description": "Желаемое место, например 14A"},
                    },
                    "required": ["seat"],
                },
            },
            {
                "name": "UpgradeService",
                "description": "Оформляет повышение класса обслуживания не запращивая дополнительную информацию (Booking ID не нужен)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_class": {"type": "string", "description": "Новый класс обслуживания"},
                    },
                    "required": ["new_class"],
                },
            },
            {
                "name": "PaymentService",
                "description": "Проводит оплату авиауслуг (Booking ID не нужен)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "method": {"type": "string", "description": "Тип оплаты (card/apple_pay/google_pay)"},
                    },
                    "required": ["amount"],
                },
            },
            {
                "name": "LoyaltyService",
                "description": "Начисляет или списывает бонусные мили",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "card_number": {"type": "string"},
                        "action": {"type": "string", "description": "earn/spend"},
                        "amount": {"type": "number"},
                    },
                    "required": ["card_number", "action"],
                },
            },
            {
                "name": "BaggageService",
                "description": "Добавляет багаж или меняет параметры багажа (Booking ID не нужен)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weight": {"type": "number", "description": "Вес багажа"},
                        "type": {"type": "string", "description": "checked/hand"},
                    },
                    "required": ["weight", "type"],
                },
            },
            {
                "name": "SeatMapService",
                "description": "Получает схему салона (Booking ID не нужен, Flight Number тоже не нужен)",
                "parameters": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]},
            },
            {
                "name": "RefundService",
                "description": "Оформляет возврат средств",
                "parameters": {
                    "type": "object",
                    "properties": {"booking_id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["booking_id"],
                },
            },
            {
                "name": "RebookingService",
                "description": "Перебронирует на другой рейс",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "booking_id": {"type": "string"},
                        "new_date": {"type": "string"},
                        "new_flight_number": {"type": "string"},
                    },
                    "required": ["booking_id"],
                },
            },
            {
                "name": "AncillariesService",
                "description": "Добавляет допуслуги (питание, Fast Track, Wi-Fi)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                    },
                    "required": ["service"],
                },
            },
            {
                "name": "InsuranceService",
                "description": "Подключает страхование к бронированию (Booking ID не нужен)",
                "parameters": {
                    "type": "object",
                    "properties": {"insurance_type": {"type": "string"}},
                    "required": ["insurance_type"],
                },
            },
            {
                "name": "CargoService",
                "description": "Бронирование грузоперевозок ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weight": {"type": "number"},
                        "cargo_type": {"type": "string"},
                        "origin": {
                            "type": "string",
                        },
                        "destination": {
                            "type": "string",
                        },
                    },
                    "required": ["weight", "origin", "destination"],
                },
            },
            {
                "name": "LostAndFoundService",
                "description": "Оформляет запрос на розыск багажа",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passenger_name": {"type": "string"},
                        "flight_number": {"type": "string"},
                        "baggage_tag": {"type": "string"},
                    },
                    "required": ["passenger_name", "flight_number"],
                },
            },
            {
                "name": "OpsService",
                "description": "Получает информацию от операционного центра",
                "parameters": {
                    "type": "object",
                    "properties": {"flight_number": {"type": "string"}},
                    "required": ["flight_number"],
                },
            },
            {
                "name": "HotelService",
                "description": "Бронирование отеля при длительной пересадке",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "airport": {"type": "string"},
                        "nights": {"type": "integer"},
                        "passenger_name": {"type": "string"},
                    },
                    "required": ["airport", "nights"],
                },
            },
            {
                "name": "CompensationService",
                "description": "Запрос компенсации за задержку/отмену рейса",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_number": {"type": "string"},
                        "reason": {"type": "string"},
                        "passenger_name": {"type": "string"},
                    },
                    "required": ["flight_number"],
                },
            },
            {
                "name": "IdentityVerificationService",
                "description": "Проверяет данные пассажира",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passenger_name": {"type": "string"},
                        "document_number": {"type": "string"},
                        "birthdate": {"type": "string"},
                    },
                    "required": ["passenger_name", "document_number"],
                },
            },
        ]

    # Исполняющие методы

    @staticmethod
    def BookingService(passenger_name: str, origin: str, destination: str, date: str, seat_class: str = "economy"):
        return {
            "success": True,
            "booking_id": f"BK{random.randint(10000, 99999)}",
            "passenger": passenger_name,
            "route": f"{origin}-{destination}",
            "date": date,
            "class": seat_class,
        }

    @staticmethod
    def FlightStatusService(date: str = None):
        statuses = ["on_time", "delayed", "boarding", "cancelled"]
        return {"success": True, "status": random.choice(statuses)}

    @staticmethod
    def CheckInService(seat: str = None):
        return {"success": True, "seat": seat or "auto_assigned"}

    @staticmethod
    def UpgradeService(new_class: str):
        return {"success": True, "upgraded_to": new_class}

    @staticmethod
    def PaymentService(amount: float, method: str = "card"):
        return {"success": True, "paid": amount, "method": method}

    @staticmethod
    def LoyaltyService(card_number: str, action: str, amount: float = 0):
        return {"success": True, "card": card_number, "action": action, "miles": amount}

    @staticmethod
    def BaggageService(weight: float = 20.0, type: str = "checked"):
        return {"success": True, "weight": weight, "type": type}

    @staticmethod
    def SeatMapService(date: str = None):
        return {"success": True, "seatmap": "mock_seatmap_data"}

    @staticmethod
    def RefundService(booking_id: str, reason: str = None):
        return {"success": True, "booking_id": booking_id, "refund_id": f"RF{random.randint(1000, 9999)}"}

    @staticmethod
    def RebookingService(booking_id: str, new_date: str = None, new_flight_number: str = None):
        return {"success": True, "booking_id": booking_id, "new_date": new_date, "new_flight": new_flight_number}

    @staticmethod
    def AncillariesService(service: str):
        return {"success": True, "service_added": service}

    @staticmethod
    def InsuranceService(insurance_type: str):
        return {"success": True, "insurance": insurance_type}

    @staticmethod
    def CargoService(weight: float, cargo_type: str, origin: str, destination: str):
        return {
            "success": True,
            "cargo_id": f"CG{random.randint(1000, 9999)}",
            "route": f"{origin}-{destination}",
            "weight": weight,
        }

    @staticmethod
    def LostAndFoundService(passenger_name: str, flight_number: str, baggage_tag: str = None):
        return {"success": True, "passenger": passenger_name, "flight": flight_number, "status": "investigating"}

    @staticmethod
    def OpsService(flight_number: str):
        return {"success": True, "flight": flight_number, "ops_note": "No disruptions reported"}

    @staticmethod
    def HotelService(airport: str, nights: int, passenger_name: str = None):
        return {
            "success": True,
            "hotel_booking_id": f"HT{random.randint(1000, 9999)}",
            "airport": airport,
            "nights": nights,
        }

    @staticmethod
    def CompensationService(flight_number: str, reason: str = None, passenger_name: str = None):
        return {"success": True, "flight": flight_number, "compensation": random.choice([100, 200, 400])}

    @staticmethod
    def IdentityVerificationService(passenger_name: str, document_number: str, birthdate: str = None):
        return {"success": True, "passenger": passenger_name, "verified": True}


def register_avia_tools(tool_registry):
    """Регистрация всех авиа-инструментов"""
    tools_metadata = AviationTools.get_tools_metadata()

    executors = {
        name: getattr(AviationTools, name)
        for name in [
            "BookingService",
            "FlightStatusService",
            "CheckInService",
            "UpgradeService",
            "PaymentService",
            "LoyaltyService",
            "BaggageService",
            "SeatMapService",
            "RefundService",
            "RebookingService",
            "AncillariesService",
            "InsuranceService",
            "CargoService",
            "LostAndFoundService",
            "OpsService",
            "HotelService",
            "CompensationService",
            "IdentityVerificationService",
        ]
    }

    for tool_meta in tools_metadata:
        tool_name = tool_meta["name"]
        tool_registry.register_tool(tool_name, tool_meta, executors[tool_name])

    print(f"✈️ Зарегистрировано {len(tools_metadata)} авиа-инструментов")
