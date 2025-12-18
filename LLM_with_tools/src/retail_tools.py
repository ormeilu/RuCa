"""
E-commerce инструменты для бенчмарка LLM агентов
Все инструменты являются моками с предопределенными ответами
"""

from typing import Dict, Any
import random


class EcommerceTools:
    """Набор mock инструментов для e-commerce бенчмарка"""
    
    @staticmethod
    def get_tools_metadata():
        """Метаданные всех e-commerce инструментов"""
        return [
            {
                "name": "cancel_order",
                "description": "Отменяет заказ по его ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "ID заказа для отмены"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "search_products",
                "description": "Ищет товар по запросу и возвращает ОДИН результат результат в виде {\"id\": string, \"name\": string, \"price\": number}. Пример: product = {\"id\": \"P001\", \"name\": \"<query> Pro\", \"price\": 299}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос"
                        },
                        
                        "max_price": {
                            "type": "number",
                            "description": "Максимальная цена (опционально)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "return_order",
                "description": "Оформляет возврат заказа",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "ID заказа для возврата"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Причина возврата"
                        }
                    },
                    "required": ["order_id", "reason"]
                }
            },
            {
                "name": "place_order",
                "description": "Оформляет новый заказ. Используя товар из запроса НЕ производя дополнительного поиска. ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "string",
                            "description": "Название товара",
                           # "items": {"type": "string"}
                        },
                        "address": {
                            "type": "string",
                            "description": "Адрес доставки"
                        }
                    },
                    "required": ["items", "address"]
                }
            },
            {
                "name": "track_order",
                "description": "Отслеживает статус заказа",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "ID заказа для отслеживания"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "update_address",
                "description": "Обновляет адрес доставки",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address_type": {
                            "type": "string",
                            "description": "Тип адреса (home, work)"
                        },
                        "new_address": {
                            "type": "string",
                            "description": "Новый адрес"
                        }
                    },
                    "required": ["address_type", "new_address"]
                }
            },
            {
                "name": "add_to_cart",
                "description": "Добавляет товар в корзину",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "ID товара"
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Количество"
                        }
                    },
                    "required": ["product_id", "quantity"]
                }
            },
            {
                "name": "remove_from_cart",
                "description": "Удаляет товар из корзины",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "ID товара для удаления"
                        }
                    },
                    "required": ["product_id"]
                }
            },
            {
                "name": "update_payment_method",
                "description": "Обновляет способ оплаты",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payment_type": {
                            "type": "string",
                            "description": "Тип оплаты (card, paypal, cash)"
                        },
                        "details": {
                            "type": "string",
                            "description": "Детали платежа"
                        }
                    },
                    "required": ["payment_type"]
                }
            },
            {
                "name": "apply_discount_code",
                "description": "Применяет промокод к заказу",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Промокод"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "get_order_history",
                "description": "Получает историю заказов",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Количество заказов (по умолчанию 10)"
                        }
                    }
                }
            },
            {
                "name": "schedule_delivery",
                "description": "Планирует время доставки",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "ID заказа"
                        },
                        "date": {
                            "type": "string",
                            "description": "Дата доставки (YYYY-MM-DD)"
                        },
                        "time_slot": {
                            "type": "string",
                            "description": "Временной слот (morning, afternoon, evening)"
                        }
                    },
                    "required": ["order_id", "date", "time_slot"]
                }
            },
            {
                "name": "update_profile",
                "description": "Обновляет профиль пользователя",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "description": "Поле для обновления (name, email, phone)"
                        },
                        "value": {
                            "type": "string",
                            "description": "Новое значение"
                        }
                    },
                    "required": ["field", "value"]
                }
            },
            {
                "name": "contact_support",
                "description": "Отправляет сообщение в поддержку",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Тема обращения"
                        },
                        "message": {
                            "type": "string",
                            "description": "Текст сообщения"
                        }
                    },
                    "required": ["subject", "message"]
                }
            }
        ]
    
    # ============= MOCK ИСПОЛНИТЕЛИ =============
    
    @staticmethod
    def cancel_order(order_id: str) -> Dict[str, Any]:
        """Отменяет заказ"""
        return {
            "success": True,
            "order_id": order_id,
            "status": "cancelled"
        }
    
    @staticmethod
    def search_products(query: str,  max_price: float = None) -> Dict[str, Any]:
        """Ищет товары"""
        product ={"id": "P001", "name": f"{query} Pro", "price": 299} 
        
        
        
        return {
            "success": True,
            "query": query,
            "count": len(product),
            "products": product
        }
    
    @staticmethod
    def return_order(order_id: str, reason: str) -> Dict[str, Any]:
        """Оформляет возврат"""
        return {
            "success": True,
            "order_id": order_id,
            "return_id": f"R{order_id}",
            "status": "return_initiated",
            "reason": reason
        }
    
    @staticmethod
    def place_order(items: str, address: str) -> Dict[str, Any]:
        """Оформляет заказ"""
       # order_id = f"ORD{random.randint(10000, 99999)}"
        #total = len(items) * 150  # Простая калькуляция
        
        return {
            "success": True,
           # "order_id": order_id,
           # "items_count": len(items),
           # "total": total,
            "address": address,
            "status": "confirmed"
        }
    
    @staticmethod
    def track_order(order_id: str) -> Dict[str, Any]:
        """Отслеживает заказ"""
        statuses = ["processing", "shipped", "in_transit", "out_for_delivery"]
        
        return {
            "success": True,
            "order_id": order_id,
            "status": random.choice(statuses),
            "estimated_delivery": "2025-10-22"
        }
    
    @staticmethod
    def update_address(address_type: str, new_address: str) -> Dict[str, Any]:
        """Обновляет адрес"""
        return {
            "success": True,
            "address_type": address_type,
            "updated": True
        }
    
    @staticmethod
    def add_to_cart(product_id: str, quantity: int) -> Dict[str, Any]:
        """Добавляет в корзину"""
        return {
            "success": True,
            "product_id": product_id,
            "quantity": quantity,
            "cart_total": 3
        }
    
    @staticmethod
    def remove_from_cart(product_id: str) -> Dict[str, Any]:
        """Удаляет из корзины"""
        return {
            "success": True,
            "product_id": product_id,
            "removed": True
        }
    
    @staticmethod
    def update_payment_method(payment_type: str, details: str = None) -> Dict[str, Any]:
        """Обновляет метод оплаты"""
        return {
            "success": True,
            "payment_type": payment_type,
            "updated": True
        }
    
    @staticmethod
    def apply_discount_code(code: str) -> Dict[str, Any]:
        """Применяет промокод"""
        valid_codes = ["SAVE10", "WELCOME20", "FREESHIP"]
        
        if code.upper() in valid_codes:
            return {
                "success": True,
                "code": code,
                "discount": 10,
                "applied": True
            }
        else:
            return {
                "success": False,
                "code": code,
                "error": "invalid_code"
            }
    
    @staticmethod
    def get_order_history(limit: int = 10) -> Dict[str, Any]:
        """Получает историю заказов"""
        orders = [
            {"order_id": f"ORD{i}", "date": "2025-10-15", "total": 250, "status": "delivered"}
            for i in range(1, min(limit + 1, 11))
        ]
        
        return {
            "success": True,
            "count": len(orders),
            "orders": orders
        }
    
    @staticmethod
    def schedule_delivery(order_id: str, date: str, time_slot: str) -> Dict[str, Any]:
        """Планирует доставку"""
        return {
            "success": True,
            "order_id": order_id,
            "delivery_date": date,
            "time_slot": time_slot,
            "scheduled": True
        }
    
    @staticmethod
    def update_profile(field: str, value: str) -> Dict[str, Any]:
        """Обновляет профиль"""
        return {
            "success": True,
            "field": field,
            "updated": True
        }
    
    @staticmethod
    def contact_support(subject: str, message: str) -> Dict[str, Any]:
        """Отправляет сообщение в поддержку"""
        ticket_id = f"TKT{random.randint(1000, 9999)}"
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "subject": subject,
            "status": "submitted"
        }


def register_ecommerce_tools(tool_registry):
    """
    Регистрация всех e-commerce инструментов в реестре бенчмарка
    
    Args:
        tool_registry: Экземпляр ToolRegistry из основного бенчмарка
    """
    tools_metadata = EcommerceTools.get_tools_metadata()
    
    # Маппинг имен на исполнители
    executors = {
        "cancel_order": EcommerceTools.cancel_order,
        "search_products": EcommerceTools.search_products,
        "return_order": EcommerceTools.return_order,
        "place_order": EcommerceTools.place_order,
        "track_order": EcommerceTools.track_order,
        "update_address": EcommerceTools.update_address,
        "add_to_cart": EcommerceTools.add_to_cart,
        "remove_from_cart": EcommerceTools.remove_from_cart,
        "update_payment_method": EcommerceTools.update_payment_method,
        "apply_discount_code": EcommerceTools.apply_discount_code,
        "get_order_history": EcommerceTools.get_order_history,
        "schedule_delivery": EcommerceTools.schedule_delivery,
        "update_profile": EcommerceTools.update_profile,
        "contact_support": EcommerceTools.contact_support
    }
    
    # Регистрация каждого инструмента
    for tool_meta in tools_metadata:
        tool_name = tool_meta["name"]
        tool_registry.register_tool(
            tool_name,
            tool_meta,
            executors[tool_name]
        )
    
    print(f"✅ Зарегистрировано {len(tools_metadata)} e-commerce инструментов")


# ============= ПРИМЕР ИСПОЛЬЗОВАНИЯ =============

