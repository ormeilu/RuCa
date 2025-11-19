from typing import Dict, Any

class TranslateTools:
    """Mock переводчик для бенчмарка LLM-агентов"""

    # ======== EN -> RU словарь ========
    en_ru = {
        "hello": "привет", "hi": "привет",
        "world": "мир",
        "good": "хороший", "bad": "плохой",
        "yes": "да", "no": "нет",
        "cat": "кот", "dog": "собака",
        "food": "еда", "water": "вода",
        "order": "заказ", "product": "товар",
        "house": "дом", "street": "улица",
        "car": "машина", "bus": "автобус",
        "train": "поезд", "airport": "аэропорт",
        "money": "деньги", "price": "цена",
        "buy": "купить", "sell": "продать",
        "name": "имя", "email": "почта",
        "phone": "телефон",
        "day": "день", "night": "ночь",
        "happy": "счастливый", "sad": "грустный",
        "fast": "быстрый", "slow": "медленный",
        "big": "большой", "small": "маленький",
        "new": "новый", "old": "старый",
        "market": "рынок", "shop": "магазин",
        "apple": "яблоко", "bread": "хлеб",
        "coffee": "кофе", "tea": "чай",
        "city": "город", "country": "страна",
        "family": "семья", "friend": "друг",
        "work": "работа", "home": "дом",
        "school": "школа", "university": "университет"
    }

    # ======== RU -> EN из словаря выше ========
    ru_en = {v: k for k, v in en_ru.items()}

    # ======== RU -> FR словарь ========
    ru_fr = {
        "привет": "bonjour", "мир": "monde",
        "хороший": "bon", "плохой": "mauvais",
        "да": "oui", "нет": "non",
        "кот": "chat", "собака": "chien",
        "еда": "nourriture", "вода": "eau",
        "дом": "maison", "улица": "rue",
        "машина": "voiture", "автобус": "bus",
        "поезд": "train", "аэропорт": "aéroport",
        "деньги": "argent", "цена": "prix",
        "купить": "acheter", "продать": "vendre",
        "имя": "nom", "почта": "email",
        "телефон": "téléphone",
        "день": "jour", "ночь": "nuit",
        "большой": "grand", "маленький": "petit",
        "новый": "nouveau", "старый": "ancien",
        "рынок": "marché", "магазин": "magasin",
        "яблоко": "pomme", "хлеб": "pain",
        "кофе": "café", "чай": "thé",
        "город": "ville", "страна": "pays",
        "семья": "famille", "друг": "ami",
        "работа": "travail", "школа": "école",
        "университет": "université"
    }

    # FR -> RU
    fr_ru = {v: k for k, v in ru_fr.items()}

    @staticmethod
    def get_tools_metadata():
        return [
            {
                "name": "translate",
                "description": "Переводит текст между языками (mock). Поддержка EN↔RU, RU↔FR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source": {"type": "string"},
                        "target": {"type": "string"}
                    },
                    "required": ["text", "source", "target"]
                }
            }
        ]

    @staticmethod
    def translate(text: str, source: str, target: str) -> Dict[str, Any]:
        words = text.lower().split()

        # Определяем словарь
        if source == "en" and target == "ru":
            dictionary = TranslateTools.en_ru
        elif source == "ru" and target == "en":
            dictionary = TranslateTools.ru_en
        elif source == "ru" and target == "fr":
            dictionary = TranslateTools.ru_fr
        elif source == "fr" and target == "ru":
            dictionary = TranslateTools.fr_ru
        else:
            return {
                "success": False,
                "error": "unsupported_language_pair",
                "message": "Supported: en↔ru, ru↔fr"
            }

        translated = [dictionary.get(w, f"UNKNOWN({w})") for w in words]

        return {
            "success": True,
            "input": text,
            "source": source,
            "target": target,
            "translated": " ".join(translated)
        }


def register_translate_tools(tool_registry):
    meta = TranslateTools.get_tools_metadata()
    tool_registry.register_tool("translate", meta[0], TranslateTools.translate)
    print(f"✅ Зарегистрирован mock-переводчик — {len(TranslateTools.en_ru)} EN/RU слов, {len(TranslateTools.ru_fr)} RU/FR слов")
