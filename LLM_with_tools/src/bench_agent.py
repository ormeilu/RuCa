from openai import OpenAI
from settings import OpenAISettings
import dotenv
from typing import Any, Dict, List, Optional
import json
import sys
import os

# Импортируем функции из основного файла
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

dotenv.load_dotenv(dotenv.find_dotenv())


def get_client() -> OpenAI:
    """Создаёт OpenAI клиента с настройками."""
    settings = OpenAISettings()
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=str(settings.openai_base_url),
    )


def convert_retail_tools_to_openai(retail_tools_metadata: List[Dict[str, Any]], strict: bool = False) -> List[Dict[str, Any]]:
    """Конвертирует метаданные инструментов в OpenAI формат."""
    openai_tools = []
    
    for retail_tool in retail_tools_metadata:
        parameters = retail_tool.get("parameters", {})
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": retail_tool.get("name", ""),
                "description": retail_tool.get("description", ""),
                "parameters": {
                    "type": parameters.get("type", "object"),
                    "properties": parameters.get("properties", {}),
                    "required": parameters.get("required", [])
                }
            }
        }
        
        if strict:
            openai_tool["function"]["parameters"]["additionalProperties"] = False
            openai_tool["function"]["strict"] = True
        
        openai_tools.append(openai_tool)
    
    return openai_tools


class BenchmarkAgent:
    """
    Агент для batch-обработки запросов в бенчмарке.
    Инициализируется один раз, затем обрабатывает множество запросов.
    """
    
    def __init__(
        self,
        model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        use_retail: bool = True,
        use_weather: bool = True,
        use_translate: bool = True,
        use_calculator: bool = True,
        use_trash: bool = True,
        # use_airbnb:bool = True,
        verbose: bool = False
    ):
        """
        Инициализация агента и загрузка инструментов.
        
        Args:
            model: Название модели для использования
            use_retail: Загрузить retail инструменты
            use_weather: Загрузить weather/currency инструменты
            use_translate: Загрузить translate инструмент
            use_calculator: Загрузить calculator инструмент
            use_trash: Загрузить trash (тестовые) инструменты
            verbose: Выводить отладочную информацию
        """
        self.model = model
        self.verbose = verbose
        self.client = get_client()
        self.openai_tools = []
        
        # Загрузка инструментов
        if use_retail:
            self._load_retail_tools()
        
        if use_weather:
            self._load_weather_tools()
        
        if use_translate:
            self._load_translate_tools()
        
        if use_calculator:
            self._load_calculator_tools()
        
        if use_trash:
            self._load_trash_tools()
        
        # if use_airbnb:
        #     self._load_airbnb_tools()
        if self.verbose:
            print(f"✅ BenchmarkAgent initialized with {len(self.openai_tools)} tools")
            print(f"   Model: {self.model}")
    
    def _load_retail_tools(self):
        """Загрузка retail инструментов."""
        try:
            from retail_tools import EcommerceTools
            retail_tools = EcommerceTools.get_tools_metadata()
            retail_openai_tools = convert_retail_tools_to_openai(retail_tools, strict=False)
            self.openai_tools.extend(retail_openai_tools)
            if self.verbose:
                print(f"   Loaded {len(retail_tools)} retail tools")
        except Exception as e:
            print(f"⚠️  Failed to load retail tools: {e}")
    
    def _load_weather_tools(self):
        """Загрузка weather/currency инструментов."""
        try:
            from weather_and_convert import MiscTools
            weather_tools = MiscTools.get_tools_metadata()
            weather_openai_tools = convert_retail_tools_to_openai(weather_tools, strict=False)
            self.openai_tools.extend(weather_openai_tools)
            if self.verbose:
                print(f"   Loaded {len(weather_tools)} weather/currency tools")
        except Exception as e:
            print(f"⚠️  Failed to load weather tools: {e}")
    
    def _load_translate_tools(self):
        """Загрузка translate инструмента."""
        try:
            from trans import TranslateTools
            translate_tools = TranslateTools.get_tools_metadata()
            translate_openai_tools = convert_retail_tools_to_openai(translate_tools, strict=False)
            self.openai_tools.extend(translate_openai_tools)
            if self.verbose:
                print(f"   Loaded {len(translate_tools)} translate tool")
        except Exception as e:
            print(f"⚠️  Failed to load translate tools: {e}")
    
    def _load_calculator_tools(self):
        """Загрузка calculator инструмента."""
        try:
            from caclute import CalculatorTool
            calculator_tools = CalculatorTool.get_tools_metadata()
            calculator_openai_tools = convert_retail_tools_to_openai(calculator_tools, strict=False)
            self.openai_tools.extend(calculator_openai_tools)
            if self.verbose:
                print(f"   Loaded {len(calculator_tools)} calculator tool")
        except Exception as e:
            print(f"⚠️  Failed to load calculator tools: {e}")
    
    def _load_trash_tools(self):
        """Загрузка trash (тестовых) инструментов."""
        try:
            from trash import NullTools
            trash_tools = NullTools.get_tools_metadata()
            trash_openai_tools = convert_retail_tools_to_openai(trash_tools, strict=False)
            self.openai_tools.extend(trash_openai_tools)
            if self.verbose:
                print(f"   Loaded {len(trash_tools)} trash tools")
        except Exception as e:
            print(f"⚠️  Failed to load trash tools: {e}")
    
    def run_single_query(
        self,
        user_query: str,
        system_prompt: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Обработка одного запроса.
        
        Args:
            user_query: Запрос пользователя
            system_prompt: Системный промпт (опционально)
            query_id: ID запроса для логирования
        
        Returns:
            Словарь с результатами в формате бенчмарка:
            {
                "tool_call": {...} или null,
                "clarification_question": str или null,
                "user_message": str или null,
                "internal": {
                    "reasoning": str или null,
                    "raw_response": {...},
                    "errors": str или null
                }
            }
        """
        if self.verbose and query_id:
            print(f"\n{'='*60}")
            print(f"Processing query: {query_id}")
            print(f"{'='*60}")
        
        # Формируем сообщения
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        try:
            # Вызов модели
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools if self.openai_tools else None,
                tool_choice="auto"
            )
            
            # Парсинг ответа
            return self._parse_response(response, query_id)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error processing query {query_id}: {e}")
            
            return {
                "tool_call": None,
                "clarification_question": None,
                "user_message": None,
                "internal": {
                    "reasoning": None,
                    "raw_response": None,
                    "errors": f"API_ERROR: {str(e)}"
                }
            }
    
    def _parse_response(
        self,
        response: Any,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Парсинг ответа модели в формат бенчмарка.
        
        Args:
            response: Ответ от OpenAI API
            query_id: ID запроса для логирования
        
        Returns:
            Словарь в формате бенчмарка
        """
        message = response.choices[0].message
        
        # Извлекаем данные
        tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else None
        content = message.content if hasattr(message, 'content') else None
        
        result = {
            "tool_call": None,
            "clarification_question": None,
            "user_message": None,
            "internal": {
                "reasoning": None,
                "raw_response": {
                    "content": content,
                    "tool_calls": []
                },
                "errors": None
            }
        }
        
        # Если есть tool calls
        if tool_calls and len(tool_calls) > 0:
            # Берём первый tool call (для бенчмарка нужен только один)
            first_tool_call = tool_calls[0]
            
            try:
                arguments = json.loads(first_tool_call.function.arguments)
            except json.JSONDecodeError as e:
                arguments = {}
                result["internal"]["errors"] = f"JSON_PARSE_ERROR: {str(e)}"
            
            result["tool_call"] = {
                "name": first_tool_call.function.name,
                "arguments": arguments
            }
            
            # Сохраняем все tool calls для internal
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    args = tc.function.arguments
                
                result["internal"]["raw_response"]["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args
                })
            
            if self.verbose:
                print(f"✅ Tool call detected: {first_tool_call.function.name}")
                print(f"   Arguments: {json.dumps(arguments, ensure_ascii=False)}")
        
        # Если нет tool calls, но есть текстовый ответ
        elif content:
            # Проверяем, похоже ли на уточняющий вопрос
            if self._is_clarification_question(content):
                result["clarification_question"] = content
                if self.verbose:
                    print(f"❓ Clarification question detected")
            else:
                result["user_message"] = content
                if self.verbose:
                    print(f"💬 Text response (no tool call)")
        
        # Если нет ни tool calls, ни контента
        else:
            result["internal"]["errors"] = "EMPTY_RESPONSE: No tool calls and no content"
            if self.verbose:
                print(f"⚠️  Empty response")
        
        return result
    
    def _is_clarification_question(self, text: str) -> bool:
        """
        Эвристическая проверка, является ли текст уточняющим вопросом.
        
        Args:
            text: Текст ответа модели
        
        Returns:
            True если похоже на уточняющий вопрос
        """
        # Простые эвристики (можно улучшить)
        question_markers = ['?', 'уточните', 'уточнить', 'какой', 'какая', 'какое', 
                           'какие', 'что именно', 'не понял', 'не ясно', 'поясните']
        
        text_lower = text.lower()
        
        # Если есть вопросительный знак и один из маркеров
        if '?' in text:
            return any(marker in text_lower for marker in question_markers)
        
        return False
    
    def get_tools_info(self) -> List[Dict[str, str]]:
        """
        Получить информацию о загруженных инструментах.
        
        Returns:
            Список словарей с информацией об инструментах
        """
        tools_info = []
        for tool in self.openai_tools:
            func = tool.get("function", {})
            tools_info.append({
                "name": func.get("name", "unknown"),
                "description": func.get("description", "")
            })
        return tools_info