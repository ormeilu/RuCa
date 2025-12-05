# try_agent.py

import asyncio
from openai import OpenAI


# =======================
#   SIMPLE TOOL REGISTRY
# =======================

class ToolRegistry:
    """Хранилище инструментов, совместимое с моками."""
    def __init__(self):
        # каждый инструмент: {name, description, parameters, executor}
        self.tools = []

    def register_tool(self, name, metadata, executor):
        """
        Регистрирует один инструмент.
        name       - имя функции
        metadata   - словарь, содержащий "description" + "parameters"
        executor   - функция Python, которую затем нужно будет выполнить
        """
        self.tools.append({
            "name": name,
            "description": metadata["description"],
            "parameters": metadata["parameters"],
            "executor": executor
        })


# =======================
#         AGENT
# =======================

class ToolAgent:
    """Агент, умеющий работать с инструментами через OpenAI Tools API."""

    def __init__(self, model="gpt-4.1"):
        self.client = OpenAI()
        self.model = model

        # создаём локальный registry
        self.registry = ToolRegistry()

        # Импортируем и вызываем регистраторы инструментов
        # ⚠️ ВАЖНО: эти файлы должны лежать рядом в src/ или быть в sys.path
        from retail_tools import register_ecommerce_tools
        from trans import register_translate_tools
        from weather_and_convert import register_misc_tools

        register_ecommerce_tools(self.registry)
        
        register_translate_tools(self.registry)
        register_misc_tools(self.registry)

    # ---- Service API ----

    def get_tools_info(self):
        """Возвращает список подключенных инструментов (для отображения)."""
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self.registry.tools
        ]

    # ---- Query Runner ----

    async def _run_query_async(self, user_query, system_prompt, query_id):
        """
        Внутренний обработчик запроса к LLM.
        Если модель выберет tool_call, агент выполнит нужную Python-функцию.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"]
                    }
                }
                for t in self.registry.tools
            ],
            tool_choice="auto"
        )

        choice = response.choices[0]

        # Если LLM вызывает инструмент
        if choice.finish_reason == "tool_calls":
            call = choice.message.tool_calls[0]
            tool_name = call.function.name
            args = call.function.arguments  # dict

            # находим executor
            executor = None
            for t in self.registry.tools:
                if t["name"] == tool_name:
                    executor = t["executor"]
                    break

            if executor is None:
                return {
                    "query_id": query_id,
                    "error": "TOOL_NOT_FOUND",
                    "tool": tool_name
                }

            # ВЫПОЛНЯЕМ Python-функцию
            try:
                result = executor(**args)
            except Exception as e:
                result = {"success": False, "error": str(e)}

            return {
                "query_id": query_id,
                "tool": tool_name,
                "output": result
            }

        # Если модель НЕ вызывает инструмент, просто возвращаем текст
        return {
            "query_id": query_id,
            "output": choice.message.content
        }

    def run_single_query(self, user_query, system_prompt, query_id):
        """Синхронная обёртка вокруг async вызова."""
        return asyncio.run(
            self._run_query_async(user_query, system_prompt, query_id)
        )
