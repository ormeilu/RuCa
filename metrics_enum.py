from enum import StrEnum

class Metrics(StrEnum):
    """Перечисление всех метрик бенчмарка."""
    DECISION = "Decision"
    TOOL_SELECTION = "Tool selection"
    PARAMS = "Params"
    RESULT = "Result"
    ERROR_HANDLING = "Error Handling"
    EXECUTION = "Execution"
    NOISE = "Noise"
    ADAPTABILITY = "Adaptability"
    AMBIGUITY = "Ambiguity"
    
    @property
    def column_name(self) -> str:
        """Возвращает имя для колонки DataFrame (lowercase, snake_case)."""
        return self.value.lower().replace(" ", "_")