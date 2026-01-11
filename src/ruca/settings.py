from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    openai_api_key: SecretStr = Field()
    openai_base_url: HttpUrl = Field()

    model_config = SettingsConfigDict(env_file=".env")


if __name__ == "__main__":
    settings = OpenAISettings()
    print(settings)
