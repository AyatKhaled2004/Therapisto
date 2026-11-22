import os
import certifi
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get("SSL_CERT_FILE")

if not ssl_cert or not os.path.exists(ssl_cert):
    os.environ["SSL_CERT_FILE"] = certifi.where()
# --------------------------------------------------------

load_dotenv()


class Settings(BaseSettings):

    # API Keys
    google_api_key: str = ""
    tavily_api_key: str = ""
    ncbi_api_key: str = ""
    ncbi_email: str = ""

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "Therapisto"

    # Which LLM answers the questions: "ollama" (local) or "gemini" (API)
    llm_provider: str = "ollama"

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_num_ctx: int = 8192

    # Gemini
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_vision_model: str = "gemini-3.8-flash"
    gemini_fallback_models: str = ""

    # Backend Security
    api_token: str = ""

    # CORS
    allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()