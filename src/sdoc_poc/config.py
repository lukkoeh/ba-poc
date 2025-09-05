import os

class Config:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        self.dry_run = os.getenv("DRY_RUN", "0") == "1"
        self.max_docs = int(os.getenv("MAX_DOCS", "0"))
        self.artifacts = os.getenv("ARTIFACTS_DIR", "artifacts")
        os.makedirs(self.artifacts, exist_ok=True)
