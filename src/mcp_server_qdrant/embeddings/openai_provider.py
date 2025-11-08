from __future__ import annotations

from typing import List

from openai import AsyncOpenAI, OpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.
    Uses the AsyncOpenAI client for runtime requests while probing the vector size
    synchronously during initialization to avoid async calls inside __init__.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._async_client = AsyncOpenAI()
        # Prime the vector size once so collection creation knows the dimension.
        sync_client = OpenAI()
        probe = sync_client.embeddings.create(
            model=self.model_name, input="vector-dimension-probe"
        )
        self._vector_size = len(probe.data[0].embedding)
        safe_model_name = (
            self.model_name.replace("/", "-").replace(":", "-").replace(" ", "-")
        )
        self._vector_name = f"openai-{safe_model_name}".lower()

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        response = await self._async_client.embeddings.create(
            model=self.model_name, input=documents
        )
        # Ensure results are returned in the same order as the inputs
        sorted_data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        return [list(item.embedding) for item in sorted_data]

    async def embed_query(self, query: str) -> list[float]:
        response = await self._async_client.embeddings.create(
            model=self.model_name, input=query
        )
        return list(response.data[0].embedding)

    def get_vector_name(self) -> str:
        return self._vector_name

    def get_vector_size(self) -> int:
        return self._vector_size
