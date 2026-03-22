from abc import ABC, abstractmethod
from typing import Any

class AbsGenerator(ABC):
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    @abstractmethod
    def generate_single(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str] | None = None,
        messages_list: list[list[dict[str, str]]] | None = None,
        schema: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pass