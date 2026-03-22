

from __future__ import annotations

from typing import Dict, Optional, Tuple

from gam.prompts import MemoryAgent_PROMPT
from gam.schemas import (
    MemoryState, Page, MemoryUpdate, MemoryStore, PageStore,
    InMemoryMemoryStore, InMemoryPageStore, Retriever
)
from gam.generator import AbsGenerator

class MemoryAgent:

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        page_store: PageStore | None = None,
        generator: AbsGenerator | None = None,
        dir_path: Optional[str] = None,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for MemoryAgent")
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.page_store = page_store or InMemoryPageStore(dir_path=dir_path)
        self.generator = generator
        

        default_system_prompts = {
            "memory": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:

            self.system_prompts = {**default_system_prompts, **system_prompts}


    def memorize(self, message: str) -> MemoryUpdate:
        message = message.strip()
        state = self.memory_store.load()


        abstract, header, decorated_new_page = self._decorate(message, state)


        self.memory_store.add(abstract)


        page = Page(header=header, content=message, meta={"decorated": decorated_new_page})
        self.page_store.add(page)
        

        updated_state = self.memory_store.load()

        return MemoryUpdate(new_state=updated_state, new_page=page, debug={"decorated_page": decorated_new_page})


    def _decorate(self, message: str, memory_state: MemoryState) -> Tuple[str, str, str]:

        if memory_state.abstracts:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        else:
            memory_context = "No memory currently."
        

        system_prompt = self.system_prompts.get("memory")
        template_prompt = MemoryAgent_PROMPT.format(
            input_message=message,
            memory_context=memory_context
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        
        try:
            response = self.generator.generate_single(prompt=prompt)
            abstract = response.get("text", "").strip()
        except Exception as e:
            print(f"Error generating abstract: {e}")
            abstract = message[:200]
        

        header = f"[ABSTRACT] {abstract}".strip()
        decorated_new_page = f"{header}; {message}"
        return abstract, header, decorated_new_page