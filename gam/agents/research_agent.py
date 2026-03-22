

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

from gam.prompts import Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from gam.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from gam.generator import AbsGenerator

class ResearchAgent:

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: AbsGenerator | None = None,
        max_iters: int = 3,
        dir_path: Optional[str] = None,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        

        default_system_prompts = {
            "planning": "",
            "integration": "",
            "reflection": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:

            self.system_prompts = {**default_system_prompts, **system_prompts}


        for name, r in self.retrievers.items():
            try:

                r.build(self.page_store)
                print(f"Successfully built {name} retriever")
            except Exception as e:
                print(f"Failed to build {name} retriever: {e}")
                pass


    def research(self, request: str) -> ResearchOutput:

        self._update_retrievers()
        
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request

        for step in range(self.max_iters):

            memory_state = self.memory_store.load()
            plan = self._planning(next_request, memory_state)

            temp = self._search(plan, temp, request)

            decision = self._reflection(request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
            })

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request


        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    def _update_retrievers(self):

        current_page_count = len(self.page_store.load())
        

        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            print(f"检测到页面数量变化 ({self._last_page_count} -> {current_page_count})，更新检索器索引...")
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever index")
                except Exception as e:
                    print(f"❌ Failed to update {name} retriever: {e}")
        

        self._last_page_count = current_page_count

    def _safe_json_from_response(self, response: Dict[str, Any], debug_label: str = "") -> Dict[str, Any]:
        data = response.get("json")
        if data is not None and isinstance(data, dict):
            return data
        raw_text = (response.get("text") or "").strip()
        if not raw_text:
            if debug_label:
                print(f"[DEBUG] {debug_label} 返回空内容")
            return {}
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                start, end = raw_text.find("{"), raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(raw_text[start : end + 1])
            except Exception:
                pass
            if debug_label:
                print(f"[DEBUG] {debug_label} 返回非 JSON，前 200 字: {raw_text[:200]!r}")
            return {}


    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_prompt: Optional[str] = None
    ) -> SearchPlan:

        if not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        
        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(request=request, memory=memory_context)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        

        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4
        print(f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)")

        try:
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            data = self._safe_json_from_response(response, "Planning")
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", [])
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            )
    

    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        searching_prompt: Optional[str] = None
    ) -> Result:
        all_hits: List[Hit] = []


        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:

                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)

                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:

                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)

                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)

                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)


        if not all_hits:
            return result
        

        unique_hits: Dict[str, Hit] = {}
        hits_without_id: List[Hit] = []
        for hit in all_hits:
            if hit.page_id:

                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:

                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:

                hits_without_id.append(hit)
        

        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        

        return self._integrate(sorted_hits, result, question)

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        all_hits: List[Hit] = []


        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:

                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)

                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:

                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)

                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)

                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)


        if not all_hits:
            return result
        

        unique_hits: Dict[str, Hit] = {}
        hits_without_id: List[Hit] = []
        for hit in all_hits:
            if hit.page_id:

                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:

                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:

                hits_without_id.append(hit)
        
        evidence_text = []
        sources = []
        seen_sources = set()
        

        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        
        for i, hit in enumerate(sorted_hits, 1):

            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            

            if hit.page_id and hit.page_id not in seen_sources:
                sources.append(hit.page_id)
                seen_sources.add(hit.page_id)
        
        formatted_content = "\n".join(evidence_text)
        
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources
        )

    def _integrate(
        self, 
        hits: List[Hit], 
        result: Result, 
        question: str,
        integration_prompt: Optional[str] = None
    ) -> Result:
        
        evidence_text = []
        sources = []
        for i, hit in enumerate(hits, 1):

            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            if hit.page_id:
                sources.append(hit.page_id)
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "无搜索结果"
        
        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            data = self._safe_json_from_response(response, "Integration")
            

            llm_sources = data.get("sources", sources)
            if llm_sources:

                sources_list = []
                for s in llm_sources:
                    if s is not None:
                        sources_list.append(str(s))
                sources = sources_list if sources_list else sources
            else:
                sources = sources
            
            return Result(
                content=data.get("content", ""),
                sources=sources
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return result


    def _search_by_keyword(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:

                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in keyword search: {e}")
                return []

        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="keyword", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return []

        return []

    def _search_by_page_index(self, page_index: List[int]) -> List[List[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:

                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []
        

        out: List[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]
        
        
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_prompt: Optional[str] = None
    ) -> ReflectionDecision:
        
        try:
            system_prompt = self.system_prompts.get("reflection")
            

            result_content_chars = len(result.content)
            estimated_result_tokens = result_content_chars // 4
            print(f"[DEBUG] Reflection result.content length: {result_content_chars} chars (~{estimated_result_tokens} tokens)")
            

            template_check_prompt = InfoCheck_PROMPT.format(request=request, result=result.content)
            if system_prompt:
                check_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_check_prompt}"
            else:
                check_prompt = template_check_prompt
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            check_data = self._safe_json_from_response(check_response, "Reflection check")
            
            enough = check_data.get("enough", False)
            

            if enough:
                return ReflectionDecision(enough=True, new_request=None)
            

            template_generate_prompt = GenerateRequests_PROMPT.format(
                request=request, 
                result=result.content
            )
            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            generate_data = self._safe_json_from_response(generate_response, "Reflection generate")
            

            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            return ReflectionDecision(
                enough=False,
                new_request=new_request
            )
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            return ReflectionDecision(enough=False, new_request=None)