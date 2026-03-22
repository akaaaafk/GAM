import os
import json
import subprocess
import shutil
import sys
import time
from typing import Dict, Any, List

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    LuceneSearcher = None

from gam.retriever.base import AbsRetriever
from gam.schemas import InMemoryPageStore, Hit, Page


def _safe_rmtree(path: str, max_retries: int = 3, delay: float = 0.5) -> None:
    if not os.path.exists(path):
        return
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)

            if not os.path.exists(path):
                return
            time.sleep(delay)
        except OSError as e:
            if attempt == max_retries - 1:

                try:

                    import subprocess
                    subprocess.run(['rm', '-rf', path], check=False, capture_output=True)
                    if not os.path.exists(path):
                        return
                except Exception:
                    pass
                raise OSError(f"无法删除目录 {path}: {e}")
            time.sleep(delay)


class BM25Retriever(AbsRetriever):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if LuceneSearcher is None:
            raise ImportError("BM25Retriever requires pyserini to be installed")
        self.index_dir = self.config["index_dir"]
        self.searcher: LuceneSearcher | None = None
        self.pages: List[Page] = []

    def _pages_dir(self):
        return os.path.join(self.index_dir, "pages")

    def _lucene_dir(self):
        return os.path.join(self.index_dir, "index")

    def _docs_dir(self):
        return os.path.join(self.index_dir, "documents")

    def load(self) -> None:

        if not os.path.exists(self._lucene_dir()):
            raise RuntimeError("BM25 index not found, need build() first.")
        self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        self.searcher = LuceneSearcher(self._lucene_dir())

    def build(self, page_store: InMemoryPageStore) -> None:


        _safe_rmtree(self._lucene_dir())
        _safe_rmtree(self._docs_dir())
        

        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self._docs_dir(), exist_ok=True)


        pages = page_store.load()
        docs_path = os.path.join(self._docs_dir(), "documents.jsonl")
        with open(docs_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(pages):
                text = p.content
                json.dump({"id": str(i), "contents": text}, f, ensure_ascii=False)
                f.write("\n")


        os.makedirs(self._lucene_dir(), exist_ok=True)


        input_dir = os.path.abspath(self._docs_dir()).replace("\\", "/")
        index_dir = os.path.abspath(self._lucene_dir()).replace("\\", "/")
        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", input_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.config.get("threads", 1)),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        

        max_build_retries = 2
        for attempt in range(max_build_retries):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt == max_build_retries - 1:
                    print(f"[ERROR] Pyserini 索引构建失败:")
                    print(f"  stdout: {e.stdout}")
                    print(f"  stderr: {e.stderr}")
                    raise
                print(f"[WARN] Pyserini 索引构建失败，重试 {attempt + 1}/{max_build_retries}...")

                _safe_rmtree(self._lucene_dir())
                os.makedirs(self._lucene_dir(), exist_ok=True)
                time.sleep(1)


        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(pages)
        

        self.pages = pages
        self.searcher = LuceneSearcher(self._lucene_dir())

    def update(self, page_store: InMemoryPageStore) -> None:


        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if self.searcher is None:

            self.load()

        results_all: List[List[Hit]] = []
        for q in query_list:
            q = q.strip()
            if not q:
                results_all.append([])
                continue

            hits_for_q = []
            py_hits = self.searcher.search(q, k=top_k)
            for rank, h in enumerate(py_hits):

                idx = int(h.docid)
                if idx < 0 or idx >= len(self.pages):
                    continue
                page = self.pages[idx]
                snippet = page.content
                hits_for_q.append(
                    Hit(
                        page_id=str(idx),
                        snippet=snippet,
                        source="keyword",
                        meta={"rank": rank, "score": float(h.score)}
                    )
                )
            results_all.append(hits_for_q)
        return results_all
