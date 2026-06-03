"""语义检索：869 条 description → Top-K"""
import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from src.llm_factory import load_local_llm_settings

MODEL_ENV = "SENTENCE_TRANSFORMER_MODEL"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "skills"
MODEL_MARKER_FILE = "_embedding_model.txt"
INDEX_VERSION = "v2"  # 索引格式变更时递增，触发全量重建
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "paraphrase-multilingual-MiniLM-L12-v2"

class SkillRetriever:
    def __init__(self, registry_path: str = "skill_registry.json"):
        # registry 负责保存技能元数据，后续同时构建按 id 查询的字典。
        self.registry = self._load(registry_path)
        self.registry_by_id = {s["id"]: s for s in self.registry if "id" in s}

        # 检索模型路径按“环境变量 > 本地配置文件 > 项目默认目录”优先级解析。
        local_settings = load_local_llm_settings()
        model_id = (
            os.environ.get(MODEL_ENV)
            or local_settings.get("sentence_transformer_model")
            or str(DEFAULT_MODEL_DIR)
        )
        model_id = str(Path(model_id))

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._ensure_collection(model_id)

        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_id
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.ef
        )
        self._index()

    def _ensure_collection(self, model_id: str):
        # 记录当前使用的 embedding 模型 + 索引版本；任一变更则重建 collection。
        marker_path = Path(CHROMA_DIR) / MODEL_MARKER_FILE
        marker_path.parent.mkdir(parents=True, exist_ok=True)

        current_marker = f"{model_id}|{INDEX_VERSION}"
        prev_marker = None
        if marker_path.exists():
            prev_marker = marker_path.read_text(encoding="utf-8").strip() or None

        if prev_marker and prev_marker != current_marker:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        marker_path.write_text(current_marker, encoding="utf-8")

    def _index(self):
        existing = set(self.collection.get()["ids"])
        new_skills = [s for s in self.registry if s["id"] not in existing]
        if not new_skills:
            return
        
        # 当前技能量不大，可一次性入库；description 为空时回退到 skill id，避免空文本嵌入。
        ids = []
        documents = []
        metadatas = []
        for s in new_skills:
            ids.append(s["id"])
            desc = s.get("description", "")
            # 索引文档拼接 ID，让 embedding 感知技能名称信号
            documents.append(f"{s['id']}. {desc}" if desc else s["id"])
            metadatas.append({
                "path": s["path"],
                "has_examples": s["has_examples"],
                "has_references": s["has_references"],
                "has_scripts": s["has_scripts"],
            })
            
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=k)
        if not results["ids"] or not results["ids"][0]:
            return []
            
        return [
            {"id": id_, "description": doc, **meta}
            for id_, doc, meta in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
            )
        ]

    def retrieve_with_scores(self, query: str, k: int = 3) -> tuple[list[dict], list[float]]:
        """检索技能并返回距离分数，用于相关性阈值过滤。

        ChromaDB 使用余弦距离：0 = 完全相同，2 = 完全相反。
        距离越小越相关，建议阈值为 0.7（余弦相似度 ≥ 0.3）。

        Returns:
            (skills_list, distances_list) — 两个等长列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        if not results["ids"] or not results["ids"][0]:
            return [], []

        skills = []
        for id_, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            skills.append({"id": id_, "description": doc, **meta})

        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        return skills, distances

    def get(self, skill_id: str) -> dict | None:
        return self.registry_by_id.get(skill_id)

    def _load(self, path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
