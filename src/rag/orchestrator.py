"""
RAG Orchestrator
Multi-Agent System의 전체 흐름을 조율
"""
from typing import Dict, Any
from src.rag.agents.query_agent import QueryUnderstandingAgent
from src.rag.agents.retriever_agent import RetrieverAgent
from src.rag.agents.reasoner_agent import ReasonerAgent
from src.rag.agents.generator_agent import GeneratorAgent


class RAGOrchestrator:
    """RAG 전체 흐름 조율자"""
    
    def __init__(
        self,
        api_key: str,
        embeddings_path: str = None,
        index_path: str = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Args:
            api_key: OpenAI API 키
            embeddings_path: embeddings 파일 경로
            index_path: FAISS index 파일 경로
            model: 사용할 모델
        """
        self.query_agent = QueryUnderstandingAgent(api_key, model)
        self.retriever_agent = RetrieverAgent(api_key, embeddings_path, index_path)
        self.reasoner_agent = ReasonerAgent(api_key, model)
        self.generator_agent = GeneratorAgent(api_key, model)
    
    def generate_lyrics(
        self,
        study_text: str,
        top_k: int = 5,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        학습 텍스트로부터 가사 생성 (전체 RAG 파이프라인)
        
        Args:
            study_text: 학습 텍스트
            top_k: 검색할 상위 k개 동요
            use_rag: RAG 사용 여부
            
        Returns:
            {
                "lyrics": 생성된 가사,
                "query_result": Query Agent 결과,
                "retrieved_docs": 검색된 문서들,
                "reasoner_result": Reasoner Agent 결과
            }
        """
        if not use_rag:
            # RAG 없이 직접 생성
            lyrics = self.generator_agent.generate_lyrics(
                study_text,
                {"style_guide": "", "recommendations": ""},
                []
            )
            return {
                "lyrics": lyrics,
                "query_result": None,
                "retrieved_docs": [],
                "reasoner_result": None
            }
        
        # 1. Query Understanding Agent
        query_result = self.query_agent.process(study_text)
        
        # 2. Retriever Agent
        retrieved_docs = self.retriever_agent.retrieve(
            query_result["search_query"],
            top_k=top_k
        )
        
        # numpy 타입을 Python 기본 타입으로 변환 (JSON 직렬화를 위해)
        def convert_numpy_types(obj):
            """재귀적으로 numpy 타입을 Python 기본 타입으로 변환"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # retrieved_docs의 numpy 타입 변환
        retrieved_docs = convert_numpy_types(retrieved_docs)
        
        # 3. Reasoner Agent
        reasoner_result = self.reasoner_agent.reason(
            query_result,
            retrieved_docs,
            task_type="lyrics_generation"
        )
        
        # reasoner_result의 numpy 타입 변환
        reasoner_result = convert_numpy_types(reasoner_result)
        
        # 4. Generator Agent
        lyrics = self.generator_agent.generate_lyrics(
            study_text,
            reasoner_result,
            retrieved_docs
        )
        
        return {
            "lyrics": str(lyrics) if lyrics else "",
            "query_result": convert_numpy_types(query_result),
            "retrieved_docs": retrieved_docs,
            "reasoner_result": reasoner_result
        }

