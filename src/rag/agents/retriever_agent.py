"""
Retriever Agent
벡터 DB에서 관련된 가사 또는 특징 요약을 검색
"""
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from src.rag.vector_db import DongyoVectorDB


class RetrieverAgent:
    """검색 에이전트"""
    
    def __init__(
        self, 
        api_key: str,
        embeddings_path: str = None,
        index_path: str = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Args:
            api_key: OpenAI API 키
            embeddings_path: embeddings 파일 경로
            index_path: FAISS index 파일 경로
            embedding_model: 임베딩 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.db = DongyoVectorDB(embeddings_path=embeddings_path, index_path=index_path)
    
    def retrieve(self, search_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        벡터 DB에서 관련 문서 검색
        
        Args:
            search_query: 검색 쿼리
            top_k: 반환할 상위 k개 결과
            
        Returns:
            검색된 동요 정보 리스트
        """
        # 검색 쿼리를 임베딩으로 변환
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=search_query
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # 벡터 DB에서 검색
        results = self.db.search_similar(query_embedding, top_k=top_k)
        
        return results

