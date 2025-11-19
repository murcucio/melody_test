"""
동요 Vector DB 로더 및 RAG 검색 모듈
"""
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class DongyoVectorDB:
    """동요 Vector DB 클래스"""
    
    def __init__(self, embeddings_path: str = None, index_path: str = None):
        """
        Vector DB 초기화
        
        Args:
            embeddings_path: embeddings pickle 파일 경로
            index_path: FAISS index 파일 경로
        """
        if embeddings_path is None:
            embeddings_path = project_root / "data" / "dongyo_embeddings.pkl"
        if index_path is None:
            index_path = project_root / "data" / "dongyo_faiss.index"
        
        self.embeddings_path = Path(embeddings_path)
        self.index_path = Path(index_path)
        
        # 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """Vector DB 데이터 로드"""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings 파일을 찾을 수 없습니다: {self.embeddings_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index 파일을 찾을 수 없습니다: {self.index_path}")
        
        # 메타데이터 로드 (제목, 가사 특징 요약, 가사)
        with open(self.embeddings_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        # FAISS index 로드
        self.index = faiss.read_index(str(self.index_path))
        
        # 동요 개수 계산 (딕셔너리 또는 리스트 형태 모두 지원)
        if isinstance(self.metadata, dict):
            song_count = len(self.metadata.get("titles", []))
        else:
            song_count = len(self.metadata)
        
        print(f"✅ Vector DB 로드 완료: {song_count}개 동요")
    
    def search_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        유사한 동요 검색
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 상위 k개 결과
            
        Returns:
            유사한 동요 정보 리스트 (제목, 가사 특징 요약, 가사 포함)
        """
        # query_embedding을 2D 배열로 변환 (1, dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # float32로 변환
        query_embedding = query_embedding.astype(np.float32)
        
        # FAISS 검색
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for i, idx in enumerate(indices[0]):
            # numpy 타입을 Python 기본 타입으로 변환
            try:
                if hasattr(idx, 'item'):
                    idx_int = int(idx.item())
                else:
                    idx_int = int(idx)
            except (ValueError, TypeError):
                idx_int = int(idx)
            
            # metadata가 딕셔너리인지 리스트인지 확인
            if isinstance(self.metadata, dict):
                # 딕셔너리 형태인 경우
                if idx_int < len(self.metadata.get("titles", [])):
                    titles = self.metadata.get("titles", [])
                    lyrics_list = self.metadata.get("lyrics", [])
                    title = titles[idx_int] if idx_int < len(titles) else ""
                    lyrics = lyrics_list[idx_int] if idx_int < len(lyrics_list) else ""
                    feature_summary = ""
                else:
                    continue
            else:
                # 리스트 형태인 경우
                if idx_int < len(self.metadata):
                    meta = self.metadata[idx_int]
                    # 다양한 키 이름 지원 (제목, 가사 특징 요약, 가사)
                    title = meta.get("제목") if isinstance(meta, dict) else (meta.get("title") if isinstance(meta, dict) else "")
                    feature_summary = meta.get("가사 특징 요약") if isinstance(meta, dict) else (meta.get("feature_summary") if isinstance(meta, dict) else (meta.get("특징") if isinstance(meta, dict) else ""))
                    lyrics = meta.get("가사") if isinstance(meta, dict) else (meta.get("lyrics") if isinstance(meta, dict) else "")
                else:
                    continue
            
            # numpy 타입을 Python 기본 타입으로 변환
            try:
                if hasattr(distances[0][i], 'item'):
                    distance_float = float(distances[0][i].item())
                else:
                    distance_float = float(distances[0][i])
            except (ValueError, TypeError):
                distance_float = float(distances[0][i])
            
            result = {
                "index": idx_int,
                "distance": distance_float,
                "title": str(title) if title else "",
                "feature_summary": str(feature_summary) if feature_summary else "",
                "lyrics": str(lyrics) if lyrics else "",
            }
            results.append(result)
        
        return results

