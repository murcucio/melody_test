"""
Reasoner Agent
Query Agent 결과와 Retriever Agent 결과를 통합하여 최종 답변 생성
"""
from typing import Dict, Any, List
from openai import OpenAI


class ReasonerAgent:
    """응답 조합 에이전트"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def reason(
        self,
        query_result: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        task_type: str = "lyrics_generation"
    ) -> Dict[str, Any]:
        """
        Query 결과와 검색된 문서를 통합하여 최종 답변 생성
        
        Args:
            query_result: Query Agent 결과
            retrieved_docs: Retriever Agent 결과
            task_type: 작업 유형 (lyrics_generation, style_analysis, recommendation 등)
            
        Returns:
            {
                "reasoning": 추론 과정,
                "recommendations": 추천 사항,
                "style_guide": 스타일 가이드,
                "context_summary": 컨텍스트 요약
            }
        """
        # 검색된 문서를 컨텍스트로 포맷팅
        context = ""
        if retrieved_docs:
            context = "\n[참고 동요 정보]\n"
            for i, doc in enumerate(retrieved_docs, 1):
                context += f"\n{i}. {doc['title']}\n"
                context += f"   특징: {doc['feature_summary']}\n"
        
        categories_str = ", ".join([
            f"{k}: {v}" for k, v in query_result.get("categories", {}).items() 
            if v
        ])
        
        prompt = f"""다음 정보를 바탕으로 {task_type} 작업을 위한 추론과 가이드를 생성해주세요.

[사용자 의도]
{query_result.get('intent', '가사 생성')}

[추출된 카테고리]
{categories_str if categories_str else "없음"}

[원본 질문]
{query_result.get('original_query', '')}
{context}

[출력 포맷]
다음 JSON 형식으로만 출력해주세요:
{{
    "reasoning": "검색된 동요들과 사용자 요청을 통합한 추론 과정",
    "recommendations": "가사 생성 시 고려할 사항들",
    "style_guide": "추천하는 스타일과 톤 (예: 밝고 경쾌한, 따뜻한, 교육적인 등)",
    "context_summary": "참고할 동요들의 공통 특징 요약"
}}

[요구사항]
- reasoning은 검색된 동요와 사용자 요청을 연결하는 논리적 추론
- recommendations는 구체적이고 실행 가능한 제안
- style_guide는 가사 생성 시 참고할 스타일
- JSON 형식만 출력하고 다른 설명은 하지 마세요"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "너는 검색된 정보와 사용자 요청을 통합하여 최적의 가이드를 제공하는 전문가입니다. JSON 형식으로만 답변합니다."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content.strip())
        
        return {
            "reasoning": result.get("reasoning", ""),
            "recommendations": result.get("recommendations", ""),
            "style_guide": result.get("style_guide", ""),
            "context_summary": result.get("context_summary", ""),
            "categories": query_result.get("categories", {}),
            "intent": query_result.get("intent", "")
        }


