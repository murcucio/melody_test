"""
Generator Agent
실제 가사/멜로디 생성 및 멜로디 가이드 생성
"""
from typing import Dict, Any, List, Optional
from openai import OpenAI


class GeneratorAgent:
    """노래/멜로디 생성 에이전트"""
    
    SYSTEM_CORE = (
        "너는 학습자를 위한 기억 보조 작곡가다. "
        "입력된 학습 텍스트를 쉽고 경쾌하게 외울 수 있도록 리듬, 멜로디, 반복 구조를 설계해라. "
        "한국어로 답하고, 간결하지만 구체적으로 안내해."
    )
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_lyrics(
        self,
        study_text: str,
        reasoner_result: Dict[str, Any] = None,
        retrieved_docs: List[Dict[str, Any]] = None
    ) -> str:
        """
        가사 생성
        
        Args:
            study_text: 학습 텍스트
            reasoner_result: Reasoner Agent 결과 (선택, 없으면 기본값 사용)
            retrieved_docs: 검색된 문서들 (선택, 없으면 기본값 사용)
            
        Returns:
            생성된 가사
        """
        # 기본값 설정
        if reasoner_result is None:
            reasoner_result = {"style_guide": "", "recommendations": ""}
        if retrieved_docs is None:
            retrieved_docs = []
        
        # 검색된 문서 컨텍스트 구성
        context = ""
        if retrieved_docs:
            context = "\n\n[참고할 동요들의 특징과 느낌]\n"
            for i, doc in enumerate(retrieved_docs, 1):
                context += f"\n{i}. {doc['title']}\n"
                context += f"   특징: {doc['feature_summary']}\n"
                if doc.get('lyrics'):
                    lyrics_preview = doc['lyrics'][:100] + "..." if len(doc['lyrics']) > 100 else doc['lyrics']
                    context += f"   가사 일부: {lyrics_preview}\n"
        
        # Reasoner 결과 활용
        style_guide = reasoner_result.get("style_guide", "")
        recommendations = reasoner_result.get("recommendations", "")
        
        prompt = f"""다음 학습용 텍스트를 노래 가사로 변환해주세요.

[학습 텍스트]
{study_text}
{context}

[스타일 가이드]
{style_guide if style_guide else "동요 스타일로 작성"}

[추천 사항]
{recommendations if recommendations else ""}

[요구사항]
- 학습 내용의 핵심을 모두 포함해야 합니다
- 노래로 부르기 쉬운 자연스러운 문장으로 작성해주세요
- 4~12줄 정도의 적절한 길이로 작성해주세요
- 반복되는 후렴구를 포함하면 더 좋습니다
- 학습자가 외우기 쉽도록 리듬감 있는 표현을 사용해주세요
- 한국어로 작성해주세요
{f"- 위에 제공된 참고 동요들의 특징과 느낌을 참고하여 비슷한 톤과 스타일로 작성해주세요" if context else ""}

[생성된 가사]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "너는 학습용 노래 가사를 만드는 전문 작사가입니다. 학습 내용을 노래로 부르기 쉬운 형태로 변환해줍니다."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        
        lyrics = response.choices[0].message.content.strip()
        
        # 불필요한 설명 제거 (가사만 추출)
        return self._clean_lyrics(lyrics)
    
    def _clean_lyrics(self, lyrics: str) -> str:
        """
        가사에서 불필요한 설명 제거
        
        Args:
            lyrics: 원본 가사
            
        Returns:
            정리된 가사
        """
        lines = lyrics.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 설명성 문구 제거
            if line and not line.startswith('[') and not line.startswith('(') and '가사' not in line:
                cleaned_lines.append(line)
        
        if cleaned_lines:
            return '\n'.join(cleaned_lines)
        
        return lyrics
    
    def generate_mnemonic_plan(
        self,
        study_text: str,
        final_lyrics: Optional[str] = None
    ) -> str:
        """
        멜로디 가이드 생성
        
        Args:
            study_text: 학습 텍스트
            final_lyrics: 이미 생성된 최종 가사 (있으면 포함)
            
        Returns:
            생성된 멜로디 가이드
        """
        if final_lyrics:
            # 가사가 이미 생성된 경우, 그 가사를 포함하여 멜로디 가이드 생성
            prompt = f"""
다음 학습용 텍스트와 생성된 노래 가사를 바탕으로 멜로디 가이드를 만들어라.

[학습 텍스트]
{study_text}

[생성된 최종 가사]
{final_lyrics}

[출력 포맷]
1) 요약 포인트 3~5개 (암기할 핵심 단위)
2) 추천 리듬/템포/박자 (예: 4/4, 90BPM, 스윙 등)
3) 음 높이 가이드 (계이름 또는 숫자음으로 한 줄, 필요한 경우 두 줄)
4) 반복 구조와 하이라이트 (후렴, 콜앤리스폰스 등)
5) 최종 가창 가이드 가사 (위에 제공된 생성된 최종 가사를 그대로 전체 표시)
6) 보너스 암기 팁 한 줄

조건:
- 5번 항목에는 위에 제공된 "생성된 최종 가사"를 그대로 전체 표시해야 합니다.
- 가사를 수정하거나 요약하지 말고 전체를 그대로 표시하세요.
- 음 높이는 초보자가 따라 부르기 쉽게 단계적으로 움직이도록 제안.
- 다른 설명은 하지 말고 위 포맷만 채워서 출력.
""".strip()
        else:
            # 가사가 없는 경우 기존 방식
            prompt = f"""
다음 학습용 텍스트를 빠르게 외울 수 있도록 멜로디 가이드를 만들어라.

[학습 텍스트]
{study_text}

[출력 포맷]
1) 요약 포인트 3~5개 (암기할 핵심 단위)
2) 추천 리듬/템포/박자 (예: 4/4, 90BPM, 스윙 등)
3) 음 높이 가이드 (계이름 또는 숫자음으로 한 줄, 필요한 경우 두 줄)
4) 반복 구조와 하이라이트 (후렴, 콜앤리스폰스 등)
5) 최종 가창 가이드 가사 (학습 텍스트를 적절히 변형하되 의미 유지, 4~8줄)
6) 보너스 암기 팁 한 줄

조건:
- 학습 텍스트의 핵심 용어는 가창 가이드에 반드시 포함.
- 음 높이는 초보자가 따라 부르기 쉽게 단계적으로 움직이도록 제안.
- 다른 설명은 하지 말고 위 포맷만 채워서 출력.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_CORE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()


