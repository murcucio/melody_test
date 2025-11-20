"""
Generator Agent
실제 가사/멜로디 생성 및 멜로디 가이드 생성
"""
from typing import Dict, Any, List, Optional
from collections import Counter
import re
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
        rhythm_pattern = reasoner_result.get("rhythm_pattern", "")
        melody_style = reasoner_result.get("melody_style", "")
        rhyme_scheme = reasoner_result.get("rhyme_scheme", "")
        
        # 단어장 형식 감지 (예: "apple : 사과", "book: 책" 등)
        is_vocabulary = self._detect_vocabulary_format(study_text)
        
        # 원본 텍스트에서 핵심 키워드 추출 (간단한 방법)
        # 문장 부호 제거 후 단어 추출
        words = re.findall(r'\b\w+\b', study_text)
        # 2글자 이상의 단어 중 빈도가 높은 것들 (최대 10개)
        word_freq = Counter([w for w in words if len(w) >= 2])
        key_terms = [word for word, count in word_freq.most_common(10)]
        key_terms_str = ", ".join(key_terms[:10]) if key_terms else ""
        
        # 단어장인 경우 특별한 프롬프트 사용
        if is_vocabulary:
            vocabulary_pairs = self._extract_vocabulary_pairs(study_text)
            vocabulary_list = "\n".join([f"- {pair['word']} : {pair['meaning']}" for pair in vocabulary_pairs[:20]])  # 최대 20개만 표시
            
            prompt = f"""다음은 단어장 형식의 학습 텍스트입니다. 이 단어들을 노래 가사로 변환해주세요.

[학습 텍스트 - 반드시 이 내용만 사용하세요]
{study_text}

[추출된 단어-뜻 쌍 - 반드시 모두 포함해야 합니다]
{vocabulary_list}

{context}

[스타일 가이드]
{style_guide if style_guide else "동요 스타일로 작성"}

[추천 사항]
{recommendations if recommendations else ""}

{f"[리듬 패턴]\n{rhythm_pattern}\n" if rhythm_pattern else ""}
{f"[가락 스타일]\n{melody_style}\n" if melody_style else ""}
{f"[운율 패턴]\n{rhyme_scheme}\n" if rhyme_scheme else ""}

[단어장 가사 생성 규칙 - 절대적으로 지켜야 합니다]
1. **원본 단어와 뜻만 사용**: 위에 나열된 단어와 뜻만 가사에 포함하세요. 원본 텍스트에 없는 단어, 인물명, 장소명, 조직명 등을 절대 추가하지 마세요.
2. **모든 단어-뜻 쌍 포함**: 가능한 한 많은 단어-뜻 쌍을 가사에 포함하세요.
3. **구조 유지**: 각 단어와 그 뜻을 함께 언급하세요 (예: "apple은 사과", "book은 책").
4. **추가 설명 완전 금지**: "선생님", "학교", "연합", "힘으로" 등 원본에 없는 단어나 문구를 절대 사용하지 마세요.
5. **배경 설명 금지**: 단어장의 출처, 작성자, 목적 등에 대한 설명을 추가하지 마세요.
6. **노래로 부르기 쉬운 형태**: 단어와 뜻을 리듬감 있게 반복하세요.
7. **후렴구**: 주요 단어들을 반복하는 후렴구를 만들되, 원본에 없는 내용은 포함하지 마세요.
8. **한국어**: 한국어로 작성해주세요.
{f"9. **스타일 참고**: 위에 제공된 참고 동요들의 톤과 스타일만 참고하되, 내용은 반드시 원본 단어장만 사용하세요." if context else ""}

[생성된 가사]"""
        else:
            # 일반 텍스트인 경우 기존 프롬프트 사용
            prompt = f"""다음 학습용 텍스트를 노래 가사로 변환해주세요.

[학습 텍스트 - 반드시 이 내용을 기반으로 가사를 작성하세요]
{study_text}

[핵심 키워드 - 반드시 포함해야 할 주요 단어들]
{key_terms_str if key_terms_str else "위 학습 텍스트의 모든 주요 내용"}

{context}

[스타일 가이드]
{style_guide if style_guide else "동요 스타일로 작성"}

[추천 사항]
{recommendations if recommendations else ""}

{f"[리듬 패턴]\n{rhythm_pattern}\n" if rhythm_pattern else ""}
{f"[가락 스타일]\n{melody_style}\n" if melody_style else ""}
{f"[운율 패턴]\n{rhyme_scheme}\n" if rhyme_scheme else ""}

[중요 요구사항 - 반드시 지켜야 합니다]
1. **원본 텍스트의 핵심 내용을 100% 반영**: 위에 제공된 "학습 텍스트"의 주요 정보, 사실, 개념을 모두 가사에 포함해야 합니다.
2. **핵심 키워드 필수 포함**: 위에 나열된 핵심 키워드들을 가능한 한 많이 가사에 포함하세요.
3. **추가 설명 금지**: 원본 텍스트에 없는 내용(인물명, 장소명, 조직명, 배경 설명 등)을 임의로 추가하거나 설명하지 마세요.
4. **정보 정확성**: 원본 텍스트의 정보를 왜곡하거나 변경하지 마세요.
5. **노래로 부르기 쉬운 형태**: 정보는 그대로 유지하되, 노래로 부르기 쉬운 자연스러운 문장으로 변환하세요.
6. **길이**: 4~12줄 정도의 적절한 길이로 작성해주세요.
7. **후렴구**: 핵심 내용을 반복하는 후렴구를 포함하면 더 좋습니다.
8. **리듬감**: 학습자가 외우기 쉽도록 리듬감 있는 표현을 사용해주세요.
9. **한국어**: 한국어로 작성해주세요.
{f"10. **스타일 참고**: 위에 제공된 참고 동요들의 톤과 스타일만 참고하되, 내용은 반드시 원본 학습 텍스트를 기반으로 작성하세요." if context else ""}

[생성된 가사]"""

        # 단어장인 경우 더 엄격한 system 메시지
        if is_vocabulary:
            system_message = (
                "너는 단어장을 노래 가사로 변환하는 전문 작사가입니다. "
                "원본 단어장에 있는 단어와 뜻만 사용하여 가사를 작성합니다. "
                "원본에 없는 인물명, 장소명, 조직명, 배경 설명 등을 절대 추가하지 않습니다. "
                "각 단어와 그 뜻을 리듬감 있게 반복하여 외우기 쉬운 가사를 만듭니다."
            )
        else:
            system_message = (
                "너는 학습용 노래 가사를 만드는 전문 작사가입니다. "
                "사용자가 제공한 학습 텍스트의 모든 핵심 내용을 정확하게 반영하여 가사를 작성합니다. "
                "원본 텍스트에 없는 내용을 추가하거나 정보를 왜곡하지 않으며, 제공된 텍스트의 주요 정보를 그대로 노래 가사 형태로 변환합니다. "
                "원본 텍스트의 핵심 키워드와 주요 정보를 반드시 포함해야 합니다."
            )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 if is_vocabulary else 0.5,  # 단어장은 더 낮은 temperature로 정확도 향상
            max_tokens=1000,
        )
        
        lyrics = response.choices[0].message.content.strip()
        
        # 불필요한 설명 제거 (가사만 추출)
        return self._clean_lyrics(lyrics)
    
    def _detect_vocabulary_format(self, text: str) -> bool:
        """
        단어장 형식인지 감지
        예: "apple : 사과", "book: 책", "word - 뜻" 등
        
        Args:
            text: 학습 텍스트
            
        Returns:
            단어장 형식이면 True
        """
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # 단어장 패턴: "단어 : 뜻" 또는 "단어: 뜻" 또는 "단어 - 뜻" 등
        vocabulary_pattern = re.compile(
            r'^[^\s:：\-—]+[\s]*[:：\-—]+[\s]*[^\s:：\-—]+',
            re.MULTILINE
        )
        
        matches = vocabulary_pattern.findall(text)
        # 전체 줄의 50% 이상이 단어장 형식이면 단어장으로 판단
        if len(matches) >= max(2, len(lines) * 0.5):
            return True
        
        return False
    
    def _extract_vocabulary_pairs(self, text: str) -> List[Dict[str, str]]:
        """
        단어장 텍스트에서 단어-뜻 쌍 추출
        
        Args:
            text: 단어장 형식의 텍스트
            
        Returns:
            [{"word": "apple", "meaning": "사과"}, ...] 형식의 리스트
        """
        pairs = []
        lines = text.strip().split('\n')
        
        # 다양한 구분자 패턴: ":", "：", "-", "—", " : ", ": " 등
        pattern = re.compile(r'^([^\s:：\-—]+)[\s]*[:：\-—]+[\s]*(.+)$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = pattern.match(line)
            if match:
                word = match.group(1).strip()
                meaning = match.group(2).strip()
                if word and meaning:
                    pairs.append({"word": word, "meaning": meaning})
        
        return pairs
    
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


