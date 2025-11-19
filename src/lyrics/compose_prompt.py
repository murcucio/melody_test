# src/compose_prompt.py
import os
from openai import OpenAI
from src.lyrics.lyrics_extractor import get_lyrics_from_mnemonic_plan

# Suno API 가사 길이 제한 (커스텀 모드)
MAX_LYRICS_LENGTH = 5000


def truncate_lyrics(lyrics: str, max_length: int = MAX_LYRICS_LENGTH) -> str:
    """
    가사를 최대 길이로 제한합니다.
    너무 길면 마지막 문장을 잘라서 자연스럽게 끝냅니다.
    """
    if len(lyrics) <= max_length:
        return lyrics
    
    # 최대 길이까지 자르기
    truncated = lyrics[:max_length]
    
    # 마지막 문장이 잘리지 않도록 조정
    # 마지막 줄바꿈이나 문장 끝을 찾아서 자르기
    last_newline = truncated.rfind('\n')
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    
    # 가장 마지막 문장 종료 기호 찾기
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    if last_sentence_end > max_length * 0.8:  # 80% 이상이면 문장 끝에서 자르기
        return truncated[:last_sentence_end + 1]
    elif last_newline > max_length * 0.8:  # 줄바꿈에서 자르기
        return truncated[:last_newline]
    else:
        # 그냥 최대 길이에서 자르기
        return truncated + "..."


def summarize_for_lyrics(text: str, api_key: str, max_length: int = MAX_LYRICS_LENGTH) -> str:
    """
    텍스트가 너무 길면 노래 가사로 만들 수 있도록 요약합니다.
    """
    if len(text) <= max_length:
        return text
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""다음 학습 자료를 노래 가사로 만들 수 있도록 핵심 내용만 간결하게 요약해주세요.
요약된 내용은 {max_length}자 이하여야 하며, 노래로 부를 수 있는 자연스러운 문장으로 작성해주세요.
중요한 정보는 빠뜨리지 말고, 반복되는 내용은 제거해주세요.

[원본 내용]
{text}

[요약된 가사]"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 학습 자료를 노래 가사로 변환하는 전문가입니다. 핵심 내용만 간결하게 요약하여 노래로 부를 수 있는 형태로 정리해줍니다."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2000,  # 충분한 토큰 할당
        )
        summarized = resp.choices[0].message.content.strip()
        
        # 요약 후에도 길면 잘라내기
        return truncate_lyrics(summarized, max_length)
    except Exception as e:
        # 요약 실패 시 그냥 잘라내기
        return truncate_lyrics(text, max_length)


def build_suno_payload(mnemonic_plan, study_text, final_lyrics: str = None, api_key: str = None, emotion_tags: list = None):
    """
    Build a request payload for Suno's song generation endpoint.
    Reference: https://api.sunoapi.org/docs
    
    Args:
        mnemonic_plan: 멜로디 가이드
        study_text: 학습 텍스트
        final_lyrics: 최종 가사 (제공되면 이걸 사용)
        api_key: OpenAI API 키 (가사가 너무 길 때 요약에 사용)
        emotion_tags: 감정 태그 리스트 (선택사항)
    """
    # 최종 가사가 제공되면 그걸 사용, 없으면 멜로디 가이드에서 추출
    if final_lyrics:
        lyrics = final_lyrics
    else:
        # 멜로디 가이드에서 최종 가창 가이드 가사(5번 항목) 추출
        lyrics = get_lyrics_from_mnemonic_plan(mnemonic_plan, study_text)
    
    # 가사 길이 확인 및 제한
    if len(lyrics) > MAX_LYRICS_LENGTH:
        if api_key:
            # API 키가 있으면 요약 시도
            lyrics = summarize_for_lyrics(lyrics, api_key, MAX_LYRICS_LENGTH)
        else:
            # API 키가 없으면 그냥 잘라내기
            lyrics = truncate_lyrics(lyrics, MAX_LYRICS_LENGTH)
    
    # 감정 태그를 스타일에 반영
    emotion_style_parts = []
    if emotion_tags and len(emotion_tags) > 0:
        # 감정 태그를 영어로 변환하여 스타일에 추가
        emotion_translations = {
            "통통튀는": "bouncy, upbeat",
            "신나는": "energetic, exciting",
            "슬픈": "sad, melancholic",
            "밝은": "bright, cheerful",
            "따뜻한": "warm, cozy",
            "차분한": "calm, peaceful",
            "활기찬": "lively, vibrant",
            "부드러운": "soft, gentle",
            "강렬한": "intense, powerful",
            "평화로운": "peaceful, serene",
            "에너지 넘치는": "high energy, dynamic",
            "로맨틱한": "romantic, tender",
            "웃긴": "funny, humorous",
            "장난스러운": "playful, mischievous",
            "진지한": "serious, solemn",
            "드라마틱한": "dramatic, theatrical",
            "몽환적인": "dreamy, ethereal",
            "격렬한": "fierce, intense",
            "우아한": "elegant, graceful",
            "자유로운": "free, liberating",
            "긴장감 있는": "tense, suspenseful",
            "편안한": "comfortable, relaxed",
            "신비로운": "mysterious, mystical",
            "웅장한": "grand, majestic",
            "섬세한": "delicate, refined",
            "역동적인": "dynamic, energetic",
            "감성적인": "emotional, sentimental",
            "경쾌한": "light, breezy",
            "잔잔한": "calm, tranquil",
            "열정적인": "passionate, fiery"
        }
        
        for tag in emotion_tags:
            if tag in emotion_translations:
                emotion_style_parts.append(emotion_translations[tag])
    
    # 기본 스타일 + 감정 태그 스타일
    # 두 개의 트랙을 생성: 하나는 여자 보컬, 다른 하나는 남자 보컬
    base_style_female = (
        "K-pop ballad / Korean language / Korean lyrics / "
        "warm female vocal / soft piano & strings / 85–92 BPM / "
        "bright educational jingle, clear Korean diction, playful synth pop, "
        "memorable hook, repetition for easy memorisation"
    )
    
    base_style_male = (
        "K-pop ballad / Korean language / Korean lyrics / "
        "warm male vocal / soft piano & strings / 85–92 BPM / "
        "bright educational jingle, clear Korean diction, playful synth pop, "
        "memorable hook, repetition for easy memorisation"
    )
    
    if emotion_style_parts:
        style_female = f"{base_style_female}, {', '.join(emotion_style_parts)}"
        style_male = f"{base_style_male}, {', '.join(emotion_style_parts)}"
    else:
        style_female = base_style_female
        style_male = base_style_male
    
    # 두 가지 스타일을 모두 포함 (Suno API가 두 개의 트랙을 생성하도록)
    # 첫 번째는 여자 보컬, 두 번째는 남자 보컬
    style = f"{style_female} | {style_male}"
    
    # callBackUrl 설정 (환경 변수에서 가져오거나 기본값 사용)
    callback_url = os.getenv("SUNO_CALLBACK_URL", "https://httpbin.org/post")
    
    payload = {
        "customMode": True,
        "instrumental": False,
        "model": "V4_5",  # V3_5 | V4 | V4_5 | V4_5PLUS | V5
        "style": style,
        "title": "Learning Song",
        # 커스텀 모드에서 prompt가 '가사'로 사용됨 (한국어 가사)
        "prompt": lyrics,
        "callBackUrl": callback_url,
        "callbackUrl": callback_url,  # 두 가지 형식 모두 지원
    }
    
    return payload