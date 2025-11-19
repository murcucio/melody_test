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


def build_suno_payload(mnemonic_plan, study_text, final_lyrics: str = None, api_key: str = None):
    """
    Build a request payload for Suno's song generation endpoint.
    Reference: https://api.sunoapi.org/docs
    
    Args:
        mnemonic_plan: 멜로디 가이드
        study_text: 학습 텍스트
        final_lyrics: 최종 가사 (제공되면 이걸 사용)
        api_key: OpenAI API 키 (가사가 너무 길 때 요약에 사용)
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
    
    # 한국어 가사임을 명시적으로 표시하는 스타일 설정
    style = (
        "K-pop ballad / Korean language / Korean lyrics / "
        "warm female vocal / soft piano & strings / 85–92 BPM / "
        "bright educational jingle, clear Korean diction, playful synth pop, "
        "memorable hook, repetition for easy memorisation"
    )
    
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