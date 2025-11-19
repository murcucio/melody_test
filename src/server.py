"""
FastAPI 백엔드 서버: 이미지에서 학습 텍스트 추출, 멜로디 가이드 생성, Mureka 노래 생성 API 제공
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import base64
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.core.mureka_utils import find_audio_urls
from src.core.workflow import (
    build_suno_request,
    create_mnemonic_plan,
    extract_study_text_from_base64,
    request_suno_song,
)
from src.processors.image_analyzer import analyze_multiple_images
from src.processors.pdf_processor import extract_text_from_pdf, is_pdf_file

load_dotenv()

app = FastAPI(title="학습용 멜로디 생성 API")

# 프론트엔드(web 폴더) 경로
FRONTEND_DIR = project_root / "web"

# 정적 파일 서빙: /static/main.js 처럼 접근
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)

# 루트(/)에서 index.html 반환
@app.get("/", include_in_schema=False)
async def serve_front():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html 파일을 찾을 수 없습니다.")
    return FileResponse(index_path)

# CORS 설정: 웹 프론트엔드에서 접근 가능하도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractTextRequest(BaseModel):
    image_base64: str


class ExtractTextResponse(BaseModel):
    study_text: str


class MnemonicPlanRequest(BaseModel):
    study_text: str


class MnemonicPlanResponse(BaseModel):
    mnemonic_plan: str


class GenerateSongRequest(BaseModel):
    study_text: str
    mnemonic_plan: str
    wait_for_audio: bool = True


class GenerateSongResponse(BaseModel):
    task_id: Optional[str] = None
    audio_urls: list[str] = []
    status: str = "completed"


def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY가 설정되지 않았습니다.")
    return key


def get_suno_key() -> Optional[str]:
    return os.getenv("SUNO_API_KEY")


@app.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(req: ExtractTextRequest) -> ExtractTextResponse:
    """이미지(base64)에서 학습용 텍스트 추출"""
    try:
        api_key = get_openai_key()
        study_text = extract_study_text_from_base64(req.image_base64, api_key)
        if not study_text.strip():
            raise HTTPException(status_code=400, detail="텍스트를 추출하지 못했습니다.")
        return ExtractTextResponse(study_text=study_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 추출 실패: {str(e)}")


@app.post("/extract-from-files", response_model=ExtractTextResponse)
async def extract_from_files(files: List[UploadFile] = File(...)) -> ExtractTextResponse:
    """
    다중 파일(이미지 최대 5장, PDF 1개)에서 학습용 텍스트 추출 및 종합
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="파일이 업로드되지 않았습니다.")
        
        # 파일 타입 확인
        images = []
        pdfs = []
        
        for file in files:
            if is_pdf_file(file.filename):
                pdfs.append(file)
            elif file.content_type and file.content_type.startswith("image/"):
                images.append(file)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"지원하지 않는 파일 형식입니다: {file.filename}"
                )
        
        # 제한 확인
        if len(images) > 5:
            raise HTTPException(status_code=400, detail="이미지는 최대 5장까지 업로드할 수 있습니다.")
        if len(pdfs) > 1:
            raise HTTPException(status_code=400, detail="PDF는 최대 1개까지 업로드할 수 있습니다.")
        
        api_key = get_openai_key()
        all_texts = []
        
        # PDF 처리
        for pdf_file in pdfs:
            try:
                pdf_bytes = await pdf_file.read()
                if not pdf_bytes or len(pdf_bytes) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF 파일이 비어있습니다: {pdf_file.filename}"
                    )
                
                pdf_text = extract_text_from_pdf(pdf_bytes)
                if pdf_text.strip():
                    all_texts.append(f"[PDF: {pdf_file.filename}]\n{pdf_text}")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF 파일에서 텍스트를 추출하지 못했습니다: {pdf_file.filename}. "
                               "이미지로만 구성된 PDF이거나 텍스트가 없는 PDF일 수 있습니다."
                    )
            except ImportError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF 처리 라이브러리가 설치되지 않았습니다. "
                           "다음 명령어로 설치해주세요: pip install pdfplumber PyPDF2"
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF 처리 실패 ({pdf_file.filename}): {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF 파일 처리 중 오류 발생 ({pdf_file.filename}): {str(e)}"
                )
        
        # 이미지 처리
        if images:
            image_b64_list = []
            for img_file in images:
                img_bytes = await img_file.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                image_b64_list.append(img_b64)
            
            # 여러 이미지 분석 및 종합
            if len(image_b64_list) == 1:
                # 단일 이미지: 간단한 분석
                from src.processors.image_analyzer import analyze_image_for_education
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                img_text = analyze_image_for_education(image_b64_list[0], client)
                if img_text.strip():
                    all_texts.append(f"[이미지: {images[0].filename}]\n{img_text}")
            else:
                # 다중 이미지: 종합 분석
                img_text = analyze_multiple_images(image_b64_list, api_key)
                if img_text.strip():
                    all_texts.append(f"[이미지 {len(images)}장 종합]\n{img_text}")
        
        if not all_texts:
            raise HTTPException(status_code=400, detail="파일에서 내용을 추출하지 못했습니다.")
        
        # 모든 내용 종합
        if len(all_texts) == 1:
            study_text = all_texts[0]
        else:
            # 여러 파일 내용을 종합하여 요약
            combined_text = "\n\n".join(all_texts)
            
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            summary_prompt = f"""다음은 여러 학습 자료(이미지, PDF)에서 추출한 내용입니다.
이 내용들을 종합하여 하나의 일관된 학습 자료로 정리해주세요.
중복되는 내용은 제거하고, 핵심 내용만 간결하게 정리해주세요.
노래 가사로 만들 수 있도록 자연스러운 문장으로 작성해주세요.

[추출된 내용]
{combined_text}

[요약된 학습 자료]"""

            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "너는 교육 자료 요약 전문가입니다. 여러 자료를 종합하여 학습자가 쉽게 외울 수 있는 형태로 정리해줍니다."
                        },
                        {"role": "user", "content": summary_prompt},
                    ],
                    temperature=0.5,
                )
                study_text = resp.choices[0].message.content.strip()
            except Exception:
                # 요약 실패 시 원본 텍스트 반환
                study_text = combined_text
        
        if not study_text.strip():
            raise HTTPException(status_code=400, detail="파일에서 내용을 추출하지 못했습니다.")
        
        return ExtractTextResponse(study_text=study_text)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 실패: {str(e)}")


@app.post("/mnemonic-plan", response_model=MnemonicPlanResponse)
async def mnemonic_plan(req: MnemonicPlanRequest) -> MnemonicPlanResponse:
    """학습 텍스트로부터 가사를 먼저 생성하고, 그 가사를 포함한 멜로디 가이드 생성"""
    try:
        api_key = get_openai_key()
        
        # 1. 가사를 먼저 생성
        from src.rag.orchestrator import RAGOrchestrator
        orchestrator = RAGOrchestrator(api_key=api_key)
        result = orchestrator.generate_lyrics(req.study_text, top_k=3, use_rag=True)
        final_lyrics = result["lyrics"]
        
        # 2. 생성된 가사를 포함하여 멜로디 가이드 생성
        plan = create_mnemonic_plan(req.study_text, api_key, final_lyrics=final_lyrics)
        
        return MnemonicPlanResponse(mnemonic_plan=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"멜로디 가이드 생성 실패: {str(e)}")


@app.post("/generate-song", response_model=GenerateSongResponse)
async def generate_song(req: GenerateSongRequest) -> GenerateSongResponse:
    """Suno API를 사용해 노래 생성"""
    suno_key = get_suno_key()
    if not suno_key:
        raise HTTPException(status_code=500, detail="SUNO_API_KEY가 설정되지 않았습니다.")

    try:
        # OpenAI API 키를 가져와서 가사 길이 제한 시 요약에 사용
        openai_key = get_openai_key()
        
        # 멜로디 가이드에서 최종 가사 추출
        from src.lyrics.lyrics_extractor import extract_final_lyrics
        final_lyrics = extract_final_lyrics(req.mnemonic_plan)
        if not final_lyrics:
            # 추출 실패 시 가사를 다시 생성
            from src.rag.agents.generator_agent import GeneratorAgent
            generator_agent = GeneratorAgent(api_key=openai_key)
            final_lyrics = generator_agent.generate_lyrics(req.study_text)
        
        payload = build_suno_request(req.study_text, req.mnemonic_plan, final_lyrics=final_lyrics, api_key=openai_key)
        result = request_suno_song(payload, suno_key, wait=req.wait_for_audio)

        # Suno 응답에서 오디오 URL 추출
        audio_urls = []
        if "tracks" in result:
            for track in result["tracks"]:
                if "audioUrl" in track and track["audioUrl"]:
                    audio_urls.append(track["audioUrl"])
        # fallback: 기존 find_audio_urls도 시도
        if not audio_urls:
            audio_urls = find_audio_urls(result)

        if req.wait_for_audio:
            return GenerateSongResponse(
                task_id=result.get("task_id") or result.get("id"),
                audio_urls=audio_urls,
                status=result.get("status", "completed"),
            )
        else:
            return GenerateSongResponse(
                task_id=result.get("task_id") or result.get("id"),
                audio_urls=[],
                status="pending",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"노래 생성 실패: {str(e)}")


@app.get("/api-info")
async def root() -> Dict[str, Any]:
    """루트 엔드포인트: API 정보 제공"""
    return {
        "message": "학습용 멜로디 생성 API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract-text": "이미지에서 텍스트 추출",
            "POST /extract-from-files": "다중 파일(이미지/PDF)에서 텍스트 추출 및 종합",
            "POST /mnemonic-plan": "멜로디 가이드 생성",
            "POST /generate-song": "Suno 노래 생성",
            "GET /health": "헬스 체크",
        },
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}

