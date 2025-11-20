const statusEl = document.getElementById("status");
const backendUrlEl = document.getElementById("backend-url");
const backendBase = window.BACKEND_BASE || "http://localhost:8000";

backendUrlEl.textContent = backendBase;

const generateBtn = document.getElementById("generate-btn");
const generateBtnText = document.getElementById("generate-btn-text");
const generateBtnSpinner = document.getElementById("generate-btn-spinner");
const cancelBtn = document.getElementById("cancel-btn");
const generateMelodyBtn = document.getElementById("generate-melody-btn");
const melodyBtnText = document.getElementById("melody-btn-text");
const melodyBtnSpinner = document.getElementById("melody-btn-spinner");
const fileInput = document.getElementById("file-input");
const fileListEl = document.getElementById("file-list");
const textInput = document.getElementById("text-input");
const charCountEl = document.getElementById("char-count");
const studyTextEl = document.getElementById("study-text");
const lyricsTextEl = document.getElementById("lyrics-text");
const planTextEl = document.getElementById("plan-text");
const audioContainer = document.getElementById("audio-output");

// 선택된 파일들 관리
let selectedFiles = [];
// 가사 생성 취소를 위한 AbortController
let lyricsAbortController = null;
// 생성된 가사와 학습 텍스트 저장
let generatedLyrics = null;
let currentStudyText = null;
// 검색된 동요 정보 저장 (멜로디 생성 시 활용)
let retrievedDocs = null;
let reasonerResult = null;
// 선택된 감정 태그
let selectedEmotionTags = [];

// 감정 태그 목록 (30개)
const emotionTags = [
  "통통튀는", "신나는", "슬픈", "밝은", "따뜻한", "차분한", "활기찬", 
  "부드러운", "강렬한", "평화로운", "에너지 넘치는", "로맨틱한", 
  "웃긴", "장난스러운", "진지한", "드라마틱한", "몽환적인", 
  "격렬한", "우아한", "자유로운", "긴장감 있는", "편안한", 
  "신비로운", "웅장한", "섬세한", "역동적인", "감성적인", 
  "경쾌한", "잔잔한", "열정적인"
];

function setStatus(message) {
  statusEl.textContent = message;
}

function setPre(el, value) {
  el.textContent = value?.trim() || "-";
}

function clearAudio() {
  audioContainer.innerHTML = "";
  audioContainer.append("없음");
}

function downloadAudio(url, filename) {
  // 오디오 다운로드 함수
  fetch(url)
    .then(response => response.blob())
    .then(blob => {
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = filename || "learning-song.mp3";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(downloadUrl);
    })
    .catch(error => {
      console.error("다운로드 실패:", error);
      alert("오디오 다운로드에 실패했습니다.");
    });
}

function renderAudio(urls) {
  audioContainer.innerHTML = "";
  if (!urls || urls.length === 0) {
    audioContainer.append("없음");
    return;
  }
  urls.forEach((url, index) => {
    const audioWrapper = document.createElement("div");
    audioWrapper.style.marginBottom = "16px";
    
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = url;
    audio.style.width = "100%";
    audio.style.marginBottom = "8px";
    
    const downloadBtn = document.createElement("button");
    downloadBtn.textContent = "다운로드";
    downloadBtn.style.cssText = `
      background: #10b981;
      color: white;
      border: none;
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.9rem;
      cursor: pointer;
      margin-top: 4px;
    `;
    downloadBtn.onmouseover = () => {
      downloadBtn.style.background = "#059669";
    };
    downloadBtn.onmouseout = () => {
      downloadBtn.style.background = "#10b981";
    };
    
    const filename = `learning-song-${index + 1}.mp3`;
    downloadBtn.onclick = () => downloadAudio(url, filename);
    
    audioWrapper.appendChild(audio);
    audioWrapper.appendChild(downloadBtn);
    audioContainer.appendChild(audioWrapper);
  });
}

// 파일 목록 업데이트
function updateFileList() {
  fileListEl.innerHTML = "";
  if (selectedFiles.length === 0) {
    return;
  }
  
  selectedFiles.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = "file-item";
    
    const fileName = document.createElement("span");
    fileName.className = "file-name";
    fileName.textContent = file.name;
    
    const fileType = document.createElement("span");
    fileType.className = "file-type";
    fileType.textContent = file.type === "application/pdf" ? "PDF" : "이미지";
    
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "삭제";
    removeBtn.onclick = () => {
      selectedFiles.splice(index, 1);
      updateFileList();
      updateFileInput();
    };
    
    item.appendChild(fileName);
    item.appendChild(fileType);
    item.appendChild(removeBtn);
    fileListEl.appendChild(item);
  });
}

// 파일 입력 업데이트 (DataTransfer 사용)
function updateFileInput() {
  const dt = new DataTransfer();
  selectedFiles.forEach(file => dt.items.add(file));
  fileInput.files = dt.files;
}

// 파일 선택 이벤트
fileInput.addEventListener("change", (e) => {
  const files = Array.from(e.target.files);
  const images = files.filter(f => f.type.startsWith("image/"));
  const pdfs = files.filter(f => f.type === "application/pdf");
  
  // 이미지 개수 확인 (최대 5장)
  const currentImages = selectedFiles.filter(f => f.type.startsWith("image/"));
  if (currentImages.length + images.length > 5) {
    alert("이미지는 최대 5장까지 업로드할 수 있습니다.");
    return;
  }
  
  // PDF 개수 확인 (최대 1개)
  const currentPdfs = selectedFiles.filter(f => f.type === "application/pdf");
  if (currentPdfs.length + pdfs.length > 1) {
    alert("PDF는 최대 1개까지 업로드할 수 있습니다.");
    return;
  }
  
  // 새 파일 추가
  selectedFiles.push(...files);
  updateFileList();
  updateFileInput();
});

// 글자수 카운터 업데이트
function updateCharCount() {
  const length = textInput.value.length;
  const maxLength = 300;
  charCountEl.textContent = `${length} / ${maxLength}`;
  
  // 색상 변경
  charCountEl.classList.remove("warning", "error");
  if (length > maxLength * 0.9) {
    charCountEl.classList.add("error");
  } else if (length > maxLength * 0.7) {
    charCountEl.classList.add("warning");
  }
}

// 텍스트 입력란 글자수 카운터 이벤트
textInput.addEventListener("input", updateCharCount);
textInput.addEventListener("paste", () => {
  setTimeout(updateCharCount, 0);
});

function setButtonLoading(button, textEl, spinnerEl, isLoading, loadingText = null) {
  if (isLoading) {
    button.disabled = true;
    if (loadingText) {
      textEl.textContent = loadingText;
    }
    spinnerEl.style.display = "inline-block";
    spinnerEl.classList.add("spinner");
  } else {
    button.disabled = false;
    spinnerEl.style.display = "none";
    spinnerEl.classList.remove("spinner");
  }
}

function disableControls() {
  generateBtn.disabled = true;
  fileInput.disabled = true;
  textInput.disabled = true;
}

function enableControls() {
  generateBtn.disabled = false;
  fileInput.disabled = false;
  textInput.disabled = false;
}

function showCancelButton() {
  cancelBtn.style.display = "inline-block";
}

function hideCancelButton() {
  cancelBtn.style.display = "none";
}

function showMelodyButton() {
  generateMelodyBtn.style.display = "inline-block";
}

function hideMelodyButton() {
  generateMelodyBtn.style.display = "none";
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("파일을 읽을 수 없습니다."));
    reader.readAsDataURL(file);
  });
}

async function postJSON(path, payload, signal = null) {
  const options = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
  
  if (signal) {
    options.signal = signal;
  }
  
  const resp = await fetch(`${backendBase}${path}`, options);

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${path} 요청 실패 (${resp.status}): ${text}`);
  }

  return resp.json();
}

async function postFormData(path, formData) {
  const resp = await fetch(`${backendBase}${path}`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${path} 요청 실패 (${resp.status}): ${text}`);
  }

  return resp.json();
}

async function handleGenerate() {
  try {
    disableControls();
    hideMelodyButton();
    
    // 텍스트 입력 또는 파일 업로드 확인
    const inputText = textInput.value.trim();
    const hasFiles = selectedFiles.length > 0;
    
    if (!inputText && !hasFiles) {
      throw new Error("텍스트를 입력하거나 파일을 업로드해주세요.");
    }
    
    if (inputText.length > 300) {
      throw new Error("텍스트는 300자를 초과할 수 없습니다.");
    }

    clearAudio();
    setPre(studyTextEl, "-");
    setPre(lyricsTextEl, "-");
    setPre(planTextEl, "-");
    generatedLyrics = null;
    currentStudyText = null;
    retrievedDocs = null;
    reasonerResult = null;

    let studyText = "";
    
    // 텍스트 입력이 있으면 우선 사용
    if (inputText) {
      studyText = inputText;
      setStatus("입력된 텍스트 사용 중...");
      setPre(studyTextEl, studyText);
    } else if (hasFiles) {
      // 파일이 있으면 다중 파일 처리
      setStatus("파일 분석 중...");
      setButtonLoading(generateBtn, generateBtnText, generateBtnSpinner, true, "파일 분석 중...");
      
      const formData = new FormData();
      selectedFiles.forEach((file, index) => {
        formData.append("files", file);
      });
      
      const extractResp = await postFormData("/extract-from-files", formData);
      studyText = extractResp.study_text?.trim();
      
      if (!studyText) {
        throw new Error("파일에서 내용을 추출하지 못했습니다.");
      }
      
      setPre(studyTextEl, studyText);
      setButtonLoading(generateBtn, generateBtnText, generateBtnSpinner, false);
      generateBtnText.textContent = "가사 생성";
    }

    if (!studyText) {
      throw new Error("처리할 텍스트가 없습니다.");
    }

    currentStudyText = studyText;
    
    // 가사 생성 (취소 가능)
    setStatus("가사 생성 중...");
    setButtonLoading(generateBtn, generateBtnText, generateBtnSpinner, true, "가사 생성 중...");
    showCancelButton();
    
    // AbortController 생성
    lyricsAbortController = new AbortController();
    
    try {
      const lyricsResp = await postJSON(
        "/generate-lyrics", 
        { study_text: studyText },
        lyricsAbortController.signal
      );
      
      generatedLyrics = lyricsResp.lyrics || "";
      // 검색된 동요 정보 저장 (멜로디 생성 시 활용)
      retrievedDocs = lyricsResp.retrieved_docs || null;
      reasonerResult = lyricsResp.reasoner_result || null;
      setPre(lyricsTextEl, generatedLyrics);
      setStatus("가사 생성 완료! 멜로디 생성을 진행하시겠습니까?");
      setButtonLoading(generateBtn, generateBtnText, generateBtnSpinner, false);
      generateBtnText.textContent = "가사 생성";
      showMelodyButton();
    } catch (error) {
      if (error.name === 'AbortError') {
        setStatus("가사 생성이 중단되었습니다.");
        setPre(lyricsTextEl, "-");
      } else {
        throw error;
      }
    } finally {
      hideCancelButton();
      lyricsAbortController = null;
      setButtonLoading(generateBtn, generateBtnText, generateBtnSpinner, false);
      generateBtnText.textContent = "가사 생성";
      enableControls();
    }
  } catch (error) {
    console.error(error);
    setStatus(`에러: ${error.message || error}`);
    hideCancelButton();
    lyricsAbortController = null;
    setButtonLoading(generateBtn, generateBtnText, generateBtnSpinner, false);
    generateBtnText.textContent = "가사 생성";
  } finally {
    enableControls();
  }
}

async function handleCancel() {
  if (lyricsAbortController) {
    lyricsAbortController.abort();
    hideCancelButton();
    setStatus("가사 생성 중단 중...");
  }
}

async function handleGenerateMelody() {
  if (!generatedLyrics || !currentStudyText) {
    setStatus("먼저 가사를 생성해주세요.");
    return;
  }
  
  try {
    disableControls();
    setButtonLoading(generateMelodyBtn, melodyBtnText, melodyBtnSpinner, true, "멜로디 생성 중...");
    
    setStatus("멜로디 가이드 생성 중...");
    
    // 멜로디 가이드 생성 (이미 생성된 가사 사용)
    const planResp = await postJSON("/mnemonic-plan", { 
      study_text: currentStudyText,
      lyrics: generatedLyrics
    });
    const mnemonicPlan = planResp.mnemonic_plan || "";
    setPre(planTextEl, mnemonicPlan);

    setStatus("Suno 노래 생성 중...");
    const songResp = await postJSON("/generate-song", {
      study_text: currentStudyText,
      mnemonic_plan: mnemonicPlan,
      lyrics: generatedLyrics,  // 생성된 가사를 직접 전달 (중요!)
      wait_for_audio: true,  // 멜로디 생성은 항상 완료까지 대기
      emotion_tags: selectedEmotionTags,  // 선택된 감정 태그 전달
      retrieved_docs: retrievedDocs,  // 검색된 동요 정보 전달
      reasoner_result: reasonerResult,  // 추론 결과 전달
    });
    renderAudio(songResp.audio_urls || []);

    if (songResp.audio_urls && songResp.audio_urls.length > 0) {
      setStatus("완료! 곡을 재생해보세요.");
    } else {
      setStatus("생성 완료. 오디오 URL을 응답에서 찾지 못했습니다.");
    }
  } catch (error) {
    console.error(error);
    setStatus(`에러: ${error.message || error}`);
  } finally {
    setButtonLoading(generateMelodyBtn, melodyBtnText, melodyBtnSpinner, false);
    melodyBtnText.textContent = "멜로디 생성";
    enableControls();
  }
}


// 감정 태그 UI 초기화
function initEmotionTags() {
  const container = document.getElementById("emotion-tags-container");
  const countDisplay = document.getElementById("emotion-tags-count");
  
  if (!container) return;
  
  container.innerHTML = "";
  
  emotionTags.forEach(tag => {
    const tagBtn = document.createElement("button");
    tagBtn.type = "button";
    tagBtn.className = "emotion-tag";
    tagBtn.textContent = tag;
    tagBtn.dataset.tag = tag;
    
    tagBtn.addEventListener("click", () => {
      toggleEmotionTag(tag);
    });
    
    container.appendChild(tagBtn);
  });
  
  updateEmotionTagsCount();
}

// 감정 태그 토글
function toggleEmotionTag(tag) {
  const index = selectedEmotionTags.indexOf(tag);
  
  if (index > -1) {
    // 이미 선택된 태그면 제거
    selectedEmotionTags.splice(index, 1);
  } else {
    // 최대 5개까지만 선택 가능
    if (selectedEmotionTags.length >= 5) {
      alert("감정 태그는 최대 5개까지 선택할 수 있습니다.");
      return;
    }
    selectedEmotionTags.push(tag);
  }
  
  updateEmotionTagsUI();
  updateEmotionTagsCount();
}

// 감정 태그 UI 업데이트
function updateEmotionTagsUI() {
  const tags = document.querySelectorAll(".emotion-tag");
  tags.forEach(tagBtn => {
    const tag = tagBtn.dataset.tag;
    if (selectedEmotionTags.includes(tag)) {
      tagBtn.classList.add("selected");
    } else {
      tagBtn.classList.remove("selected");
    }
    
    // 최대 5개 선택 시 나머지 비활성화
    if (selectedEmotionTags.length >= 5 && !selectedEmotionTags.includes(tag)) {
      tagBtn.classList.add("disabled");
    } else {
      tagBtn.classList.remove("disabled");
    }
  });
}

// 감정 태그 개수 업데이트
function updateEmotionTagsCount() {
  const countDisplay = document.getElementById("emotion-tags-count");
  if (countDisplay) {
    countDisplay.textContent = `${selectedEmotionTags.length} / 5개 선택됨`;
    if (selectedEmotionTags.length >= 5) {
      countDisplay.style.color = "#ef4444";
    } else {
      countDisplay.style.color = "#667295";
    }
  }
}

// 감정 태그 UI 초기화
initEmotionTags();

// 추출된 텍스트 접기/펼치기 함수
function toggleStudyText() {
  const container = document.getElementById("study-text-container");
  const arrow = document.getElementById("study-text-arrow");
  
  if (container.style.display === "none") {
    container.style.display = "block";
    arrow.textContent = "▼";
    arrow.style.transform = "rotate(0deg)";
  } else {
    container.style.display = "none";
    arrow.textContent = "▶";
    arrow.style.transform = "rotate(0deg)";
  }
}

// 전역 함수로 등록 (HTML에서 호출 가능하도록)
window.toggleStudyText = toggleStudyText;

generateBtn.addEventListener("click", handleGenerate);
cancelBtn.addEventListener("click", handleCancel);
generateMelodyBtn.addEventListener("click", handleGenerateMelody);
