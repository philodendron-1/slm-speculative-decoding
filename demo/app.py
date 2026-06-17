import chainlit as cl
import whisper
import httpx
import tempfile
import os
import asyncio

# Whisper 모델 로드 (base 모델, 빠름)
print("Whisper 로드 중...", flush=True)
whisper_model = whisper.load_model("base")

VLLM_URL = "http://moana-y2:8000/v1/completions"

SYSTEM_PROMPT = """You are an English tutor evaluating a Korean speaker's spoken interpretation of a Korean sentence into English.

Classify the error using ONE of the following categories ONLY:
- word choice
- tense
- word order
- preposition
- naturalness
- grammar
- subject-verb agreement
- verb form
- article
- redundancy

Respond in this exact format:
ERROR_TYPE: <category>
CORRECTION: <corrected sentence>
SHORT_REASON: <brief reason>

"""

async def get_feedback(instruction: str) -> str:
    prompt = SYSTEM_PROMPT + f"### Instruction:\n{instruction}\n\n### Response:\nERROR_TYPE:"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            VLLM_URL,
            json={
                "model": "/data/sonnet18s/models/qwen2.5-7b",
                "prompt": prompt,
                "max_tokens": 80,
                "temperature": 0,
            }
        )
        result = response.json()
        return "ERROR_TYPE:" + result["choices"][0]["text"]

@cl.on_chat_start
async def start():
    await cl.Message(
        content="안녕하세요! 🎤 한국어 원문을 듣고 영어로 통역한 음성을 업로드해주세요.\n\n"
                "**사용 방법:**\n"
                "1. 한국어 원문을 입력하세요\n"
                "2. 영어 통역 음성 파일을 업로드하세요\n"
                "3. AI가 피드백을 드립니다!"
    ).send()
    cl.user_session.set("korean_original", None)

@cl.on_message
async def main(message: cl.Message):
    korean_original = cl.user_session.get("korean_original")

    # 음성 파일이 첨부된 경우
    if message.elements:
        for elem in message.elements:
            if hasattr(elem, 'path') and elem.path:
                await cl.Message(content="🎤 음성 인식 중...").send()

                # Whisper STT
                result = whisper_model.transcribe(elem.path, language="en")
                english_attempt = result["text"].strip()

                await cl.Message(
                    content=f"📝 **인식된 영어 통역:**\n{english_attempt}"
                ).send()

                if not korean_original:
                    await cl.Message(
                        content="⚠️ 한국어 원문을 먼저 텍스트로 입력해주세요!"
                    ).send()
                    return

                # 피드백 생성
                await cl.Message(content="🤖 피드백 생성 중...").send()

                instruction = (
                    f"Korean original: {korean_original}\n"
                    f"Learner's English attempt: {english_attempt}"
                )

                try:
                    feedback = await get_feedback(instruction)
                    lines = feedback.strip().split("\n")
                    formatted = ""
                    for line in lines:
                        if line.startswith("ERROR_TYPE:"):
                            formatted += f"❌ **오류 유형:** {line.replace('ERROR_TYPE:', '').strip()}\n\n"
                        elif line.startswith("CORRECTION:"):
                            formatted += f"✅ **수정 제안:** {line.replace('CORRECTION:', '').strip()}\n\n"
                        elif line.startswith("SHORT_REASON:"):
                            formatted += f"💡 **설명:** {line.replace('SHORT_REASON:', '').strip()}\n\n"

                    await cl.Message(content=f"## 피드백\n\n{formatted}").send()
                    cl.user_session.set("korean_original", None)

                except Exception as e:
                    await cl.Message(content=f"❌ 오류 발생: {str(e)}").send()

    # 텍스트 입력 (한국어 원문)
    else:
        cl.user_session.set("korean_original", message.content)
        await cl.Message(
            content=f"✅ **한국어 원문 저장:**\n{message.content}\n\n"
                    "이제 영어 통역 음성 파일(.wav/.mp3/.m4a)을 업로드해주세요!"
        ).send()
