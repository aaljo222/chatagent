import os
import openai
import asyncio
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
import nest_asyncio
nest_asyncio.apply()

# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 모델 출력 정의
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# 에이전트 정의
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="""
        Check if the user is asking about homework, academic subjects, or general knowledge questions.
        Be generous — treat simple educational or philosophical questions as homework.
    """,
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples.",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

# Streamlit 앱 UI
st.set_page_config(page_title="AI Tutor with Guardrail", page_icon="📘")
st.title("📘 AI Tutor with Guardrail")

user_input = st.text_input("질문을 입력하세요:", "")
bypass_guardrail = st.checkbox("✅ 숙제 검사를 건너뛰고 질문 보내기")

# Guardrail 함수
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)

    # 숙제 판단 결과 표시
    st.markdown(f"### 🧠 숙제 판단 결과")
    st.markdown(f"- **숙제인가요?** → `{final_output.is_homework}`")
    st.markdown(f"- **이유:** {final_output.reasoning}")

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework and not bypass_guardrail,
    )

# Triage Agent
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)],
)

# 질문 처리
if st.button("질문하기") and user_input.strip():
    with st.spinner("답변 생성 중..."):
        async def process():
            try:
                result = await Runner.run(triage_agent, user_input)
                return result.final_output or "❌ AI가 응답하지 않았습니다."
            except Exception as e:
                return f"❌ Guardrail에 의해 차단되었습니다.\n\n**에러:** `{e}`"

        # Streamlit에서 async 지원 시 await 사용
        answer = asyncio.run(process())  # <-- 이 줄을 다음으로 바꿔보세요
        # answer = await process()      # (Streamlit 내부에서 async 가능 시)
        st.markdown("### 📘 AI Tutor의 답변")
        st.write(str(answer))