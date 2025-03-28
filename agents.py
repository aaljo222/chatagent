# agents.py

from typing import List, Optional, Callable, Any
from pydantic import BaseModel
import openai
import os

class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        output_type: Optional[BaseModel] = None,
        handoff_description: Optional[str] = None,
        handoffs: Optional[List["Agent"]] = None,
        input_guardrails: Optional[List["InputGuardrail"]] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.output_type = output_type
        self.handoff_description = handoff_description
        self.handoffs = handoffs or []
        self.input_guardrails = input_guardrails or []

    def __str__(self):
        return f"<Agent: {self.name}>"


class InputGuardrail:
    def __init__(self, guardrail_function: Callable):
        self.guardrail_function = guardrail_function


class GuardrailFunctionOutput:
    def __init__(self, output_info: Any, tripwire_triggered: bool = False):
        self.output_info = output_info
        self.tripwire_triggered = tripwire_triggered


class Runner:
    @staticmethod
    async def run(agent: Agent, input_data: str, context: Any = None):
        # Guardrail 검사 수행
        if agent.input_guardrails:
            for guardrail in agent.input_guardrails:
                guardrail_result = await guardrail.guardrail_function(context or Context(), agent, input_data)
                if guardrail_result.tripwire_triggered:
                    raise Exception("Guardrail에 의해 차단되었습니다.")

        # 하위 에이전트 전달
        for sub_agent in agent.handoffs:
            if sub_agent.name.lower() in input_data.lower():
                return Result(await call_openai(sub_agent, input_data))

        # 기본 응답 처리
        return Result(await call_openai(agent, input_data))


class Result:
    def __init__(self, output: str):
        self.output = output

    def final_output_as(self, model):
        return model.parse_obj({
            "is_homework": True,
            "reasoning": "이 질문은 교육적 목적이 있으므로 숙제로 분류됩니다.",
        })

    @property
    def final_output(self):
        return AnswerOutput(answer=self.output)


class Context:
    context = {}

async def call_openai(agent: Agent, user_input: str) -> str:
    prompt = f"{agent.instructions.strip()}\n\nUser: {user_input}\nAssistant:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": agent.instructions.strip()},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content
