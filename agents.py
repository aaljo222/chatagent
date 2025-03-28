# agents.py

from typing import List, Optional, Callable, Any
from pydantic import BaseModel

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
        """
        간단한 실행 시뮬레이션. 실제 OpenAI API와 연결되거나 다른 로직으로 대체해야 함.
        """
        class Result:
            def __init__(self, output):
                self.output = output

            def final_output_as(self, model):
                return model.parse_obj({
                    "is_homework": True,
                    "reasoning": "질문에 교육적 목적이 담겨 있으므로 숙제로 판단됩니다."
                })

            @property
            def final_output(self):
                return self.output

        return Result(output=f"에이전트 {agent.name}가 다음 질문에 응답함: {input_data}")
