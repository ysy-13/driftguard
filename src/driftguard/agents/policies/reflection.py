from .base import RecoveryDecision, RecoveryPolicy


class ReflectionPolicy(RecoveryPolicy):
    method = "reflection"
    prompt_name = "reflection_v1.txt"

    def after_failure(self, call, response):
        return RecoveryDecision(request_llm=True, context={"reflection": {"call": call, "response": response}})

