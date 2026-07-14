from .base import RecoveryDecision, RecoveryPolicy


class ValidationGuidedPolicy(RecoveryPolicy):
    method = "validation_guided"
    prompt_name = "validation_guided_v1.txt"

    def after_failure(self, call, response):
        return RecoveryDecision(request_llm=True, context={"local_validation": call.get("local_validation"), "runtime_response": response})

