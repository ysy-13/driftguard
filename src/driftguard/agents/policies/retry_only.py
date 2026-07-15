from .base import RecoveryDecision, RecoveryPolicy


class RetryOnlyPolicy(RecoveryPolicy):
    method = "retry_only"

    def __init__(self):
        self._retried = False

    def after_failure(self, call, response):
        if call.get("local_validation", {}).get("valid") is False:
            return RecoveryDecision(request_llm=True)
        if response.get("payload", {}).get("ok") is False and not self._retried:
            self._retried = True
            return RecoveryDecision(exact_retry=True, request_llm=False)
        return RecoveryDecision(request_llm=True)

    def end_task(self):
        self._retried = False
