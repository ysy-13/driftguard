from __future__ import annotations

from .issue import assign_issue, close_issue, create_issue, get_issue
from .membership import add_member, get_member, update_member_role
from .pipeline import get_pipeline_status, retry_pipeline, trigger_pipeline
from .repository import get_repository, update_repository


HANDLERS = {
    "get_repository": get_repository,
    "update_repository": update_repository,
    "create_issue": create_issue,
    "get_issue": get_issue,
    "assign_issue": assign_issue,
    "close_issue": close_issue,
    "trigger_pipeline": trigger_pipeline,
    "get_pipeline_status": get_pipeline_status,
    "retry_pipeline": retry_pipeline,
    "add_member": add_member,
    "get_member": get_member,
    "update_member_role": update_member_role,
}

__all__ = ["HANDLERS"]
