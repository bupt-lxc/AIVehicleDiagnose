import unittest

from app.diagnostic.graph import AutoDiagnoseGraph


class _NoopAgent:
    pass


class _NoopAuditLogger:
    def start_session(self, thread_id):
        pass

    def log_event(self, *args, **kwargs):
        pass

    def log_node_execution(self, *args, **kwargs):
        pass


class _Snapshot:
    def __init__(self, values):
        self.values = values


class _FakeCompiledGraph:
    def __init__(self):
        self.state = {
            "thread_id": "thread-auto",
            "current_node": "hitl_report_approval",
            "hitl_required": True,
            "current_hitl_point": "report_approval",
            "hitl_question": "approve?",
            "hitl_options": ["A", "B", "C"],
            "hitl_feedbacks": [],
        }
        self.update_payloads = []

    async def astream(self, input_state, run_config, stream_mode="updates"):
        if input_state is not None:
            yield {"hitl_report_approval": self.state}
        else:
            self.state = {
                **self.state,
                "current_node": "report_generator",
                "final_report": {"root_cause": "ok"},
            }
            yield {"report_generator": self.state}

    async def aget_state(self, run_config):
        return _Snapshot(self.state)

    async def aupdate_state(self, run_config, values):
        self.update_payloads.append(dict(values))
        self.state = {**self.state, **values}


class AutoApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_interactive_report_approval_auto_approves_and_resumes(self):
        graph = AutoDiagnoseGraph(
            parser_agent=_NoopAgent(),
            surveyor_agent=_NoopAgent(),
            mcp_librarian=_NoopAgent(),
            rca_engine=_NoopAgent(),
            auditor_agent=_NoopAgent(),
            report_generator=_NoopAgent(),
        )
        fake_graph = _FakeCompiledGraph()
        graph.graph = fake_graph
        graph.audit_logger = _NoopAuditLogger()

        final_state = await graph.run(fake_graph.state, interactive=False)

        self.assertEqual(final_state["current_node"], "report_generator")
        self.assertTrue(final_state["report_approved"])
        self.assertEqual(final_state["hitl_feedbacks"][-1]["user_choice"], "A")
        self.assertEqual(final_state["final_report"]["root_cause"], "ok")


if __name__ == "__main__":
    unittest.main()
