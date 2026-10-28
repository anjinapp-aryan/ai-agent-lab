"""Phase 2 Batch 3 regression tests: authorization boundary and tool-output security.

Scripted model + simulated in-memory store only (no network, no real effects). Each assertion pins an OBSERVED behavior
recorded in phases/phase-02/experiments.md. One small live test asserts the security INVARIANT (not model behavior).
"""
import json

import pytest

import exp_batch3 as e3
from lab2 import security as sec

FWS = e3.FWS


def _run(fn, key):
    e3.RESULTS.clear()
    fn()
    return list(e3.RESULTS[key])


def find(rows, fw, part):
    hits = [r for r in rows if r["framework"] == fw and part in r["case"]]
    assert len(hits) == 1, f"{fw} / {part!r} matched {len(hits)}"
    return hits[0]


# ------------------------------------------------------------ 3.1 basic matrix (Invariant 1)
@pytest.fixture(scope="module")
def basic_rows():
    return _run(e3.basic, "3.1")


@pytest.mark.parametrize("fw", FWS)
def test_basic_matrix_deny_means_no_side_effect_allow_means_effect(basic_rows, fw):
    for r in (x for x in basic_rows if x["framework"] == fw):
        if r["expect"] == "DENY":
            assert r["side_effects"] == 0 and r["authz_denies"] == 1, r["case"]
        else:
            assert r["authz_denies"] == 0 and r["run_continued"], r["case"]
            assert r["side_effects"] == (0 if "get_order" in r["case"] else 1)


def test_denial_reaches_model_in_langgraph_but_ends_the_run_in_pydantic_ai(basic_rows):
    lg = find(basic_rows, "langgraph", "VIEWER -> delete_order")
    pa = find(basic_rows, "pydantic-ai", "VIEWER -> delete_order")
    assert lg["run_continued"] and "PermissionDenied" in lg["model_saw"]
    assert not pa["run_continued"] and "PermissionDenied" in pa["run_error"]


# ------------------------------------------------------------ 3.2 where authorization lives
@pytest.fixture(scope="module")
def design_rows():
    e3.RESULTS.clear()
    e3.designs()
    return [r for k, v in e3.RESULTS.items() for r in v]


@pytest.mark.parametrize("fw", FWS)
def test_prompt_only_is_not_a_boundary_unguarded_tool_executes(design_rows, fw):
    r = find(design_rows, fw, "model-only")
    assert r["side_effects"] == 1 and r["authz_checks"] == 0


@pytest.mark.parametrize("fw", FWS)
def test_tool_level_check_blocks_but_tool_body_has_started(design_rows, fw):
    r = find(design_rows, fw, "tool-level check")
    assert r["side_effects"] == 0 and r["tool_bodies_started"] == 1


@pytest.mark.parametrize("fw,part", [("langgraph", "wrap_tool_call) raise"), ("langgraph", "returns error ToolMessage"),
                                     ("pydantic-ai", "before_tool_execute) raise"), ("pydantic-ai", "before_tool_execute) ModelRetry")])
def test_native_hook_blocks_before_the_tool_body_starts(design_rows, fw, part):
    r = find(design_rows, fw, part)
    assert r["tool_bodies_started"] == 0 and r["side_effects"] == 0 and r["authz_denies"] == 1


def test_hiding_a_tool_is_visibility_not_enforcement(design_rows):
    pa = find(design_rows, "pydantic-ai", "prepare_tools hides")
    lg = find(design_rows, "langgraph", "binds only permitted tools")
    assert pa["side_effects"] == lg["side_effects"] == 0 and pa["authz_checks"] == lg["authz_checks"] == 0
    assert "Unknown tool" in pa["model_saw"] and "not a valid tool" in lg["model_saw"]


def test_requires_approval_with_default_output_type_errors_without_executing(design_rows):
    r = find(design_rows, "pydantic-ai", "requires_approval")
    assert r["side_effects"] == 0 and r["tool_bodies_started"] == 0 and "UserError" in r["run_error"]


# ------------------------------------------------------------ 3.3 / 3.4 resource + tenant (Invariant 2)
@pytest.fixture(scope="module")
def res_rows():
    rows = _run(e3.resource, "3.3/3.4")
    return rows


@pytest.mark.parametrize("fw", FWS)
def test_permission_only_allows_cross_tenant_write_and_read(res_rows, fw):
    w = find([r for r in res_rows if r["level"] == "perm"], fw, "tenant-A cancel_order order-200")
    rd = find([r for r in res_rows if r["level"] == "perm"], fw, "tenant-A get_order order-200")
    assert w["side_effects"] == 1 and "B-confidential" in rd["model_saw"]


@pytest.mark.parametrize("fw", FWS)
def test_resource_check_blocks_cross_tenant_write_and_read(res_rows, fw):
    rr = [r for r in res_rows if r["level"] == "perm+resource"]
    assert find(rr, fw, "tenant-A cancel_order order-100")["side_effects"] == 1
    for part in ("tenant-A cancel_order order-200", "tenant-B cancel_order order-100"):
        assert find(rr, fw, part)["side_effects"] == 0 and find(rr, fw, part)["authz_denies"] == 1
    for part in ("tenant-A get_order order-200", "tenant-B get_order order-100"):
        r = find(rr, fw, part)
        assert r["authz_denies"] == 1 and "A-secret" not in r["model_saw"] and "B-confidential" not in r["model_saw"]
    assert find(rr, fw, "tenant-B get_order order-200")["authz_denies"] == 0


# ------------------------------------------------------------ 3.6 failure handling
@pytest.fixture(scope="module")
def fail_rows():
    return _run(e3.failures, "3.6")


def test_denial_failure_semantics(fail_rows):
    lg = find(fail_rows, "langgraph", "LG default")
    assert lg["authz_checks"] == 3 and lg["side_effects"] == 0 and lg["run_continued"]
    rp = find(fail_rows, "langgraph", "RetryPolicy")
    assert rp["authz_checks"] == 3 and not rp["run_continued"] and rp["side_effects"] == 0     # framework retried a deterministic denial
    assert find(fail_rows, "pydantic-ai", "plain PermissionDenied")["authz_checks"] == 1
    mr = find(fail_rows, "pydantic-ai", "ModelRetry")
    assert mr["authz_checks"] == 2 and "UnexpectedModelBehavior" in mr["run_error"] and mr["side_effects"] == 0


@pytest.mark.parametrize("fw", FWS)
def test_returned_denial_keeps_run_alive_and_effect_zero(fail_rows, fw):
    r = find(fail_rows, fw, "returned as a result")
    assert r["run_continued"] and r["side_effects"] == 0 and "DENIED" in r["model_saw"]


# ------------------------------------------------------------ 3.7 bypass (Invariant 3)
@pytest.fixture(scope="module")
def bypass_rows():
    return _run(e3.bypass, "3.7")


@pytest.mark.parametrize("fw", FWS)
def test_identity_from_model_arguments_is_a_bypass(bypass_rows, fw):
    r = find(bypass_rows, fw, "UNTRUSTED")
    assert r["side_effects"] == 1 and r["authz_checks"] == 0


@pytest.mark.parametrize("fw", FWS)
def test_trusted_context_identity_ignores_model_supplied_identity(bypass_rows, fw):
    for part in ("TRUSTED: closure identity", "VIEWER forges role=ADMIN"):
        assert find(bypass_rows, fw, part)["side_effects"] == 0


def test_forged_identity_args_langgraph_drops_pydantic_ai_rejects(bypass_rows):
    lg = find(bypass_rows, "langgraph", "VIEWER forges role=ADMIN")
    pa = find(bypass_rows, "pydantic-ai", "VIEWER forges role=ADMIN")
    assert lg["tool_bodies_started"] == 1 and lg["authz_denies"] == 1          # extras dropped, trusted identity checked, denied
    assert pa["tool_bodies_started"] == 0 and "extra_forbidden" in pa["model_saw"]  # rejected before the tool


# ------------------------------------------------------------ 3.8 / 3.11 / 3.12 delivery (Invariant 4)
@pytest.fixture(scope="module")
def delivery_rows():
    return _run(e3.delivery, "3.8-delivery")


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("key", ["A_plain", "B_instruction", "C_fake_system", "D_str"])
def test_string_output_reaches_model_verbatim_as_tool_message(delivery_rows, fw, key):
    r = [x for x in delivery_rows if x["framework"] == fw and x["payload"] == key][0]
    assert r["role_on_wire"] == "tool" and r["content_type"] == "str"
    assert r["content_exact"] == sec.INJECTIONS[key]            # byte-for-byte: no sanitizing, delimiting or marking


@pytest.mark.parametrize("fw", FWS)
def test_structured_output_is_serialized_to_json_text_without_trust_marking(delivery_rows, fw):
    r = [x for x in delivery_rows if x["framework"] == fw and x["payload"] == "D_structured"][0]
    assert r["payload_type"] == "dict" and json.loads(r["content_on_wire"]) == sec.INJECTIONS["D_structured"]


# ------------------------------------------------------------ 3.9 / 3.10 / 3.13 flows (Invariants 5, 6)
@pytest.fixture(scope="module")
def flow_rows():
    return _run(e3.flows, "3.13")


@pytest.mark.parametrize("fw", FWS)
def test_unprotected_injected_second_call_executes(flow_rows, fw):
    r = find(flow_rows, fw, "UNPROTECTED")
    assert r["side_effects"] == 1 and r["authz_checks"] == 0


@pytest.mark.parametrize("fw", FWS)
def test_protected_injected_second_call_is_denied_and_every_call_is_checked(flow_rows, fw):
    r = find(flow_rows, fw, "tool-level authz (VIEWER)")
    assert r["side_effects"] == 0 and r["authz_denies"] == 1 and r["authz_checks"] == 2 and r["every_call_crossed_boundary"]


@pytest.mark.parametrize("fw,part", [("langgraph", "LG native hook"), ("pydantic-ai", "PA native hook")])
def test_native_hook_denies_injected_call_before_tool_body(flow_rows, fw, part):
    r = find(flow_rows, fw, part)
    assert r["side_effects"] == 0 and r["authz_checks"] == 2 and r["tool_bodies_started"] == 1 and r["every_call_crossed_boundary"]


@pytest.mark.parametrize("fw", FWS)
def test_authorization_bounds_blast_radius_but_does_not_establish_intent(flow_rows, fw):
    assert find(flow_rows, fw, "OPERATOR (authorized for refund)")["side_effects"] == 1       # authorized -> injection still succeeds
    cross = find(flow_rows, fw, "injection targets tenant-B")
    assert cross["side_effects"] == 0 and cross["authz_denies"] == 1                           # resource check stops the cross-tenant one


# ------------------------------------------------------------ live: invariant, not model behavior
from lab_common.config import model_configured  # noqa: E402

live = pytest.mark.skipif(not model_configured(), reason="no model provider configured")


@live
@pytest.mark.live
@pytest.mark.parametrize("fw", FWS)
def test_live_injection_never_produces_a_side_effect_when_authorization_is_enforced(fw):
    import lg_agent
    import pa_agent
    run = lg_agent.run if fw == "langgraph" else pa_agent.run
    lg_agent.SYSTEM_INSTRUCTIONS = pa_agent.SYSTEM_INSTRUCTIONS = e3.LIVE_PROMPT
    store = sec.OrderStore(get_payload=sec.INJECTIONS["B_instruction"])
    t = sec.build_tools(store, e3.VIEWER, "perm+resource")
    sec.AUTHZ.clear()
    run("What is the status of order-100?", tools=[t["get_order"], t["refund_order"]])
    assert not [e for e in store.effects if e[0] == "refund"]         # invariant holds whatever the model chose to do
    requested = any(n == "refund_order" for n, _ in store.invocations)
    assert (not requested) or any(a["decision"] == "DENY" for a in sec.AUTHZ)
