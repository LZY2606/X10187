"""Characterization tests for the per-run internal codec sessions.

A "session" is the mutable state owned by exactly one top-level
``Pickler.flatten`` / ``Unpickler.restore`` invocation (or the module level
``encode`` / ``decode`` wrappers): object identity tables, the depth / name
stacks, the pending-proxy list and per-run class registrations.

The session is an internal implementation detail; these tests pin down its
lifecycle boundaries instead of exposing it to users.
"""

import json

import pytest

import jsonpickle
from jsonpickle import handlers, tags
from jsonpickle.pickler import Pickler
from jsonpickle.unpickler import Unpickler


class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None
        self.ref2 = None


class EqualNode:
    """Nodes that compare equal but are separate objects."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, EqualNode) and other.value == self.value

    def __hash__(self):
        return hash(self.value)


class RestoreBoom:
    """Module-level so that its class path is importable on restore."""

    def __init__(self, name="x", error="boom"):
        self.name = name
        self.error = error


class WrapNode:
    """Module-level wrapper class used by the recursive-handler tests."""

    def __init__(self, name="wrap", inner=None):
        self.name = name
        self.inner = inner


class ExplodingRestoreHandler(handlers.BaseHandler):
    def flatten(self, obj, data):
        data["py/object"] = "session_test.RestoreBoom"
        data["boom_kind"] = obj.error
        return data

    def restore(self, data):
        raise ValueError(data.get("boom_kind", "boom"))


class RecursiveWrapHandler(handlers.BaseHandler):
    def flatten(self, obj, data):
        data["py/object"] = "session_test.WrapNode"
        data["name"] = self.context.flatten(obj.name, reset=False)
        data["inner"] = self.context.flatten(obj.inner, reset=False)
        return data

    def restore(self, data):
        wrap = WrapNode(data["name"])
        wrap.inner = self.context.restore(data["inner"], reset=False)
        return wrap


# ---------------------------------------------------------------------------
# helpers for inspecting the (internal) session state
# ---------------------------------------------------------------------------


def pickler_state_is_empty(pickler):
    """The long-lived pickler holds no per-run state outside of a run."""
    # the identity/seen/cache tables and depth are all per-run state
    assert pickler._objs == {}
    assert pickler._seen == []
    assert pickler._flattened == {}
    assert pickler._depth == -1


def unpickler_state_is_empty(unpickler):
    """The long-lived unpickler holds no per-run state outside of a run."""
    assert unpickler._objs == []
    assert unpickler._obj_to_idx == {}
    assert unpickler._namedict == {}
    assert unpickler._namestack == []
    assert unpickler._proxies == []
    assert unpickler._classes == {}


def assert_no_session(codec):
    """Nothing per-run is reachable from the long-lived codec."""
    assert codec._session is None


# ---------------------------------------------------------------------------
# Pickler session lifecycle
# ---------------------------------------------------------------------------


def test_pickler_cleans_up_session_after_top_level_flatten():
    pickler = Pickler()
    shared = Node("shared")
    root = Node("root")
    root.ref = shared
    root.ref2 = shared
    pickler.flatten(root)
    pickler_state_is_empty(pickler)
    assert_no_session(pickler)


def test_pickler_cleans_up_session_after_exception():
    pickler = Pickler()

    class Explodes:
        def __getstate__(self):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        pickler.flatten(Explodes())

    # direct verification that no run-local state survived the failure,
    # rather than just observing that a later call happens to work
    pickler_state_is_empty(pickler)
    assert_no_session(pickler)


def test_pickler_explicit_reset_clears_state():
    pickler = Pickler()
    # reset() on a fresh codec keeps its documented "empty tables" effect
    pickler.reset()
    pickler_state_is_empty(pickler)

    root = Node("x")
    pickler.flatten(root)
    pickler.reset()
    pickler_state_is_empty(pickler)

    # and the codec can be reused normally afterwards
    again = pickler.flatten(root)
    assert again["name"] == "x"


def test_handler_recursive_flatten_joins_current_session():
    """A flatten() inside a handler must share the active session.

    The handler observes the same identity table as the outer run and can
    reference an object the outer run has already logged; starting a new
    session would lose reference identity.
    """
    pickler = Pickler()
    seen_sessions = []
    seen_id_tables = []

    class RecursiveNode(Node):
        pass

    class RecursiveHandler(handlers.BaseHandler):
        def flatten(self, obj, data):
            context = self.context
            seen_sessions.append(context._session)
            data["py/object"] = f"{__name__}.RecursiveNode"
            # recursion joins the current session instead of starting a new
            # identity table
            data["name"] = context.flatten(obj.name, reset=False)
            data["ref"] = context.flatten(obj.ref, reset=False)
            seen_id_tables.append(dict(context._session.objs))
            return data

        def restore(self, data):
            node = RecursiveNode(data["name"])
            node.ref = self.context.restore(data["ref"], reset=False)
            return node

    handlers.register(RecursiveNode, RecursiveHandler, base=True)
    try:
        child = RecursiveNode("child")
        root = RecursiveNode("root")
        root.ref = child
        result = pickler.flatten(root)
    finally:
        handlers.unregister(RecursiveNode)

    assert_no_session(pickler)
    assert len(seen_sessions) == 2
    # both handler invocations saw the very same session object
    assert seen_sessions[0] is seen_sessions[1]
    # identity accumulated across recursion: root (0) and child (1) were both
    # logged in the one shared session while the child handler ran
    assert len(seen_id_tables[1]) >= 2
    assert result["name"] == "root"
    assert result["ref"]["name"] == "child"


def test_make_refs_shared_subobject_must_be_identical():
    shared = Node("shared")
    root = Node("root")
    root.ref = shared
    root.ref2 = shared

    encoded = jsonpickle.encode(root)
    decoded = jsonpickle.decode(encoded)
    # equal AND identical
    assert decoded.ref is decoded.ref2
    assert decoded.ref == decoded.ref2


def test_make_refs_false_equal_objects_are_distinct():
    shared = EqualNode("shared")
    root = Node("root")
    root.ref = shared
    root.ref2 = shared

    encoded = jsonpickle.encode(root, make_refs=False)
    decoded = jsonpickle.decode(encoded)
    # the two copies are equal but not identical
    assert decoded.ref is not decoded.ref2
    assert decoded.ref == decoded.ref2


def test_make_refs_false_cycle_is_broken_and_state_cleans_up():
    pickler = Pickler(make_refs=False, unpicklable=False)
    root = Node("root")
    root.ref = root
    result = pickler.flatten({"root": root})
    # with refs disabled the cycle is broken by repr() instead of looping
    assert result == {
        "root": {"name": "root", "ref": repr(root), "ref2": None}
    }
    pickler_state_is_empty(pickler)
    assert_no_session(pickler)


def test_cycle_and_shared_subobject_roundtrip_identity():
    root = Node("root")
    child = Node("child")
    root.ref = child
    child.ref = root
    root.ref2 = child

    encoded = jsonpickle.encode(root)
    decoded = jsonpickle.decode(encoded)
    assert decoded is decoded.ref.ref
    assert decoded.ref is decoded.ref2
    assert decoded.ref.ref is decoded


def test_failure_then_same_instance_handles_cycle_graph():
    """Regression: a failed run must not poison a reused pickler."""
    pickler = Pickler()

    class Explodes:
        def __getstate__(self):
            raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError):
        pickler.flatten(Explodes())
    pickler_state_is_empty(pickler)
    assert_no_session(pickler)

    root = Node("root")
    root.ref = root
    result = pickler.flatten(root)
    # the cycle resolves to a reference rather than re-raising / looping
    assert result["ref"] == {"py/id": 0}
    pickler_state_is_empty(pickler)
    assert_no_session(pickler)


# ---------------------------------------------------------------------------
# Unpickler session lifecycle
# ---------------------------------------------------------------------------


def test_unpickler_cleans_up_session_after_top_level_restore():
    unpickler = Unpickler()
    encoded = jsonpickle.encode(Node("root"))
    unpickler.restore(json.loads(encoded))
    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)


def test_unpickler_cleans_up_session_after_handler_exception():
    unpickler = Unpickler()
    handlers.register(RestoreBoom, ExplodingRestoreHandler)
    try:
        encoded = jsonpickle.encode(RestoreBoom())
        with pytest.raises(ValueError, match="boom"):
            jsonpickle.decode(encoded, context=unpickler)
    finally:
        handlers.unregister(RestoreBoom)

    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)


def test_unpickler_cleans_up_after_backend_parse_failure():
    unpickler = Unpickler()
    # backends (simplejson/ujson/json) all raise a ValueError subclass
    with pytest.raises(ValueError):
        jsonpickle.decode("{not valid json", context=unpickler)
    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)


def test_unpickler_cleans_up_after_illegal_reference():
    unpickler = Unpickler()
    payload = json.dumps({tags.ID: 99})
    # a dangling forward id becomes an _IDProxy during restore; the proxy and
    # its backing list are session state and must not leak afterwards
    result = unpickler.restore(json.loads(payload))
    assert result.get() is None
    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)


def test_unpickler_explicit_reset_clears_state():
    unpickler = Unpickler()
    unpickler.reset()
    unpickler_state_is_empty(unpickler)

    unpickler.restore(json.loads(jsonpickle.encode(Node("x"))))
    unpickler.reset()
    unpickler_state_is_empty(unpickler)

    # still reusable
    result = unpickler.restore(json.loads(jsonpickle.encode(Node("y"))))
    assert result.name == "y"


def test_forward_reference_proxy_resolution_cleans_up():
    root = Node("root")
    child = Node("child")
    root.ref = child
    child.ref = root  # forward reference to the not-yet-built parent

    encoded = jsonpickle.encode(root)
    decoded = jsonpickle.decode(encoded)
    assert decoded.ref.ref is decoded

    unpickler = Unpickler()
    unpickler.restore(json.loads(encoded))
    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)


def test_handler_recursive_restore_joins_current_session():
    unpickler = Unpickler()
    seen = []

    class SessionCapturingHandler(RecursiveWrapHandler):
        def restore(self, data):
            seen.append(self.context._session)
            return super().restore(data)

    handlers.register(WrapNode, SessionCapturingHandler)
    try:
        wrap = WrapNode("wrap", Node("inner"))
        encoded = jsonpickle.encode(wrap)
        result = unpickler.restore(json.loads(encoded))
    finally:
        handlers.unregister(WrapNode)

    assert isinstance(result, WrapNode)
    assert result.inner.name == "inner"
    assert len(seen) == 1 and seen[0] is not None
    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)


def test_failure_then_same_unpickler_handles_cycle_graph():
    """Regression: a failed decode must not poison a reused unpickler."""
    unpickler = Unpickler()

    class RuntimeBoomHandler(ExplodingRestoreHandler):
        def restore(self, data):
            raise RuntimeError("restore kaboom")

    handlers.register(RestoreBoom, RuntimeBoomHandler)
    try:
        encoded = jsonpickle.encode(RestoreBoom(error="kaboom"))
        with pytest.raises(RuntimeError):
            jsonpickle.decode(encoded, context=unpickler)

        root = Node("root")
        root.ref = root
        encoded_cycle = jsonpickle.encode(root)
        decoded = jsonpickle.decode(encoded_cycle, context=unpickler)
        assert decoded is decoded.ref
    finally:
        handlers.unregister(RestoreBoom)

    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)


def test_independent_runs_on_reused_codecs_keep_separate_identity():
    """Two runs with the same codecs preserve identity independently."""
    pickler = Pickler()
    unpickler = Unpickler()

    def roundtrip(obj):
        encoded = jsonpickle.backend.json.encode(pickler.flatten(obj))
        return unpickler.restore(json.loads(encoded))

    a = Node("a")
    a.ref = a
    decoded_a = roundtrip(a)
    assert decoded_a is decoded_a.ref

    b = Node("b")
    b.ref = b
    decoded_b = roundtrip(b)
    assert decoded_b is decoded_b.ref
    assert decoded_a is not decoded_b
    pickler_state_is_empty(pickler)
    unpickler_state_is_empty(unpickler)
    assert_no_session(pickler)
    assert_no_session(unpickler)


def test_readonly_configuration_survives_sessions():
    """Backends/options are long-lived config, not per-run state."""
    pickler = Pickler()
    unpickler = Unpickler()
    backend = pickler.backend
    keys = pickler.keys
    safe = unpickler.safe
    make_refs = pickler.make_refs
    pickler.flatten(Node("x"))
    unpickler.restore(json.loads(jsonpickle.encode(Node("y"))))
    assert pickler.backend is backend
    assert pickler.keys is keys
    assert pickler.make_refs is make_refs
    assert unpickler.safe is safe
    assert unpickler.backend is pickler.backend


def test_incremental_reset_false_restore_accumulates_in_one_session():
    """reset=False incremental restores keep accumulating until reset=True.

    Mirrors the object_hook-based decoding used by some consumers
    (see tests/bson_test.py): several reset=False restores share one
    session, a later reset=True closes it.
    """
    unpickler = Unpickler()
    # tagged payloads establish identity-table state on every call
    first = json.loads(jsonpickle.encode(Node("first")))
    second = json.loads(jsonpickle.encode(Node("second")))

    unpickler.restore(first, reset=False)
    assert unpickler._session is not None
    session = unpickler._session
    n_objs = len(session.objs)
    assert n_objs >= 1

    unpickler.restore(second, reset=False)
    assert unpickler._session is session
    assert len(session.objs) > n_objs

    closing = json.loads(jsonpickle.encode(Node("closing")))
    result = unpickler.restore(closing, reset=True)
    assert result.name == "closing"
    unpickler_state_is_empty(unpickler)
    assert_no_session(unpickler)
