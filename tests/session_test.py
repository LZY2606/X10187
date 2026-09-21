"""Characterization tests for the internal per-run session lifecycle.

A "session" is the mutable state used by a single top-level encode or
decode run: reference/object tables, recursion depth, the name stack and
the not-yet-resolved proxies.  These tests pin down the session
boundaries:

* a top-level run never leaves per-run state behind after an exception,
* recursive calls from inside handlers (``reset=False``) join the current
  session instead of starting a detached one,
* an explicit user ``reset()`` keeps its historical effect,
* a long-lived codec only carries read-only configuration between runs.
"""

import gc
import weakref

import pytest

import jsonpickle
from jsonpickle import handlers, tags, util
from jsonpickle.backend import json
from jsonpickle.pickler import Pickler
from jsonpickle.unpickler import Unpickler


class SharedThing:
    """A trivial object where equality and identity can diverge."""

    def __init__(self, name="thing"):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, SharedThing) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class Box:
    """An object with a custom handler that recurses into its child."""

    def __init__(self, child):
        self.child = child


class BoxHandler(handlers.BaseHandler):
    """Recurses via the *current* context, like numpy/pandas handlers do."""

    def flatten(self, obj, data):
        data["child"] = self.context.flatten(obj.child, reset=False)
        return data

    def restore(self, data):
        return Box(self.context.restore(data["child"], reset=False))


class Boom:
    """An object whose handler always fails, in both directions."""


class BoomHandler(handlers.BaseHandler):
    def flatten(self, obj, data):
        raise RuntimeError("boom flatten")

    def restore(self, data):
        raise RuntimeError("boom restore")


@pytest.fixture(autouse=True)
def registered_handlers():
    handlers.register(Box, BoxHandler)
    handlers.register(Boom, BoomHandler)
    yield
    handlers.unregister(Box)
    handlers.unregister(Boom)


def make_cycle():
    cyc = []
    cyc.append(cyc)
    return cyc


def assert_pickler_state_empty(pickler):
    """Directly assert that no per-run state is left on the pickler."""
    assert pickler._objs == {}
    assert pickler._depth == -1
    assert pickler._seen == []
    assert pickler._flattened == {}


def assert_unpickler_state_empty(unpickler):
    """Directly assert that no per-run state is left on the unpickler."""
    assert unpickler._namedict == {}
    assert unpickler._namestack == []
    assert unpickler._obj_to_idx == {}
    assert unpickler._objs == []
    assert unpickler._proxies == []


# ------------------------------------------------------------------
# Pickler session boundaries
# ------------------------------------------------------------------


def test_flatten_success_leaves_no_state_on_pickler():
    pickler = Pickler()
    shared = SharedThing()
    session_before = pickler._session
    pickler.flatten({"a": shared, "b": shared, "cycle": make_cycle()})
    assert_pickler_state_empty(pickler)
    # the run's session was discarded, not merely emptied
    assert pickler._session is not session_before


def test_flatten_exception_cleans_state_and_allows_reuse():
    """Regression: a failing run must not poison a long-lived pickler."""
    pickler = Pickler()
    with pytest.raises(RuntimeError, match="boom flatten"):
        pickler.flatten(Boom())
    # directly verify the state is empty, not just that a retry "works"
    assert_pickler_state_empty(pickler)

    # immediately reuse the same instance on a cyclic graph
    cyc = make_cycle()
    data = pickler.flatten(cyc)
    assert data == [{tags.ID: 0}]
    restored = jsonpickle.decode(json.encode(data))
    assert restored[0] is restored
    assert_pickler_state_empty(pickler)


def test_failed_encode_does_not_anchor_objects():
    """A failed top-level encode must not keep the encoded object alive."""
    pickler = Pickler()
    obj = Boom()
    ref = weakref.ref(obj)
    try:
        pickler.flatten(obj)
    except RuntimeError:
        pass
    del obj
    gc.collect()
    assert ref() is None


def test_handler_recursion_joins_current_pickler_session():
    """Objects flattened through a handler share reference identity with
    the rest of the graph instead of being encoded in a detached session."""
    shared = SharedThing()
    document = {"box": Box(shared), "shared": shared}
    pickler = Pickler()
    data = pickler.flatten(document)
    # the occurrence inside the handler output is the full object and the
    # later occurrence in the same graph is a reference to it, not a copy
    assert data["box"]["child"][tags.OBJECT] == util.importable_name(SharedThing)
    assert set(data["shared"]) == {tags.ID}
    restored = jsonpickle.decode(json.encode(data))
    assert restored["box"].child is restored["shared"]


def test_handler_observes_active_session_during_recursion():
    observations = []

    class ObservingHandler(handlers.BaseHandler):
        def flatten(self, obj, data):
            observations.append((self.context._session, self.context._depth))
            data["child"] = self.context.flatten(obj.child, reset=False)
            return data

    handlers.register(Box, ObservingHandler)
    try:
        pickler = Pickler()
        pickler.flatten(Box(SharedThing()))
    finally:
        handlers.unregister(Box)
        handlers.register(Box, BoxHandler)

    assert len(observations) == 1
    session, depth = observations[0]
    # the handler ran inside the top-level run, not at depth -1
    assert depth >= 0
    # that session is gone now; the pickler holds a fresh empty one
    assert pickler._session is not session


def test_pickler_explicit_reset_keeps_existing_effect():
    pickler = Pickler()
    shared = SharedThing()
    # simulate a mid-run state: the object table is populated
    pickler._log_ref(shared)
    assert pickler._objs != {}
    # a reset=False flatten joins the current session instead of wiping it,
    # so `shared` is referenced by its existing id
    data = pickler.flatten([shared, shared], reset=False)
    assert data == [{tags.ID: 0}, {tags.ID: 0}]
    pickler.reset()
    assert_pickler_state_empty(pickler)
    # ids restart from zero after an explicit reset: the list takes id 0
    # and the shared object is referenced by its fresh id 1
    assert pickler.flatten([shared, shared]) == [
        {tags.OBJECT: util.importable_name(SharedThing), "name": "thing"},
        {tags.ID: 1},
    ]


def test_make_refs_true_preserves_identity():
    shared = SharedThing()
    restored = jsonpickle.decode(jsonpickle.encode([shared, shared], make_refs=True))
    assert restored[0] is restored[1]


def test_make_refs_false_yields_equal_but_distinct_objects():
    shared = SharedThing()
    restored = jsonpickle.decode(jsonpickle.encode([shared, shared], make_refs=False))
    assert restored[0] == restored[1]
    assert restored[0] is not restored[1]


def test_pickler_session_holds_readonly_config_references():
    pickler = Pickler()
    session = pickler._session
    assert session.backend is json
    assert session.handlers is handlers


# ------------------------------------------------------------------
# Unpickler session boundaries
# ------------------------------------------------------------------


def test_restore_exception_cleans_state_and_allows_reuse():
    """Regression: a failing run must not poison a long-lived unpickler."""
    unpickler = Unpickler()
    doc = {tags.OBJECT: util.importable_name(Boom)}
    with pytest.raises(RuntimeError, match="boom restore"):
        unpickler.restore(doc)
    # directly verify the state is empty, not just that a retry "works"
    assert_unpickler_state_empty(unpickler)

    # immediately reuse the same instance on a cyclic graph
    result = unpickler.restore([{tags.ID: 0}])
    assert result[0] is result
    # a successful top-level restore drains the proxies and the name stack
    assert unpickler._proxies == []
    assert unpickler._namestack == []


def test_restore_missing_class_error_cleans_state():
    unpickler = Unpickler(on_missing="error")
    doc = {tags.OBJECT: "no.such.module.ClassName"}
    with pytest.raises(jsonpickle.errors.ClassNotFoundError):
        unpickler.restore(doc)
    assert_unpickler_state_empty(unpickler)


def test_decode_backend_failure_leaves_no_state():
    """A backend parse failure goes through the same cleanup path."""
    unpickler = Unpickler()
    with pytest.raises(Exception):
        jsonpickle.decode("this is not json", context=unpickler)
    assert_unpickler_state_empty(unpickler)
    # the same instance still decodes a cyclic graph afterwards
    result = jsonpickle.decode(jsonpickle.encode(make_cycle()), context=unpickler)
    assert result[0] is result


def test_handler_recursion_joins_current_unpickler_session():
    """A handler restoring with reset=False resolves references against
    the current session, preserving identity across the handler boundary."""
    shared = SharedThing()
    encoded = jsonpickle.encode({"box": Box(shared), "shared": shared})
    restored = jsonpickle.decode(encoded)
    assert restored["box"].child is restored["shared"]


def test_restore_handler_observes_single_session():
    observations = []

    class ObservingHandler(handlers.BaseHandler):
        def flatten(self, obj, data):
            data["child"] = self.context.flatten(obj.child, reset=False)
            return data

        def restore(self, data):
            session = self.context._session
            child = self.context.restore(data["child"], reset=False)
            observations.append(session is self.context._session)
            return Box(child)

    handlers.register(Box, ObservingHandler)
    try:
        encoded = jsonpickle.encode(Box(SharedThing()))
        jsonpickle.decode(encoded)
    finally:
        handlers.unregister(Box)
        handlers.register(Box, BoxHandler)

    # the recursive restore call joined the current session
    assert observations == [True]


def test_forward_reference_resolves_through_proxy():
    """A py/id pointing at a not-yet-restored object is proxied and then
    swapped for the real instance at the end of the top-level restore."""
    unpickler = Unpickler()
    result = unpickler.restore([{tags.ID: 1}, ["a", "b"]])
    assert result[0] is result[1]
    # no unresolved proxies are left behind
    assert unpickler._proxies == []
    assert unpickler._namestack == []


def test_invalid_reference_resolves_to_none_and_cleans_up():
    unpickler = Unpickler()
    result = unpickler.restore({"key": {tags.ID: 42}})
    assert result["key"] is None
    assert unpickler._proxies == []
    assert unpickler._namestack == []


def test_incremental_restore_joins_previous_session():
    """restore(reset=False) after a top-level restore resolves references
    against the previous run's object table (the bson_test.py pattern)."""
    unpickler = Unpickler()
    # the outer list takes id 0 and the dict takes id 1
    first = unpickler.restore([{"value": 1}, {tags.ID: 1}])
    assert first[1] is first[0]
    # the object table is still available for an incremental pass
    assert unpickler._objs != []
    assert unpickler.restore({tags.ID: 1}, reset=False) is first[0]


def test_unpickler_explicit_reset_keeps_existing_effect():
    unpickler = Unpickler()
    unpickler.restore([{"value": 1}])
    assert unpickler._objs != []
    unpickler.register_classes(SharedThing)
    assert unpickler._classes != {}
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)
    # reset() historically also drops the registered classes
    assert unpickler._classes == {}


def test_unpickler_session_holds_readonly_config_references():
    unpickler = Unpickler()
    session = unpickler._session
    assert session.backend is json
    assert session.handlers is handlers
    assert session.classes is unpickler._classes


def test_failed_restore_does_not_anchor_partial_objects():
    """Objects restored before a failure must not be kept alive."""
    unpickler = Unpickler()
    holder = []

    class TrackingHandler(handlers.BaseHandler):
        def restore(self, data):
            instance = Box(None)
            holder.append(weakref.ref(instance))
            raise RuntimeError("boom restore")

    handlers.register(Box, TrackingHandler)
    try:
        doc = {tags.OBJECT: util.importable_name(Box)}
        with pytest.raises(RuntimeError):
            unpickler.restore(doc)
    finally:
        handlers.unregister(Box)
        handlers.register(Box, BoxHandler)

    assert_unpickler_state_empty(unpickler)
    gc.collect()
    assert holder[0]() is None


# ------------------------------------------------------------------
# Shared graphs through the public API
# ------------------------------------------------------------------


def test_repeated_top_level_runs_are_independent():
    """Two runs on the same long-lived codecs produce identical output and
    do not share reference identity across runs."""
    pickler = Pickler()
    unpickler = Unpickler()

    def graph():
        shared = SharedThing()
        return {"items": [shared, shared], "cycle": make_cycle()}

    first = pickler.flatten(graph())
    second = pickler.flatten(graph())
    assert first == second
    assert_pickler_state_empty(pickler)

    decoded_first = unpickler.restore(first)
    assert decoded_first["items"][0] is decoded_first["items"][1]
    assert decoded_first["cycle"][0] is decoded_first["cycle"]
    decoded_second = unpickler.restore(second)
    assert decoded_second["items"][0] is decoded_second["items"][1]
    # identity is preserved within a run, never across runs
    assert decoded_first["items"][0] is not decoded_second["items"][0]
