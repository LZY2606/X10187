"""Characterization tests for the internal encode/decode session lifecycle.

A "session" is the mutable state used by a single top-level encode or
decode run: the object identity tables, recursion depth, name stack and
unresolved proxies.  These tests pin down the session boundaries:

* A top-level call cleans up after itself, whether it returns normally
  or raises.
* Recursive calls made by custom handlers join the session that is
  already in progress instead of starting a fresh one (which would lose
  object identity).
* Calling reset() explicitly keeps its long-standing effect.
* Read-only configuration (backend, class registry, handler registry)
  is shared with the codec, while per-run tables never linger on a
  long-lived Pickler/Unpickler instance.
"""

import pytest

import jsonpickle
import jsonpickle.errors
import jsonpickle.handlers
from jsonpickle.pickler import Pickler
from jsonpickle.unpickler import Unpickler


class Node:
    """A simple object that can be shared and can form cycles"""

    def __init__(self, name=None):
        self.name = name
        self.peer = None

    def __eq__(self, other):
        return (
            isinstance(other, Node)
            and self.name == other.name
            and self.peer == other.peer
        )

    def __hash__(self):
        return hash(self.name)


class Box:
    """An object serialized through a custom handler"""

    def __init__(self, item=None):
        self.item = item

    def __eq__(self, other):
        return isinstance(other, Box) and self.item == other.item


class BoxHandler(jsonpickle.handlers.BaseHandler):
    """Recursively flattens/restores through the active context"""

    def flatten(self, obj, data):
        data["item"] = self.context.flatten(obj.item, reset=False)
        return data

    def restore(self, obj):
        return Box(self.context.restore(obj["item"], reset=False))


class Boom:
    pass


BOOM_CLASS_NAME = f"{Boom.__module__}.{Boom.__qualname__}"


class BoomHandler(jsonpickle.handlers.BaseHandler):
    """A handler that always fails, in both directions"""

    def flatten(self, obj, data):
        raise RuntimeError("flatten boom")

    def restore(self, obj):
        raise RuntimeError("restore boom")


@pytest.fixture
def box_handler():
    jsonpickle.handlers.register(Box, BoxHandler)
    yield
    jsonpickle.handlers.unregister(Box)


@pytest.fixture
def boom_handler():
    jsonpickle.handlers.register(Boom, BoomHandler)
    yield
    jsonpickle.handlers.unregister(Boom)


def assert_pickler_state_empty(pickler):
    """No per-run state may linger on the pickler"""
    assert pickler._objs == {}
    assert pickler._depth == -1
    assert pickler._seen == []
    assert pickler._flattened == {}


def assert_unpickler_state_empty(unpickler):
    """No per-run state may linger on the unpickler"""
    assert unpickler._namedict == {}
    assert unpickler._namestack == []
    assert unpickler._obj_to_idx == {}
    assert unpickler._objs == []
    assert unpickler._proxies == []


def assert_unpickler_session_finalized(unpickler):
    """A completed restore leaves a finalized, resolvable session.

    Proxies have been swapped in and the name stack is unwound; the
    object tables remain available so that incremental
    restore(reset=False) calls can keep resolving references, until an
    explicit reset() or the next reset=True call releases them.
    """
    assert unpickler._namestack == []
    assert unpickler._proxies == []


def make_cycle():
    node = Node("cycle")
    node.peer = node
    return node


# -- top-level session boundaries ----------------------------------------


def test_pickler_state_clean_after_successful_flatten():
    pickler = Pickler()
    pickler.flatten(make_cycle())
    assert_pickler_state_empty(pickler)


def test_unpickler_session_finalized_after_successful_restore():
    unpickler = Unpickler()
    unpickler.restore(jsonpickle.backend.json.decode(jsonpickle.encode(make_cycle())))
    assert_unpickler_session_finalized(unpickler)
    # the tables stay resolvable for incremental restores...
    assert unpickler._objs != []
    # ...and are released by an explicit reset()
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


def test_encode_with_reused_context_leaves_no_state():
    pickler = Pickler()
    jsonpickle.encode(make_cycle(), context=pickler)
    assert_pickler_state_empty(pickler)


def test_decode_with_ephemeral_context_leaves_no_state():
    """jsonpickle.decode() never leaks session state onto its context"""
    unpickler = Unpickler()
    jsonpickle.decode(jsonpickle.encode(make_cycle()))
    # a user-supplied context is finalized but retains its object tables
    # for incremental use (see test_unpickler_restore_without_reset_*)
    jsonpickle.decode(jsonpickle.encode(make_cycle()), context=unpickler)
    assert_unpickler_session_finalized(unpickler)
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


def test_successive_decodes_with_same_unpickler():
    """Each reset=True decode starts a fresh session on the same instance"""
    unpickler = Unpickler()
    first = unpickler.restore(
        jsonpickle.backend.json.decode(jsonpickle.encode(make_cycle()))
    )
    assert first.peer is first
    second = unpickler.restore(
        jsonpickle.backend.json.decode(jsonpickle.encode(make_cycle()))
    )
    assert second.peer is second
    assert second is not first


def test_pickler_state_clean_after_flatten_without_reset():
    """A top-level flatten(reset=False) is still a complete session"""
    pickler = Pickler()
    pickler.flatten(make_cycle(), reset=False)
    assert_pickler_state_empty(pickler)


def test_unpickler_restore_without_reset_retains_session():
    """restore(reset=False) deliberately retains state for incremental use.

    This is the documented contract of reset=False: the caller manages
    the session and releases it with an explicit reset().
    """
    unpickler = Unpickler()
    payload = jsonpickle.backend.json.decode(jsonpickle.encode(Node("x")))
    unpickler.restore(payload, reset=False)
    assert unpickler._objs != []
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


def test_unpickler_incremental_restore_resolves_cross_call_references():
    """Incremental restore(reset=False) calls share one session.

    A py/id reference restored by a later call must resolve against an
    object registered by an earlier call on the same unpickler.
    """
    unpickler = Unpickler()
    shared = unpickler.restore(["a", "b"], reset=False)
    ref = unpickler.restore({jsonpickle.tags.ID: 0}, reset=False)
    assert ref is shared
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


def test_pickler_state_clean_after_handler_exception(boom_handler):
    pickler = Pickler()
    with pytest.raises(RuntimeError, match="flatten boom"):
        pickler.flatten([1, 2, Boom()])
    assert_pickler_state_empty(pickler)


def test_unpickler_state_clean_after_handler_exception(boom_handler):
    unpickler = Unpickler()
    payload = [1, 2, {"py/object": BOOM_CLASS_NAME}]
    with pytest.raises(RuntimeError, match="restore boom"):
        unpickler.restore(payload)
    assert_unpickler_state_empty(unpickler)


def test_unpickler_state_clean_after_unresolvable_class():
    """An illegal class reference aborts the session through the same path"""
    unpickler = Unpickler(on_missing="error")
    payload = {"outer": [{"py/object": "no.such.Module"}, {"py/object": "still.missing"}]}
    with pytest.raises(jsonpickle.errors.ClassNotFoundError):
        unpickler.restore(payload)
    assert_unpickler_state_empty(unpickler)


def test_unpickler_state_clean_after_backend_parse_failure():
    unpickler = Unpickler()
    with pytest.raises(ValueError):
        jsonpickle.decode("{not valid json", context=unpickler)
    assert_unpickler_state_empty(unpickler)


def test_failed_encode_then_successful_cyclic_encode_same_pickler(boom_handler):
    """A pickler that just failed must handle a cyclic graph correctly"""
    pickler = Pickler()
    with pytest.raises(RuntimeError):
        pickler.flatten([1, 2, Boom()])
    assert_pickler_state_empty(pickler)

    cyclic = []
    cyclic.append(cyclic)
    flattened = pickler.flatten(cyclic)
    assert flattened == [{jsonpickle.tags.ID: 0}]
    clone = jsonpickle.decode(jsonpickle.backend.json.encode(flattened))
    assert clone[0] is clone
    assert_pickler_state_empty(pickler)


def test_failed_decode_then_successful_cyclic_decode_same_unpickler(boom_handler):
    """An unpickler that just failed must handle a cyclic graph correctly"""
    unpickler = Unpickler()
    payload = [1, 2, {"py/object": BOOM_CLASS_NAME}]
    with pytest.raises(RuntimeError):
        unpickler.restore(payload)
    assert_unpickler_state_empty(unpickler)

    cyclic = []
    cyclic.append(cyclic)
    clone = unpickler.restore(jsonpickle.backend.json.decode(jsonpickle.encode(cyclic)))
    assert clone[0] is clone
    assert_unpickler_session_finalized(unpickler)
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


# -- recursive handler calls join the active session ----------------------


def test_handler_recursion_preserves_identity(box_handler):
    """Nested flatten(reset=False) must join the session, not start one.

    If the recursive call lost the session, ``shared`` would be flattened
    twice and the decoded objects would not be identical.
    """
    shared = Node("shared")
    payload = jsonpickle.encode([Box(shared), shared])
    assert jsonpickle.tags.ID in payload  # a py/id reference was emitted
    clone = jsonpickle.decode(payload)
    assert clone[0].item is clone[1]


def test_handler_recursion_preserves_identity_on_decode(box_handler):
    """Nested restore(reset=False) must resolve proxies in the session"""
    first = Node("first")
    second = Node("second")
    first.peer = second
    second.peer = first
    payload = jsonpickle.encode([Box(first), second])
    clone = jsonpickle.decode(payload)
    assert clone[0].item.peer is clone[1]
    assert clone[1].peer is clone[0].item


def test_handler_recursion_with_reused_context(box_handler):
    pickler = Pickler()
    unpickler = Unpickler()
    shared = Node("shared")
    payload = jsonpickle.encode([Box(shared), shared], context=pickler)
    clone = jsonpickle.decode(payload, context=unpickler)
    assert clone[0].item is clone[1]
    assert_pickler_state_empty(pickler)
    assert_unpickler_session_finalized(unpickler)
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


# -- make_refs: identity vs equality --------------------------------------


def test_make_refs_true_shared_object_stays_identical():
    """With make_refs=True a shared sub-object must remain one object"""
    shared = {"a": [1, 2, 3]}
    clone = jsonpickle.decode(jsonpickle.encode([shared, shared], make_refs=True))
    assert clone[0] == clone[1]
    assert clone[0] is clone[1]


def test_make_refs_false_shared_object_becomes_equal_but_distinct():
    """With make_refs=False duplicates are equal but not identical"""
    shared = {"a": [1, 2, 3]}
    clone = jsonpickle.decode(jsonpickle.encode([shared, shared], make_refs=False))
    assert clone[0] == clone[1]
    assert clone[0] is not clone[1]


def test_make_refs_false_pickler_state_clean_between_runs():
    """The make_refs=False flatten cache is per-session state"""
    pickler = Pickler(make_refs=False)
    shared = {"a": [1, 2, 3]}
    first = pickler.flatten([shared, shared])
    assert_pickler_state_empty(pickler)
    second = pickler.flatten([shared, shared])
    assert first == second
    assert_pickler_state_empty(pickler)


def test_make_refs_false_cycle_is_broken():
    cyclic = []
    cyclic.append(cyclic)
    encoded = jsonpickle.encode(cyclic, make_refs=False)
    assert isinstance(encoded, str)
    assert "..." in encoded  # the cycle was replaced by repr()


def test_cycle_roundtrip_preserves_identity():
    cyclic = []
    cyclic.append(cyclic)
    clone = jsonpickle.decode(jsonpickle.encode(cyclic))
    assert clone[0] is clone


def test_object_cycle_roundtrip_preserves_identity():
    first = Node("first")
    second = Node("second")
    first.peer = second
    second.peer = first
    clone = jsonpickle.decode(jsonpickle.encode(first))
    assert clone.peer.peer is clone


# -- forward references and proxy resolution -------------------------------


def test_forward_reference_resolves_to_same_object():
    """A py/id referring to a later object is resolved via a proxy"""
    unpickler = Unpickler()
    clone = jsonpickle.decode('[{"py/id": 1}, ["a", "b"]]', context=unpickler)
    assert clone[0] is clone[1]
    assert_unpickler_session_finalized(unpickler)
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


def test_forward_reference_failure_still_cleans_proxies():
    """Pending proxies must not survive a failed session"""
    unpickler = Unpickler(on_missing="error")
    payload = '[{"py/id": 2}, ["a", "b"], {"py/object": "no.such.Module"}]'
    with pytest.raises(jsonpickle.errors.ClassNotFoundError):
        jsonpickle.decode(payload, context=unpickler)
    assert_unpickler_state_empty(unpickler)


# -- explicit reset keeps its existing effect ------------------------------


def test_explicit_pickler_reset_clears_state():
    pickler = Pickler()
    pickler._log_ref(object())
    pickler._push()
    assert pickler._objs != {}
    assert pickler._depth != -1
    pickler.reset()
    assert_pickler_state_empty(pickler)


def test_explicit_unpickler_reset_clears_state():
    unpickler = Unpickler()
    unpickler._mkref(object())
    unpickler._namestack.append("key")
    unpickler.register_classes(Node)
    assert unpickler._objs != []
    assert unpickler._classes != {}
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)
    # reset() has always cleared the locally registered classes too
    assert unpickler._classes == {}


# -- configuration outlives the session ------------------------------------


def test_class_registry_survives_session_teardown():
    """The class registry is codec configuration, not per-run state"""
    unpickler = Unpickler()
    payload = jsonpickle.decode(jsonpickle.encode(Node("registered")))
    clone = unpickler.restore(payload, classes=[Node])
    assert isinstance(clone, Node)
    assert clone.name == "registered"
    # the session is finalized and the registry remains
    assert_unpickler_session_finalized(unpickler)
    assert unpickler._classes != {}


def test_manually_registered_classes_survive_incremental_restore():
    """register_classes() pairs with restore(reset=False)"""
    unpickler = Unpickler()
    unpickler.register_classes(Node)
    payload = jsonpickle.decode(jsonpickle.encode(Node("registered")))
    clone = unpickler.restore(payload, reset=False)
    assert isinstance(clone, Node)
    assert clone.name == "registered"
    assert unpickler._classes != {}
    unpickler.reset()
    assert_unpickler_state_empty(unpickler)


def test_backend_and_options_survive_session_teardown():
    pickler = Pickler()
    unpickler = Unpickler()
    backend = pickler.backend
    pickler.flatten(make_cycle())
    unpickler.restore(jsonpickle.backend.json.decode(jsonpickle.encode(make_cycle())))
    assert pickler.backend is backend
    assert unpickler.backend is backend
    assert pickler.unpicklable is True
    assert unpickler.keys is True


def test_session_state_is_not_shared_between_codecs():
    """Two codecs never share per-run state"""
    pickler_a = Pickler()
    pickler_b = Pickler()
    pickler_a.flatten(make_cycle())
    assert_pickler_state_empty(pickler_a)
    assert_pickler_state_empty(pickler_b)
    assert pickler_a._objs is not pickler_b._objs
