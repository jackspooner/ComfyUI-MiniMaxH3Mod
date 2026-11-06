"""Optional scalar timing bridge; never read files/tensors or change node results."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
from time import monotonic

import torch

try:
    from comfy import execution_metrics as _metrics
except ImportError:
    _metrics = None

_operation = ContextVar('refmod_timing_operation', default=None)
_application = ContextVar('refmod_timing_application', default=None)
_reference = ContextVar('refmod_timing_reference', default=None)


def enabled():
    try:
        return _metrics is not None and _metrics.enabled()
    except Exception:
        return False


def emit(event, **facts):
    try:
        binding = _operation.get()
        if binding is not None and enabled():
            _metrics.emit('refmod_loader_' + event, binding=binding, **facts)
    except Exception:
        pass


def loader(mode):
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            if not enabled():
                return function(*args, **kwargs)
            token = _operation.set(object())
            emit('begin', mode=mode, slots=8, version=1)
            try:
                with measure('total'):
                    result = function(*args, **kwargs)
                emit('end', references=len(result[0]))
                return result
            finally:
                _operation.reset(token)
        return run
    return decorate


@contextmanager
def measure(phase, slot=None):
    if _operation.get() is None or not enabled():
        yield
        return
    started = monotonic()
    try:
        yield
    except BaseException:
        emit('duration', phase=phase, slot=slot, seconds=max(0, monotonic() - started), succeeded=False)
        raise
    else:
        emit('duration', phase=phase, slot=slot, seconds=max(0, monotonic() - started), succeeded=True)


def loaded(slot, mod, reused, container, stamp=None):
    if _operation.get() is None or not enabled():
        return
    try:
        # A cache owner's already-read stamp is audit evidence, not a content hash.
        identity = hashlib.sha256(json.dumps(stamp, separators=(',', ':')).encode()).hexdigest() if stamp else None
        emit('loaded', slot=slot, container=container, reused=reused,
             kind=mod.kind, shape=tuple(mod.latent.shape), precision=str(mod.latent.dtype),
             device=str(mod.latent.device), tokens=mod.token_count,
             member=mod.bundle_index if container == 'bundle' else None,
             stamp_identity=identity, stamp_assurance='size_mtime_ctime' if identity else 'unavailable')
    except Exception:
        emit('metadata_unavailable', slot=slot)


def apply_emit(event, **facts):
    try:
        binding, index = _application.get(), _reference.get()
        if binding is not None and index != -1 and enabled():
            _metrics.emit('refmod_apply_' + event, binding=binding, index=index, **facts)
    except Exception:
        pass


def apply_tensor(event, tensor, **facts):
    if _application.get() is None or _reference.get() == -1 or not enabled():
        return
    try:
        apply_emit(event, shape=tuple(tensor.shape), precision=str(tensor.dtype), device=str(tensor.device),
                   requires_grad=bool(tensor.requires_grad), default_precision=str(torch.get_default_dtype()),
                   grad_enabled=torch.is_grad_enabled(), inference_mode=torch.is_inference_mode_enabled(), **facts)
    except Exception:
        apply_emit('metadata_unavailable')


def apply_plan(mod, branch, weights, strength, native):
    if _application.get() is None or _reference.get() == -1 or not enabled():
        return
    try:
        apply_tensor('reference_plan', mod.latent, native=native, kind=mod.kind,
                     frames=mod.latent_t, branch=branch, weights=weights, strength=strength)
    except Exception:
        # In particular, a dropped custom reference need not expose a latent.
        apply_emit('metadata_unavailable')


@contextmanager
def apply_measure(phase):
    if _application.get() is None or _reference.get() == -1 or not enabled():
        yield
        return
    started = monotonic()
    try:
        yield
    except BaseException:
        apply_emit('duration', phase=phase, seconds=max(0, monotonic() - started), succeeded=False)
        raise
    else:
        apply_emit('duration', phase=phase, seconds=max(0, monotonic() - started), succeeded=True)


def apply_measured(phase):
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            if _application.get() is None or _reference.get() == -1 or not enabled():
                return function(*args, **kwargs)
            with apply_measure(phase):
                return function(*args, **kwargs)
        return run
    return decorate


def application(function):
    @wraps(function)
    def run(*args, **kwargs):
        if not enabled():
            return function(*args, **kwargs)
        token = _application.set(object())
        apply_emit('begin', version=1)
        try:
            with apply_measure('total'):
                result = function(*args, **kwargs)
            apply_emit('end')
            return result
        finally:
            _application.reset(token)
    return run


def apply_selection(mods, items, scrambled):
    if _application.get() is None or not enabled():
        return ()
    try:
        # Shuffling keeps the original row objects. Repeated identical copies
        # point to their first equivalent input ordinal, without reading a latent.
        sources = {}
        for index, item in enumerate(mods):
            sources.setdefault(id(item), index)
        apply_emit('selection', inputs=len(mods), selected=len(items), scrambled=scrambled)
        return tuple(sources[id(item)] for item in items[:256])
    except Exception:
        apply_emit('metadata_unavailable')
        return ()


@contextmanager
def apply_reference(index, sources=()):
    # A legal bundle can expand into many copies. Keep telemetry bounded while
    # executing every original item and exposing the first omitted observation.
    if _application.get() is None or not enabled():
        yield
        return
    if index == 256:
        apply_emit('limit')
    token = _reference.set(index if index < 256 else -1)
    try:
        apply_emit('reference_begin', source=sources[index] if index < len(sources) else None)
        yield
    finally:
        _reference.reset(token)
