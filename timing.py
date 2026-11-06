"""Optional scalar timing bridge; never read files/tensors or change node results."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
from time import monotonic

try:
    from comfy import execution_metrics as _metrics
except ImportError:
    _metrics = None

_operation = ContextVar('refmod_timing_operation', default=None)


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
