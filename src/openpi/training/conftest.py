"""Conftest for training tests — mocks broken tensorflow binary so config-only tests can run."""
import sys
import types
from importlib.util import spec_from_loader


def _mock_tensorflow_if_broken() -> None:
    """Insert a stub tensorflow into sys.modules before any test collection begins.

    This is needed because the tensorflow binary installed in this environment has a
    broken shared library (undefined symbol in libtensorflow_cc.so.2). Config-only
    tests do not need tensorflow at runtime; they only fail because `transformers`
    calls `importlib.util.find_spec('tensorflow')` during its own import, which
    triggers the broken binary. The stub prevents that.

    The stub is NOT inserted if tensorflow is already loadable.
    """
    if "tensorflow" in sys.modules:
        return
    try:
        import importlib.util as _ilu
        # Attempt a real import probe without executing package code.
        spec = _ilu.find_spec("tensorflow")
        if spec is None:
            return
        # If find_spec succeeds, try a lightweight load to see if the binary is broken.
        import tensorflow  # noqa: F401
        return  # TF is fine — don't mock
    except (ImportError, ValueError, AttributeError):
        pass

    # Install stub for tensorflow and the sub-packages that are probed during
    # `transformers` import (`_api.v2.*`, `python.platform.*`, etc.).
    _stubs = [
        "tensorflow",
        "tensorflow.python",
        "tensorflow.python.platform",
        "tensorflow.python.platform._pywrap_cpu_feature_guard",
        "tensorflow._api",
        "tensorflow._api.v2",
        "tensorflow._api.v2.__internal__",
        "tensorflow._api.v2.__internal__.autograph",
        "tensorflow.python.autograph",
        "tensorflow.python.autograph.core",
        "tensorflow.python.autograph.core.ag_ctx",
        "tensorflow.python.autograph.utils",
        "tensorflow.python.autograph.utils.context_managers",
        "tensorflow.python.framework",
        "tensorflow.python.framework.ops",
        "tensorflow.python.pywrap_tensorflow",
    ]
    for name in _stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__spec__ = spec_from_loader(name, loader=None)
            sys.modules[name] = mod


# Run at import time so it is in effect before pytest collection.
_mock_tensorflow_if_broken()
