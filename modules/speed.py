"""
Speed helpers for tasks: TF32 matmul + torch.compile.

Usage in each task:
    from modules.speed import setup_fast_env, maybe_compile

    setup_fast_env()                    # always: enable TF32 (harmless if not used)
    ...
    parser.add_argument('--fast', action='store_true')
    ...
    model = maybe_compile(model, args.fast)

Notes:
- AMP is intentionally NOT enabled — empirically it costs ~2.5-6 dB PSNR for
  both cfloat BLA and float BLA without giving real speedup.
- torch.compile requires real-valued model (e.g. --nonlin bla_float, siren).
  cfloat BLA crashes inductor; the wrapper catches the exception and falls
  back to eager.
- /tmp is mounted noexec on this cluster, so triton kernel cache must point
  to a writable + executable directory. We default to <project>/tmp/torch_cache.
"""

import os
import torch

# Place triton/inductor cache under the project (avoid /tmp noexec issues).
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE = os.path.join(_PROJ_ROOT, 'tmp', 'torch_cache')


def setup_fast_env(cache_dir: str = None):
    """Enable TF32 + configure triton cache dir. Safe to call multiple times."""
    torch.set_float32_matmul_precision('high')
    cache = cache_dir or _DEFAULT_CACHE
    os.environ.setdefault('TRITON_CACHE_DIR', cache)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache)
    os.makedirs(cache, exist_ok=True)


def maybe_compile(model, enabled: bool, mode: str = 'default', fullgraph: bool = False):
    """Apply torch.compile if enabled. Falls back to eager on failure."""
    if not enabled:
        return model
    setup_fast_env()
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        print(f'[fast] torch.compile applied (mode={mode})')
        return compiled
    except Exception as e:
        print(f'[fast] torch.compile failed, falling back to eager: {e}')
        return model


def resolve_fast_nonlin(nonlin: str, fast: bool, fast_default: str = 'bla_float') -> str:
    """If --fast is set with cfloat 'bla' (compile-incompatible), auto-switch to bla_float.
    Other nonlins (siren, wire, gauss, etc.) pass through unchanged."""
    if fast and nonlin == 'bla':
        print(f"[fast] auto-switching --nonlin bla → {fast_default} (cfloat incompatible with torch.compile)")
        return fast_default
    return nonlin
