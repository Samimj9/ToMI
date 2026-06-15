"""tomi.hooks — dynamic hooking infrastructure."""

from tomi.hooks.cache import make_cache_hook, make_fn_patch_hook, make_patch_hook
from tomi.hooks.hook_manager import HookManager
from tomi.hooks.hook_point import HookFn, HookPoint
from tomi.hooks.naming import (
    ParsedHookName,
    attn_hook,
    embed_hook,
    mlp_hook,
    parse_hook_name,
    resid_hook,
    unembed_hook,
    validate_hook_name,
)

__all__ = [
    "HookManager",
    "HookPoint",
    "HookFn",
    "make_cache_hook",
    "make_patch_hook",
    "make_fn_patch_hook",
    "ParsedHookName",
    "attn_hook",
    "mlp_hook",
    "resid_hook",
    "embed_hook",
    "unembed_hook",
    "parse_hook_name",
    "validate_hook_name",
]
