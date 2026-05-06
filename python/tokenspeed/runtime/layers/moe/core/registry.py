# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from tokenspeed.runtime.layers.moe.core.types import BackendKey

_REGISTRY: dict[BackendKey, type] = {}


def register_backend_family(*, quant: str, impl: str, cls: type) -> None:
    supported_arches = getattr(cls, "supported_arches", frozenset({"any"}))
    if not supported_arches:
        raise ValueError(f"{cls.__name__} declares no supported_arches")

    normalized_arches = (
        ("any",) if "any" in supported_arches else tuple(sorted(supported_arches))
    )
    for arch in normalized_arches:
        key = BackendKey(arch=arch, quant=quant, impl=impl)
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"MoE backend key {key} is already registered to {existing.__name__}"
            )
        _REGISTRY[key] = cls


def get_backend_cls(key: BackendKey):
    cls = _REGISTRY.get(key)
    if cls is not None:
        return cls

    if key.arch != "any":
        cls = _REGISTRY.get(BackendKey(arch="any", quant=key.quant, impl=key.impl))
        if cls is not None:
            return cls

    raise KeyError(f"No MoE backend registered for {key}")
