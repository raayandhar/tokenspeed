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

import importlib
import logging

from tokenspeed._logging import suppress_noisy_third_party_logs

_suppress_noisy_third_party_logs = suppress_noisy_third_party_logs


def _suppress_flash_attn_jit_cache_debug_log():
    module_name = "flash_attn.cute.cache_utils"
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        importlib.import_module(module_name)
    except ImportError:
        return
    finally:
        logging.disable(previous_disable_level)

    logger_obj = logging.getLogger(module_name)
    logger_obj.setLevel(logging.WARNING)
    for handler in logger_obj.handlers:
        handler.setLevel(logging.WARNING)


_suppress_noisy_third_party_logs()
_suppress_flash_attn_jit_cache_debug_log()
