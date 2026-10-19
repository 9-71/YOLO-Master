"""Backward-compatible import alias for :mod:`studio.ui.i18n`."""

import sys

from studio.ui import i18n as _implementation

sys.modules[__name__] = _implementation
