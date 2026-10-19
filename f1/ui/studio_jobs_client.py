"""Backward-compatible import alias for :mod:`studio.ui.studio_jobs_client`."""

import sys

from studio.ui import studio_jobs_client as _implementation

sys.modules[__name__] = _implementation
