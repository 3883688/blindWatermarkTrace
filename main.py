import sys as _sys

from trace_app.compat import install_main_module as _install

_install(_sys.modules[__name__])
del _install
del _sys
