import sys

from trace_app.compat import install_main_module

install_main_module(sys.modules[__name__])
