"""
Module where the version is written.

It is executed in setup.py and imported in fluidimage/__init__.py.

See:

http://en.wikipedia.org/wiki/Software_versioning
http://legacy.python.org/dev/peps/pep-0386/

'a' or 'alpha' means alpha version (internal testing),
'b' or 'beta' means beta version (external testing).
"""
__version__ = "0.1.5"

try:
    from fluidimage._hg_rev import hg_rev
except ImportError:
    hg_rev = "?"
