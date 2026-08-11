# Import first in every script here: puts ODSA/ on sys.path and makes it the
# working dir, so pipeline imports and relative data paths resolve as before.
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
