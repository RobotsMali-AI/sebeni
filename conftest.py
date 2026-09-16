import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "beni")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
