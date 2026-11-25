"""Puts scripts/ on the import path, so the tests can say `import summarize_bench`."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
