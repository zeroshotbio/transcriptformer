import pathlib
import sys

import pycurl

print("PycURL version :", pycurl.version)
print("Loaded from     :", pathlib.Path(pycurl.__file__).resolve())
print("Python          :", sys.version.split()[0])
