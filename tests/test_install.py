import sys
from pathlib import Path

import pytest

import mstitch


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_module_needs_no_cpp_runtime_dlls():
    from mstitch._dlls import imports, missing

    pyd = next(Path(mstitch.__file__).parent.glob("_core*.pyd"))
    names = [n.lower() for n in imports(pyd)]
    print(names)
    assert not any(n.startswith(("msvcp", "vcruntime", "libstdc++", "libgcc", "libwinpthread")) for n in names)
    assert missing(pyd) == []
