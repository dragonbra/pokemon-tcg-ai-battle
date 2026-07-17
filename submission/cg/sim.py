import ctypes
import os
import platform
import sys


def _library_candidates() -> list[str]:
    """Return official cg library names for the current platform.

    ``PTCG_CG_LIBRARY`` is intentionally supported for local A/B tests with
    a source-built engine. Submission packaging still copies the normal
    platform libraries from this directory.
    """
    configured = os.environ.get("PTCG_CG_LIBRARY")
    if configured:
        return [configured]

    directory = os.path.dirname(os.path.abspath(__file__))
    machine = platform.machine().lower()
    if os.name == "nt":
        names = ["cg.dll"]
    elif sys.platform == "darwin":
        names = ["libcg.dylib", "libcg.so"]
    elif machine in {"aarch64", "arm64"}:
        names = ["libcg-arm64.so", "libcg.so"]
    else:
        names = ["libcg.so"]
    return [os.path.join(directory, name) for name in names]


def _load_library():
    errors = []
    for path in _library_candidates():
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    attempted = "\n".join(errors) or "no candidate library"
    raise OSError(f"Unable to load the official cg library. Tried:\n{attempted}")


class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]

class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int)
    ]

lib = _load_library()

lib.GameInitialize.restype = None
lib.GameInitialize.argtypes = []
lib.GameInitialize()

lib.BattleStart.restype = StartData
lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

lib.AgentStart.restype = ctypes.c_void_p

lib.BattleFinish.argtypes = [ctypes.c_void_p]

lib.GetBattleData.restype = SerialData
lib.GetBattleData.argtypes = [ctypes.c_void_p]

lib.Select.restype = ctypes.c_int
lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.VisualizeData.restype = ctypes.c_char_p
lib.VisualizeData.argtypes = [ctypes.c_void_p]

lib.SearchBegin.restype = ctypes.c_char_p
lib.SearchBegin.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int]

lib.SearchStep.restype = ctypes.c_char_p
lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.SearchEnd.argtypes = [ctypes.c_void_p]

lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]

lib.AllCard.restype = ctypes.c_char_p

lib.AllAttack.restype = ctypes.c_char_p

class Battle:
    battle_ptr = None
    obs = None
