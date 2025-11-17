"""usdview helper that reloads the current stage when the file timestamp changes."""
import os
from pxr import Usd  # noqa: F401  # usdview injects pxr already

_STAGE = appController._dataModel.stage  # type: ignore[name-defined]
_LAYER = _stage_layer = _STAGE.GetRootLayer()
try:
    _MTIME = os.path.getmtime(_stage_layer.realPath)
except OSError:
    _MTIME = None


def _tick():
    global _MTIME
    layer = appController._dataModel.stage.GetRootLayer()  # type: ignore[name-defined]
    try:
        current = os.path.getmtime(layer.realPath)
    except OSError:
        return
    if _MTIME is None or current != _MTIME:
        _MTIME = current
        try:
            appController._usdviewApi.reopenStage()  # type: ignore[attr-defined]
        except Exception:
            pass


appController._timerDelay = 0.5  # type: ignore[attr-defined]
appController._timerCallback = _tick  # type: ignore[attr-defined]
