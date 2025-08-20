# Debugging

## VS Code

1. Setup VS Code Debugger
Create a .vscode/launch.json file:
```json
{
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "port": 5678,
                "host": "localhost"
            },
            "justMyCode": false,
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "${workspaceFolder}"
                }
            ]
        }
    ]
}
```

Create a .vscode/settings.json file:
```json
{
    "python.defaultInterpreterPath": "/Applications/Slicer.app/Contents/bin/PythonSlicer",
    "python.terminal.activateEnvironment": false,
    "python.autoComplete.extraPaths": [
        "/Applications/Slicer.app/Contents/lib/Slicer-5.8/qt-scripted-modules",
        "/Applications/Slicer.app/Contents/lib/Slicer-5.8/qt-loadable-modules",
        "/Applications/Slicer.app/Contents/lib/Python/lib/python3.9/site-packages",
    ],
    "python.analysis.extraPaths": [
        "/Applications/Slicer.app/Contents/lib/Slicer-5.8/qt-scripted-modules",
        "/Applications/Slicer.app/Contents/lib/Slicer-5.8/qt-loadable-modules",
        "/Applications/Slicer.app/Contents/lib/Python/lib/python3.9/site-packages",
    ]
}
```

2. Slicer Python Console
```python
import debugpy
debugpy.listen(("localhost", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached!")
debugpy.breakpoint()
```

3. Run and Debug
Open `Run and Debug` view in VS Code, select `Python: Remote Attach` and click `Run`

4. Add breakpoints


## Useful scripts for debugging

```python
annotateWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()
annotateLogic = annotateWidget.logic
annotateNode = annotateLogic.getParameterNode()

anonymizeWidget = slicer.modules.anonymizeultrasound.widgetRepresentation().self()
anonymizeLogic = anonymizeWidget.logic
anonymizeNode = anonymizeLogic.getParameterNode()
```

```python
import os
import numpy as np
import cv2

# Save a frame as a PNG file.
def save_frame_png(frame_item: np.ndarray, out_path: str) -> bool:
    out_path = os.path.expanduser(out_path)
    img = frame_item

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img.dtype.kind == 'f':
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        if vmax > vmin:
            img = ((img - vmin) / (vmax - vmin) * 255.0).round().astype(np.uint8)
        else:
            img = np.zeros(img.shape[:2], dtype=np.uint8)
    elif img.dtype == np.uint16:
        pass
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return cv2.imwrite(out_path, img)

# Example usage:
ok = save_frame_png(frame_item, os.path.expanduser("~/Downloads/frame_item.png"))
```
