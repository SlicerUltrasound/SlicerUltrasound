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
