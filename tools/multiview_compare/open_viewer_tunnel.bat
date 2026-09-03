@echo off
setlocal
set "LOCAL_PORT=18765"
set "REMOTE_PORT=18765"

echo Starting SSH tunnel to the multi-model comparison viewer...
start "Splat Comparison Viewer Tunnel" /min ssh -o ExitOnForwardFailure=yes -N -L %LOCAL_PORT%:127.0.0.1:%REMOTE_PORT% vllm1
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:%LOCAL_PORT%/

echo Comparison viewer opened at http://127.0.0.1:%LOCAL_PORT%/
echo Close the minimized "Splat Comparison Viewer Tunnel" window to stop the tunnel.
timeout /t 5
