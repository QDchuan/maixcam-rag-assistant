@echo off
rem ---------------------------------------------------------------
rem  Start the MaixCAM assistant front end.
rem
rem    harness\start.cmd          -> port 8787
rem    harness\start.cmd 9000     -> another port
rem
rem  It boots the `web` profile with one extra patch layer
rem  (maixcam-default.yml) that makes "MaixCAM assistant" the
rem  default agent preset. Without that layer you would get a
rem  generic coding agent that happens to have a knowledge panel;
rem  the answer discipline lives in the preset, not in the shell.
rem
rem  NOTE: this file is deliberately ASCII-only. cmd.exe parses a
rem  .cmd file using the ANSI code page that was active when it
rem  started, so UTF-8 Chinese comments would be misread as
rem  commands. Chinese text belongs in the README, not here.
rem ---------------------------------------------------------------

setlocal
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8787"
set "ROOT=%~dp0.."

pushd "%ROOT%"
echo [maixcam] repo root : %ROOT%
echo [maixcam] starting  : dsh --profile web --port %PORT%
echo [maixcam] a browser window should open with the URL
echo.

dsh --profile web --patch "%~dp0maixcam-default.yml" --port %PORT%

echo.
echo [maixcam] stopped. Press any key to close.
pause >nul
popd
endlocal
