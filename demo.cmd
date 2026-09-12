@echo off
rem ---------------------------------------------------------------------------
rem MaixCAM 开发助手 —— 终端演示启动器
rem
rem   demo.cmd                 交互模式（真实模型）
rem   demo.cmd --list          只看能问什么，不需要密钥
rem   demo.cmd --fake-chat "…" 完全离线
rem
rem 为什么要有这个文件：从终端手敲的那串命令
rem     .venv\Scripts\python.exe -m maixrag demo
rem 在 PowerShell / cmd / Windows Terminal 里的写法各不相同，
rem 而且要先 cd 到仓库根目录。**演示程序的第一步不该是排环境问题。**
rem ---------------------------------------------------------------------------

rem 切到 UTF-8 代码页，否则中文横幅会变乱码
chcp 65001 >nul

rem 切到本脚本所在目录（%~dp0 自带结尾的反斜杠）
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" -m maixrag demo %*
