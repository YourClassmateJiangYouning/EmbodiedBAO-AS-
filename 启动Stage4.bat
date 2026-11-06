@echo off
chcp 936 >nul
title Stage 4 闭环实验操作台

rem 一键启动 Stage 4 操作台。
rem 用 %~dp0 定位本文件所在目录，所以从任何位置双击都能用。
rem 仓库路径含括号，因此所有路径都加引号。
rem 本文件必须是 GBK 编码：cmd 按系统本地编码解析 .bat，用 UTF-8 会把中文行拆坏。

cd /d "%~dp0"

echo ============================================================
echo   Stage 4 闭环实验操作台
echo ============================================================
echo.

set "PYCMD="
where python >nul 2>nul
if %errorlevel%==0 set "PYCMD=python"
if not defined PYCMD (
    where py >nul 2>nul
    if %errorlevel%==0 set "PYCMD=py -3"
)
if not defined PYCMD goto NOPY

echo 使用命令: %PYCMD%
echo 启动后浏览器会自动打开，默认地址 http://127.0.0.1:8765/
echo.
echo === 实验期间请勿关闭本窗口，关掉窗口程序就停止 ===
echo.

%PYCMD% "stage4\run.py" %*
goto END

:NOPY
echo [错误] 没有找到 Python。
echo        请先安装 Python 3.9 或更高版本，安装时勾选 Add python.exe to PATH。
echo.

:END
echo.
echo 程序已退出，可以关闭本窗口。
pause
