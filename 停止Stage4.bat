@echo off
chcp 936 >nul
title 停止 Stage 4 操作台

rem 关闭 Stage 4 的所有实例。
rem 按监听的端口找 PID，再逐个结束，所以不管它是在哪个窗口、由谁启动的都能停掉。
rem 本文件必须保存为 GBK 编码。

echo ============================================================
echo   停止 Stage 4 操作台
echo ============================================================
echo.

setlocal enabledelayedexpansion
set FOUND=0

for %%P in (8765 8766 8767 8768 8769 8770 8771 8772 8773 8774 8775 8776 8777 8778 8779 8780 8781 8782) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr /r /c:":%%P .*LISTENING"') do (
        if not "%%A"=="0" (
            echo 结束端口 %%P 上的进程 PID %%A
            taskkill /F /PID %%A >nul 2>nul
            set FOUND=1
        )
    )
)

if "!FOUND!"=="0" (
    echo 没有发现正在运行的 Stage 4 实例。
) else (
    echo.
    echo 已停止。
)

echo.
pause
