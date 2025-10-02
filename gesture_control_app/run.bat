@echo off
echo 启动手势控制应用...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

REM 运行应用
python main.py

pause