@echo off
echo 正在安装手势控制应用依赖包...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo 检测到Python版本：
python --version

echo.
echo 开始安装依赖包...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo 安装失败，请检查网络连接或手动安装依赖包
    pause
    exit /b 1
)

echo.
echo 安装完成！
echo 运行应用请执行：python main.py
echo.
pause