@echo off
chcp 65001 >nul
echo 正在为本项目安装依赖环境...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo 所有依赖安装完成！
echo 如遇权限或网络问题，请用管理员身份运行本文件，或手动执行 pip install -r requirements.txt
pause 