# 启动 Xvfb 虚拟显示器（分辨率 1280x720）
import subprocess
import os

# display = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1280x720x24"])
os.environ['DISPLAY'] = ':1'

# 在虚拟显示器中运行 Chrome
# subprocess.Popen(["google-chrome", "--no-sandbox", "--disable-gpu"])



# # 启动 x11vnc 服务（端口 5900）
# subprocess.Popen(["x11vnc", "-display", ":1", "-forever", "-shared", "-noxdamage"])




# Xvfb :99 -screen 0 1024x768x16 &> /dev/null &
# x11vnc -display :99 -forever -shared -noxdamage

# curl -L https://www.cpolar.com/static/downloads/install-release-cpolar.sh | sudo bash
# cpolar tcp 5900
# cpolar authtoken ZDg0ZjVkZWEtYTMzZS00MTY2LTk2MmYtMGY4Njk4ZTg4ODA

# google-chrome --no-sandbox --display=:99 --window-size=1280,1024
