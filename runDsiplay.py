# 启动 Xvfb 虚拟显示器（分辨率 1280x720）
import subprocess
import os

# display = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1280x720x24"])
os.environ['DISPLAY'] = ':1'

# 在虚拟显示器中运行 Chrome
subprocess.Popen(["google-chrome", "--no-sandbox", "--disable-gpu"])


# # 启动 x11vnc 服务（端口 5900）
# subprocess.Popen(["x11vnc", "-display", ":1", "-forever", "-shared", "-noxdamage"])
