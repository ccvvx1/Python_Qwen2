# 安装 Chrome 浏览器
apt-get update
apt-get install -y wget xvfb x11vnc fluxbox
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
dpkg -i google-chrome-stable_current_amd64.deb || apt-get install -y -f
apt-get install -y net-tools

curl ifconfig.me          # 直接返回 IP
curl icanhazip.com        # 备用方案
curl ipinfo.io/ip         # 结构化输出
curl api.ipify.org        # 纯文本响应
curl -4/-6 ident.me       # 指定 IPv4 或 IPv6

