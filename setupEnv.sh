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

git config --global user.name mw
git config --global user.email 694497013@qq.com

sudo apt update && sudo apt install language-pack-zh-hans language-pack-gnome-zh-hans

export LANG=zh_CN.UTF-8 && export LANGUAGE=zh_CN:zh:en_US:en

apt install libxcb-cursor0  # 安装核心依赖
apt install libxcb-xinerama0 libxcb-randr0 libxcb-xinput0  # 安装其他常见xcb依赖
apt install libxkbcommon-x11-0  # 输入支持库
apt install xdotool

# # 强制终止所有 git 相关进程（卡住时使用）
# pkill -f 'git (merge|rebase|pull|push|commit)' && sleep 1

# # 安装基础环境（Debian/Ubuntu）
# sudo apt install wine-stable winetricks

# # 配置 Wine 前缀（隔离不同软件环境）
# export WINEPREFIX=\~/.wine-projects

# dpkg --add-architecture i386 && apt-get update && apt-get install wine32
# winecfg  # 设置 Windows 版本和驱动参数

# # 安装依赖库（以 .NET Framework 4.8 为例）
# winetricks dotnet48 corefonts

# # 运行 EXE 程序
# wine /path/to/program.exe

# sudo apt update
# sudo apt install libxcb1 libxcb-xinerama0 libxcb-cursor0 libxcb-randr0 libxcb-xinput0 libxkbcommon-x11-0

# wget https://download.qt.io/official_releases/qtcreator/12.0/12.0.2/qt-creator-opensource-linux-x86_64-12.0.2.run
# chmod +x qt-creator-opensource-linux-x86_64-12.0.2.run
# ./qt-creator-opensource-linux-x86_64-12.0.2.run

# ./configure -no-opengl \
#             -no-egl 



