# 启动 Xvfb 虚拟显示器（分辨率 1280x720）
import subprocess
import os

# display = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1280x720x24"])
os.environ['DISPLAY'] = ':1'

# 在虚拟显示器中运行 Chrome
# subprocess.Popen(["google-chrome", "--no-sandbox", "--disable-gpu"])



# # 启动 x11vnc 服务（端口 5900）
# subprocess.Popen(["x11vnc", "-display", ":1", "-forever", "-shared", "-noxdamage"])




# Xvfb :99 -screen 0 1924x1068x16+60 &> /dev/null &
# x11vnc -display :99 -forever -shared -noxdamage

# curl -L https://www.cpolar.com/static/downloads/install-release-cpolar.sh | sudo bash
# cpolar tcp 5900
# cpolar authtoken ZDg0ZjVkZWEtYTMzZS00MTY2LTk2MmYtMGY4Njk4ZTg4ODA

# export LANG=zh_CN.UTF-8
# export LANGUAGE=zh_CN:zh:en_US:en


# export LANG=zh_CN.UTF-8 && export LANGUAGE=zh_CN:zh:en_US:en && google-chrome --no-sandbox --display=:99 --window-size=1280,1024
# nginx -c /content/jsproxy/nginx.conf -p /content/jsproxy/nginx -s reload

# 下载最新版本（以 v0.34 为例）
# wget https://github.com/openresty/headers-more-nginx-module/archive/refs/tags/v0.34.tar.gz
# tar -xzvf v0.34.tar.gz


# git clone https://github.com/openresty/lua-nginx-module.git
# git clone https://github.com/openresty/luajit2.git
# cd luajit2
# make && sudo make install

# 查看当前 Nginx 版本和编译参数
# nginx -V 2>&1 | grep arguments

# 示例输出：
# configure arguments: --prefix=/etc/nginx --sbin-path=/usr/sbin/nginx ...

# 下载对应版本的 Nginx 源码
# wget http://nginx.org/download/nginx-1.27.4.tar.gz
# tar -xzvf nginx-1.27.4.tar.gz
# cd nginx-1.27.4

    # --add-module=/content/jsproxy/lua-nginx-module \

# # 重新配置并添加模块
# ./configure \
#     --prefix=/etc/nginx \
#     --sbin-path=/usr/sbin/nginx \
#     --add-module=/content/jsproxy/headers-more-nginx-module-0.34 \
#     --with-cc-opt='-g -O2 -ffile-prefix-map=/build/nginx-niToSo/nginx-1.18.0=. -flto=auto -ffat-lto-objects -flto=auto -ffat-lto-objects -fstack-protector-strong -Wformat -Werror=format-security -fPIC -Wdate-time -D_FORTIFY_SOURCE=2' --with-ld-opt='-Wl,-Bsymbolic-functions -flto=auto -ffat-lto-objects -flto=auto -Wl,-z,relro -Wl,-z,now -fPIC' --prefix=/usr/share/nginx --conf-path=/etc/nginx/nginx.conf --http-log-path=/var/log/nginx/access.log --error-log-path=/var/log/nginx/error.log --lock-path=/var/lock/nginx.lock --pid-path=/run/nginx.pid --modules-path=/usr/lib/nginx/modules --http-client-body-temp-path=/var/lib/nginx/body --http-fastcgi-temp-path=/var/lib/nginx/fastcgi --http-proxy-temp-path=/var/lib/nginx/proxy --http-scgi-temp-path=/var/lib/nginx/scgi --http-uwsgi-temp-path=/var/lib/nginx/uwsgi --with-compat --with-debug --with-pcre-jit --with-http_ssl_module --with-http_stub_status_module --with-http_realip_module --with-http_auth_request_module --with-http_v2_module --with-http_dav_module --with-http_slice_module --with-threads  --with-http_addition_module --with-http_gunzip_module --with-http_gzip_static_module --with-http_sub_module
# 修改ngx_feature_incs="#include <luajit-2.1/luajit.h>"
# # 编译并安装
# make
# sudo make install

# lsof -i :80
# https://raw.githubusercontent.com/EtherDream/jsproxy-bin/master/Linux-x86_64/openresty-1.15.8.1.tar.gz
# /home/jsproxy/openresty/nginx/sbin/nginx -c /home/jsproxy/server/nginx.conf -p /home/jsproxy/server/nginx