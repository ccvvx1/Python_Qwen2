module clash-prj

go 1.21.6

// go 1.21

require github.com/Dreamacro/clash v1.2.3

require (
	github.com/ajg/form v1.5.1 // indirect
	github.com/eapache/queue v1.1.0 // indirect
	github.com/go-chi/chi v1.5.5 // indirect
	github.com/go-chi/render v1.0.3 // indirect
	github.com/oschwald/geoip2-golang v1.11.0 // indirect
	github.com/oschwald/maxminddb-golang v1.13.0 // indirect
	github.com/riobard/go-shadowsocks2 v0.2.3 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	golang.org/x/crypto v0.0.0-20200128174031-69ecbb4d6d5d // indirect
	golang.org/x/sys v0.20.0 // indirect
	gopkg.in/eapache/channels.v1 v1.1.0 // indirect
	gopkg.in/ini.v1 v1.67.0 // indirect
)

replace github.com/Dreamacro/clash => /content/Python_Qwen2/clash // 本地路径

// 或
//replace github.com/Dreamacro/clash => gitlab.com/your-fork/dependency v1.2.4
