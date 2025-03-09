package constant

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"net/http"
	"os"
	"os/user"
	"path"
	// "os/exec"
	"strings"

	log "github.com/sirupsen/logrus"
	"gopkg.in/ini.v1"
)

const (
	Name             = "clash"
	DefalutHTTPPort  = "7890"
	DefalutSOCKSPort = "7891"
)

var (
	HomeDir    string
	ConfigPath string
	MMDBPath   string
)

func init() {
	currentUser, err := user.Current()
	if err != nil {
		dir := os.Getenv("HOME")
		if dir == "" {
			log.Fatalf("Can't get current user: %s", err.Error())
		}
		HomeDir = dir
	} else {
		HomeDir = currentUser.HomeDir
	}

	dirPath := path.Join(HomeDir, ".config", Name)
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		if err := os.MkdirAll(dirPath, 0777); err != nil {
			log.Fatalf("Can't create config directory %s: %s", dirPath, err.Error())
		}
	}

	ConfigPath = "/content/Python_Qwen2/clash/config.ini"
	// ConfigPath = path.Join(dirPath, "config.ini")
	// if _, err := os.Stat(ConfigPath); os.IsNotExist(err) {
	// 	log.Info("Can't find config, create a empty file")
	// 	os.OpenFile(ConfigPath, os.O_CREATE|os.O_WRONLY, 0644)
	// }

	// MMDBPath = path.Join(dirPath, "Country.mmdb")
	MMDBPath = "/content/Python_Qwen2/clash/GeoLite2-Country.mmdb"
	// if _, err := os.Stat(MMDBPath); os.IsNotExist(err) {
	// 	log.Info("Can't find MMDB, start download")
	// 	cmd := exec.Command("cp", "/content/Python_Qwen2/clash/GeoLite2-Country.mmdb", MMDBPath)
	// 	out, err := cmd.CombinedOutput()
    //    if err != nil {
	// 	   log.Info("Error executing command:", err)
    //        return
    //    }
    //    log.Info(string(out))
	// 	// err := downloadMMDB(MMDBPath)
	// 	// if err != nil {
	// 	// 	log.Fatalf("Can't download MMDB: %s", err.Error())
	// 	// }
	// }
}

func downloadMMDB(path string) (err error) {
	// resp, err := http.Get("http://geolite.maxmind.com/download/geoip/database/GeoLite2-Country.tar.gz")
	resp, err := http.Get("https://cdn.jsdelivr.net/npm/geolite2-country/GeoLite2-Country.mmdb.gz")
	// resp, err := http.Get("https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key=YOUR_LICENSE_KEY&suffix=tar.gz")
	if err != nil {
		return
	}
	defer resp.Body.Close()

	gr, err := gzip.NewReader(resp.Body)
	if err != nil {
		return
	}
	defer gr.Close()

	tr := tar.NewReader(gr)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}

		if !strings.HasSuffix(h.Name, "GeoLite2-Country.mmdb") {
			continue
		}

		f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = io.Copy(f, tr)
		if err != nil {
			return err
		}
	}

	return nil
}

func GetConfig() (*ini.File, error) {
	if _, err := os.Stat(ConfigPath); os.IsNotExist(err) {
		return nil, err
	}
	return ini.LoadSources(
		ini.LoadOptions{AllowBooleanKeys: true},
		ConfigPath,
	)
}
