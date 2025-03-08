const express = require('express');
const httpProxy = require('http-proxy');
const app = express();
const proxy = httpProxy.createProxyServer({});

// 托管静态文件（前端HTML/CSS/JS）
app.use(express.static('public'));

// 解析用户提交的URL（通过GET参数或POST body）
app.get('/proxy1', (req, res) => {
  const targetUrl = req.query.url; // 从GET参数获取目标URL
  if (!targetUrl) {
    return res.status(400).send('Missing URL parameter');
  }

  // 修改请求头以绕过目标网站的反爬机制
  req.headers.host = new URL(targetUrl).host;
  req.headers.referer = targetUrl;
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST');

    // 伪造浏览器请求头
  const fakeHeaders = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.google.com/',
    };

        // 合并自定义头到原始请求头
  Object.assign(req.headers, fakeHeaders);

  console.log("目标链接：", targetUrl);
  req.url = req.query.url;
//   // 转发请求到目标网站
  proxy.web(req, res, { 
    target: targetUrl,
    followRedirects: true, // 允许重定向
    changeOrigin: true     // 修改Origin头
  });
});

// 处理代理错误
proxy.on('error', (err, req, res) => {
  res.status(500).send('Proxy Error: ' + err.message);
});

// 启动服务
app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});
