from flask import Flask, Response
import requests

app = Flask(__name__)

@app.route('/proxy/baidu')
def proxy_baidu():
    response = requests.get('https://www.google.com')
    modified_html = response.text.replace('target="_blank"', 'target="_self"')
    return Response(modified_html, mimetype='text/html')

if __name__ == '__main__':
    app.run(port=5000)
