from flask import Flask, render_template, jsonify
import json, os

app = Flask(__name__, static_folder='views', static_url_path='/views')

def load(filename):
    with open(os.path.join('data', filename)) as f:
        return json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/epochs')
def epochs():
    return jsonify(load('epochs.json'))

@app.route('/layers')
def layers():
    return jsonify(load('layers.json'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
