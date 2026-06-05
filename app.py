import os
from flask import Flask, render_template, jsonify, Response
import json
import time
import sys

# Add the current directory to sys.path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iot.corridor_sim_api import run_simulation_generator

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/api/simulation/stream')
def stream_simulation():
    def generate():
        try:
            for data in run_simulation_generator():
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(0.1) # Simulate real-time progress for the UI
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
