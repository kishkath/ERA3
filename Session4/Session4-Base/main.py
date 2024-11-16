from flask import Flask, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from train import TrainingManager
from inference import InferenceManager
from dataset import FashionMNISTDataset
import threading

app = Flask(__name__, static_folder='static', static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global training manager
training_manager = None

def broadcast_training_update(data):
    socketio.emit('training_update', data)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/prepare-dataset', methods=['POST'])
def prepare_dataset():
    dataset = FashionMNISTDataset()
    result = dataset.prepare_dataset()
    
    if result['status'] in ['exists', 'downloaded']:
        socketio.emit('dataset_status', {
            'status': 'completed',
            'message': result['message']
        })
        return jsonify({"message": result['message']})
    else:
        socketio.emit('dataset_status', {
            'status': 'error',
            'message': result['message']
        })
        return jsonify({"message": result['message']}), 500

@app.route('/train', methods=['POST'])
def start_training():
    global training_manager
    training_manager = TrainingManager(broadcast_training_update)
    
    def training_task():
        training_manager.train()
    
    thread = threading.Thread(target=training_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Training started"})

@app.route('/predict', methods=['GET'])
def get_predictions():
    inference_manager = InferenceManager()
    return jsonify(inference_manager.get_random_predictions())

if __name__ == '__main__':
    socketio.run(app, debug=True, port=1111)