let chart;
const epochs = [];
const losses = [];

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                pointRadius: 0,  // Hide individual points
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Epoch',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            // Show floating point epoch numbers
                            return value.toFixed(1);
                        },
                        stepSize: 0.5,  // Show ticks every 0.5 epochs
                        maxRotation: 0,  // Keep labels horizontal
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        font: {
                            size: 12
                        }
                    }
                }
            },
            animation: {
                duration: 0
            },
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 14
                        }
                    }
                }
            }
        }
    });
}

function showStatus(message, isSuccess = true) {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.textContent = message;
    statusDiv.style.display = 'block';
    statusDiv.className = `status-message ${isSuccess ? 'success' : 'error'}`;
}

// Initialize Socket.IO connection
const socket = io();

socket.on('dataset_status', function(data) {
    showStatus(data.message, data.status !== 'error');
    if (data.status === 'completed') {
        document.getElementById('startTraining').disabled = false;
        document.getElementById('downloadDataset').disabled = true;
    }
});

// Add a function to update logs in UI
function updateLogs(message) {
    const logsDiv = document.getElementById('trainingLogs');
    if (!logsDiv) {
        console.error('Training logs container not found');
        return;
    }
    
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    
    // Format timestamp
    const timestamp = new Date().toLocaleTimeString();
    
    // Format the message with timestamp
    logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> <span class="log-msg">${message}</span>`;
    
    // Keep only last 50 log entries for better readability
    if (logsDiv.children.length > 50) {
        logsDiv.removeChild(logsDiv.firstChild);
    }
    
    logsDiv.appendChild(logEntry);
    logsDiv.scrollTop = logsDiv.scrollHeight;
}

socket.on('training_update', function(data) {
    const progressDiv = document.getElementById('progress');
    
    if (data.status === 'progress' && progressDiv) {
        // Update progress information
        progressDiv.textContent = 
            `Training Progress: Epoch ${data.epoch}/${data.total_epochs} - ` +
            `Batch ${data.batch}/${data.total_batches} - ` +
            `Loss: ${data.loss.toFixed(4)}`;
        
        // Calculate x-axis position (epoch + progress)
        const xPos = data.epoch + data.epoch_progress;
        
        // Update chart with real-time loss data
        chart.data.labels.push(xPos);
        chart.data.datasets[0].data.push(data.loss);
        
        // Keep only last 100 points visible
        if (chart.data.labels.length > 100) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        // Update x-axis max if needed
        chart.options.scales.x.max = Math.max(data.total_epochs, ...chart.data.labels) + 0.2;
        
        chart.update('none');
    } 
    else if (data.status === 'log') {
        // Handle log messages
        updateLogs(data.message);
    }
    else if (data.status === 'completed' && progressDiv) {
        progressDiv.textContent = 'Training completed!';
        document.getElementById('startTraining').disabled = false;
        document.getElementById('getPredictions').disabled = false;
    }
    else if (data.status === 'stopped' && progressDiv) {
        progressDiv.textContent = 'Training stopped by user';
        document.getElementById('startTraining').disabled = false;
    }
});

// Download dataset button handler
document.getElementById('downloadDataset').addEventListener('click', async () => {
    try {
        showStatus('Checking dataset status...');
        const response = await fetch('/prepare-dataset', { method: 'POST' });
        const data = await response.json();
        showStatus(data.message, response.ok);
    } catch (error) {
        showStatus('Error preparing dataset: ' + error.message, false);
    }
});

// Start training button handler
document.getElementById('startTraining').addEventListener('click', async () => {
    epochs.length = 0;
    losses.length = 0;
    chart.update();
    
    showStatus('Training started...');
    await fetch('/train', { method: 'POST' });
});

// Get predictions button handler
document.getElementById('getPredictions').addEventListener('click', async () => {
    const response = await fetch('/predict');
    const data = await response.json();
    
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = '';
    
    data.images.forEach((img, index) => {
        const div = document.createElement('div');
        div.className = 'prediction-item';
        
        const imgElement = document.createElement('img');
        imgElement.src = `data:image/png;base64,${img}`;
        
        const prediction = document.createElement('p');
        prediction.textContent = `Predicted: ${data.predictions[index]}`;
        
        const actual = document.createElement('p');
        actual.textContent = `Actual: ${data.actual_labels[index]}`;
        
        div.appendChild(imgElement);
        div.appendChild(prediction);
        div.appendChild(actual);
        predictionsDiv.appendChild(div);
    });
});

// Initialize chart on page load
initChart(); 