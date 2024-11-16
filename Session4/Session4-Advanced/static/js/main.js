let charts = {
    accuracy: null,
    loss: null
};

function initializeCharts() {
    const chartConfig = {
        type: 'line',
        options: {
            responsive: true,
            animation: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    };

    charts.accuracy = new Chart(
        document.getElementById('accuracy-chart').getContext('2d'),
        {
            ...chartConfig,
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Model 1 Accuracy',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: 'Model 2 Accuracy',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }
                ]
            }
        }
    );

    charts.loss = new Chart(
        document.getElementById('loss-chart').getContext('2d'),
        {
            ...chartConfig,
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Model 1 Loss',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: 'Model 2 Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }
                ]
            }
        }
    );
}

function updateChannelProgression(modelNum) {
    const startChannels = parseInt(document.getElementById(`channels${modelNum}`).value);
    const progression = `${startChannels} → ${startChannels*2} → ${startChannels*4}`;
    document.getElementById(`progression${modelNum}`).textContent = progression;
}

function updateCharts(modelNum, data) {
    const datasetIndex = modelNum - 1;
    
    if (data.epoch !== undefined) {
        if (!charts.accuracy.data.labels.includes(data.epoch)) {
            charts.accuracy.data.labels.push(data.epoch);
            charts.loss.data.labels.push(data.epoch);
        }
        
        charts.accuracy.data.datasets[datasetIndex].data.push(data.accuracy);
        charts.loss.data.datasets[datasetIndex].data.push(data.loss);
        
        charts.accuracy.update();
        charts.loss.update();
    }
}

async function trainModel(modelNum) {
    const trainButton = document.getElementById(`train-model${modelNum}`);
    const predictButton = document.getElementById(`predict-model${modelNum}`);
    trainButton.disabled = true;
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_id: modelNum,
                start_channels: document.getElementById(`channels${modelNum}`).value,
                epochs: parseInt(document.getElementById(`epochs${modelNum}`).value),
                optimizer: document.getElementById(`optimizer${modelNum}`).value,
                batch_size: parseInt(document.getElementById(`batch-size${modelNum}`).value)
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            
            const events = decoder.decode(value).split('\n\n');
            events.forEach(event => {
                if (event.startsWith('data: ')) {
                    const data = JSON.parse(event.slice(6));
                    updateCharts(modelNum, data);
                }
            });
        }

        predictButton.disabled = false;
    } catch (error) {
        console.error('Training error:', error);
    }
    
    trainButton.disabled = false;
}

async function getPredictions(modelNum) {
    const button = document.getElementById(`predict-model${modelNum}`);
    button.disabled = true;
    button.textContent = 'Getting Predictions...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_id: modelNum,
                num_predictions: parseInt(document.getElementById(`predictions${modelNum}`).value)
            })
        });

        const predictions = await response.json();
        displayPredictions(modelNum, predictions);
    } catch (error) {
        console.error('Prediction error:', error);
    }

    button.disabled = false;
    button.textContent = 'Get Predictions';
}

function displayPredictions(modelNum, predictions) {
    if (predictions.error) {
        console.error(`Model ${modelNum} prediction error:`, predictions.error);
        return;
    }

    const displayDiv = document.getElementById(`predictions-display${modelNum}`);
    displayDiv.style.display = 'grid';
    displayDiv.innerHTML = '';

    predictions.images.forEach((image, i) => {
        const isCorrect = predictions.predictions[i] === predictions.actual[i];
        const item = document.createElement('div');
        item.className = `prediction-item ${isCorrect ? 'correct' : 'incorrect'}`;

        item.innerHTML = `
            <img src="data:image/png;base64,${image}" alt="Prediction ${i}">
            <div class="prediction-label">Predicted: ${predictions.predictions[i]}</div>
            <div class="prediction-actual">Actual: ${predictions.actual[i]}</div>
        `;

        displayDiv.appendChild(item);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    
    // Add channel progression update listeners
    document.getElementById('channels1').addEventListener('change', () => updateChannelProgression(1));
    document.getElementById('channels2').addEventListener('change', () => updateChannelProgression(2));
    
    // Initialize channel progressions
    updateChannelProgression(1);
    updateChannelProgression(2);
    
    // Add training listeners
    document.getElementById('train-model1').addEventListener('click', () => trainModel(1));
    document.getElementById('train-model2').addEventListener('click', () => trainModel(2));
    
    // Add prediction listeners
    document.getElementById('predict-model1').addEventListener('click', () => getPredictions(1));
    document.getElementById('predict-model2').addEventListener('click', () => getPredictions(2));
}); 