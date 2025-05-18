// Initialize date pickers
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Flatpickr date pickers
    flatpickr("#startDate", {
        dateFormat: "Y-m-d",
        maxDate: "today"
    });
    
    flatpickr("#endDate", {
        dateFormat: "Y-m-d",
        maxDate: "today"
    });
    
    // Set current year for year inputs
    const currentYear = new Date().getFullYear();
    document.getElementById('monthYear').value = currentYear;
    document.getElementById('year').value = currentYear;
    
    // Initialize event listeners
    initializeEventListeners();
});

function initializeEventListeners() {
    // Handle date range type changes
    document.getElementById('dateRangeType').addEventListener('change', handleDateRangeTypeChange);
    
    // Handle form submission
    document.getElementById('searchForm').addEventListener('submit', handleFormSubmit);
}

function handleDateRangeTypeChange(e) {
    const type = e.target.value;
    document.getElementById('customDateRange').classList.toggle('hidden', type !== 'custom');
    document.getElementById('monthSelection').classList.toggle('hidden', type !== 'month');
    document.getElementById('yearSelection').classList.toggle('hidden', type !== 'year');
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const dateRange = getDateRangeData();
    const query = document.getElementById('searchQuery').value;
    
    // Show loading spinner
    showLoading(true);
    
    try {
        const response = await performSearch(query, dateRange);
        displayResults(response);
    } catch (error) {
        console.error('Error:', error);
        displayError();
    } finally {
        showLoading(false);
    }
}

function getDateRangeData() {
    const dateRangeType = document.getElementById('dateRangeType').value;
    let dateRange = { type: dateRangeType };
    
    switch (dateRangeType) {
        case 'custom':
            dateRange.start = document.getElementById('startDate').value;
            dateRange.end = document.getElementById('endDate').value;
            break;
        case 'month':
            dateRange.month = document.getElementById('month').value;
            dateRange.year = document.getElementById('monthYear').value;
            break;
        case 'year':
            dateRange.year = document.getElementById('year').value;
            break;
    }
    
    return dateRange;
}

async function performSearch(query, dateRange) {
    const response = await fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query,
            dateRange
        })
    });
    
    return await response.json();
}

function displayResults(data) {
    const resultsContainer = document.getElementById('results');
    
    if (data.message) {
        resultsContainer.innerHTML = createMessageElement(data.message);
    } else if (data.results) {
        resultsContainer.innerHTML = data.results.map(email => createResultCard(email)).join('');
    }
}

function createMessageElement(message) {
    return `
        <div class="result-card">
            <p class="text-gray">${message}</p>
        </div>
    `;
}

function createResultCard(email) {
    return `
        <div class="result-card">
            <h3 class="result-title">${email.subject}</h3>
            <p class="result-meta">From: ${email.from}</p>
            <p class="result-meta">Date: ${email.date}</p>
            <p class="result-summary">${email.summary}</p>
        </div>
    `;
}

function displayError() {
    document.getElementById('results').innerHTML = `
        <div class="result-card">
            <p class="text-red">An error occurred while searching. Please try again.</p>
        </div>
    `;
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
    if (show) {
        document.getElementById('results').innerHTML = '';
    }
} 