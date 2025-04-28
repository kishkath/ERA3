document.addEventListener('DOMContentLoaded', function() {
    console.log('[Popup] Extension popup loaded');
    
    const resumeInput = document.getElementById('resumeInput');
    const uploadButton = document.getElementById('uploadButton');
    const uploadArea = document.getElementById('uploadArea');
    const resultSection = document.getElementById('resultSection');
    const scoreDisplay = document.getElementById('scoreDisplay');
    const feedbackDisplay = document.getElementById('feedbackDisplay');

    // Handle file selection button click
    uploadButton.addEventListener('click', function() {
        console.log('[Popup] Upload button clicked');
        resumeInput.click();
    });

    // Handle file selection
    resumeInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        console.log('[Popup] File selected:', file ? file.name : 'No file');
        
        if (!file) {
            console.warn('[Popup] No file selected');
            return;
        }

        // Validate file type
        if (!file.type.match(/application\/(pdf|msword|vnd\.openxmlformats-officedocument\.wordprocessingml\.document)/)) {
            console.error('[Popup] Invalid file type:', file.type);
            uploadArea.innerHTML = `
                <p style="color: red;">Error: Please upload a PDF or Word document</p>
                <button id="uploadButton">Try Again</button>
            `;
            document.getElementById('uploadButton').addEventListener('click', function() {
                resumeInput.click();
            });
            return;
        }

        // Show loading state
        console.log('[Popup] Starting file processing...');
        uploadArea.innerHTML = '<p>Processing resume...</p>';

        try {
            console.log('[Popup] Sending file to background script:', {
                name: file.name,
                type: file.type,
                size: file.size
            });
            
            // Send to background script for processing
            const response = await chrome.runtime.sendMessage({
                action: 'evaluateResume',
                file: {
                    name: file.name,
                    type: file.type,
                    size: file.size
                }
            });

            console.log('[Popup] Received response:', response);

            // Validate response
            if (!response || typeof response.score === 'undefined' || !response.feedback) {
                console.error('[Popup] Invalid response format:', response);
                throw new Error('Invalid response from evaluation service');
            }

            // Display results
            console.log('[Popup] Displaying results - Score:', response.score);
            resultSection.style.display = 'block';
            scoreDisplay.textContent = `✅ Match Score: ${response.score}/100`;
            feedbackDisplay.textContent = `📝 Feedback: ${response.feedback}`;

            // Update upload area
            console.log('[Popup] Updating UI for next upload');
            uploadArea.innerHTML = `
                <p>Upload another resume?</p>
                <button id="uploadButton">Choose File</button>
            `;

            // Reattach event listener to the new button
            document.getElementById('uploadButton').addEventListener('click', function() {
                resumeInput.click();
            });

        } catch (error) {
            console.error('[Popup] Error during processing:', error);
            uploadArea.innerHTML = `
                <p style="color: red;">Error: ${error.message || 'An unknown error occurred'}</p>
                <button id="uploadButton">Try Again</button>
            `;

            // Reattach event listener to the new button
            document.getElementById('uploadButton').addEventListener('click', function() {
                resumeInput.click();
            });
        }
    });
}); 