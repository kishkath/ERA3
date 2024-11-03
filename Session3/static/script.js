document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("submit").addEventListener("click", function(event) {
        event.preventDefault(); // Prevent page reload

        const fileInput = document.getElementById("input-file");
        const fileContentDiv = document.getElementById("file-content");
        const tokenizedContentDiv = document.getElementById("tokenized-content");

        // Grab span elements for showing/hiding header text
        const mediaContentHeader = document.getElementById("media-content-header");
        const preprocessedContentHeader = document.getElementById("preprocessed-content-header");

        // Initially hide headers and clear previous content
        mediaContentHeader.classList.add("hidden");
        preprocessedContentHeader.classList.add("hidden");
        fileContentDiv.textContent = "";
        tokenizedContentDiv.textContent = "";

        // Check if a file is uploaded
        if (fileInput.files.length === 0) {
            fileContentDiv.textContent = "Please, upload a file.";
            return;
        }

        const file = fileInput.files[0];
        const reader = new FileReader();

        // Display file content and send to API on successful read
        reader.onload = function(event) {
            const fileContent = event.target.result;
            fileContentDiv.textContent = fileContent;
            mediaContentHeader.classList.remove("hidden"); // Show media content header

            // Send file content to backend for tokenization
            fetch("http://127.0.0.1:1702/media/processing", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: fileContent })
            })
            .then(response => response.json())
            .then(data => {
                // Check if 'tokens' is in the response and is an array
                if (data.tokens && Array.isArray(data.tokens)) {
                    tokenizedContentDiv.textContent = "Tokenized Content: " + data.tokens.join(", ");
                    preprocessedContentHeader.classList.remove("hidden"); // Show preprocessed content header
                } else {
                    tokenizedContentDiv.textContent = "Error: " + (data.error || "Tokenization failed.");
                }
            })
            .catch(error => {
                tokenizedContentDiv.textContent = "Error: Could not fetch tokenized content.";
                console.error("Error:", error);
            });
        };

        // Handle errors
        reader.onerror = function() {
            fileContentDiv.textContent = "Could not read the file.";
        };

        // Read the file as plain text
        reader.readAsText(file);
    });
});
