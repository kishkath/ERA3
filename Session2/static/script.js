document.addEventListener('DOMContentLoaded', () => {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    const fileInput = document.getElementById('user-image');
    const submitButton = document.getElementById('submit');
    const fileInfoDiv = document.getElementById('file-info');
    const imagePreview = document.getElementById('image-preview');
    const animalInfoDiv = document.getElementById('animal-info');

    const defaultWidth = 350;   // Width in pixels
    const defaultHeight = 250;

    const animalData = {
        cat: {
            info: "Cats are independent and intelligent animals known for their agility and unique personalities. They have been companions to humans for thousands of years, revered for their ability to control pests."
        },
        dog: {
            info: "Dogs are loyal and social animals that have been bred for thousands of years for various purposes, including herding, guarding, and companionship. They are often referred to as man's best friend."
        },
        elephant: {
            info: "Elephants are the largest land animals on Earth, known for their intelligence, social behavior, and strong family bonds. They have long trunks that they use for various tasks, including feeding and social interactions."
        }
    };

    // Function to check if either an image is uploaded or a checkbox is selected
    function checkSubmitButton() {
        const isCheckboxSelected = Array.from(checkboxes).some(checkbox => checkbox.checked);
        const isFileSelected = fileInput.files.length > 0;

        submitButton.disabled = !(isCheckboxSelected || isFileSelected);
    }

    // Event listeners for checkboxes
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function () {
            if (this.checked) {
                checkboxes.forEach(cb => {
                    if (cb !== this) cb.checked = false;
                });
            }
            checkSubmitButton();
        });
    });

    // File input change event listener
    fileInput.addEventListener('change', () => {
        checkboxes.forEach(checkbox => checkbox.checked = false);
        checkSubmitButton();
    });

    // Submit button event listener
    submitButton.addEventListener('click', (event) => {
        event.preventDefault();
        fileInfoDiv.innerHTML = '';
        imagePreview.style.display = 'none';
        animalInfoDiv.innerHTML = '';

        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const reader = new FileReader();

            reader.onload = function (event) {
                imagePreview.src = event.target.result;
                imagePreview.style.display = 'block';
                imagePreview.width = defaultWidth;
                imagePreview.height = defaultHeight;

                fileInfoDiv.innerHTML = `
                    <h3>Uploaded Image Information:</h3>
                    <p><strong>Name:</strong> ${file.name}</p>
                    <p><strong>Size:</strong> ${(file.size / 1024).toFixed(2)} KB</p>
                    <p><strong>Type:</strong> ${file.type}</p>
                `;
            };
            reader.readAsDataURL(file);
        } else {
            let animal = '';
            if (checkboxes[0].checked) { // Cat checkbox
                animal = 'cat';
            } else if (checkboxes[1].checked) { // Dog checkbox
                animal = 'dog';
            } else if (checkboxes[2].checked) { // Elephant checkbox
                animal = 'elephant';
            }

            if (animal) {
                fileInfoDiv.innerHTML = `
                    <h3>Image Information for ${animal.charAt(0).toUpperCase() + animal.slice(1)}:</h3>
                    <p><strong>Name:</strong> ${animal}.png</p>
                    <p><strong>Path:</strong> images/${animal}.png</p>
                    <h3>About ${animal.charAt(0).toUpperCase() + animal.slice(1)}:</h3>
                    <p>${animalData[animal].info}</p>
                `;
                imagePreview.src = `images/${animal}.png`; // Set the image source for preview
                imagePreview.style.display = 'block';
                imagePreview.width = defaultWidth;
                imagePreview.height = defaultHeight;
            }
        }
    });
});
