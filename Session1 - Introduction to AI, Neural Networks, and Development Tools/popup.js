document.addEventListener('DOMContentLoaded', function() {
    const steps = ['greeting', 'website-type', 'user-details', 'customization', 'preview', 'publish'];
    let currentStep = 0;

    function showStep(stepIndex) {
        steps.forEach((step, index) => {
            document.getElementById(step).classList.toggle('hidden', index !== stepIndex);
        });
        updateNavigationButtons();
    }

    function updateNavigationButtons() {
        const backBtn = document.getElementById('back-btn');
        const nextBtn = document.getElementById('next-btn');
        
        backBtn.style.display = currentStep > 0 ? 'inline-block' : 'none';
        nextBtn.style.display = currentStep < steps.length - 1 ? 'inline-block' : 'none';
        
        if (currentStep === steps.length - 1) {
            document.getElementById('publish').style.display = 'block';
        } else {
            document.getElementById('publish').style.display = 'none';
        }
    }

    document.getElementById('start-btn').addEventListener('click', () => {
        currentStep++;
        showStep(currentStep);
    });

    document.querySelectorAll('.type-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const websiteType = this.dataset.type;
            chrome.storage.local.set({websiteType: websiteType}, () => {
                currentStep++;
                showStep(currentStep);
                showDetailsForm(websiteType);
            });
        });
    });

    function showDetailsForm(websiteType) {
        document.getElementById('personal-details').classList.toggle('hidden', websiteType !== 'personal');
        document.getElementById('professional-details').classList.toggle('hidden', websiteType !== 'professional');
    }

    document.getElementById('details-next-btn').addEventListener('click', () => {
        const websiteType = document.getElementById('personal-details').classList.contains('hidden') ? 'professional' : 'personal';
        const form = document.getElementById(`${websiteType}-form`);
        
        if (!form.checkValidity()) {
            alert('Please fill in all required fields.');
            return;
        }

        const userDetails = Object.fromEntries(new FormData(form));
        chrome.storage.local.set({userDetails: userDetails}, () => {
            currentStep++;
            showStep(currentStep);
        });
    });

    document.getElementById('generate-btn').addEventListener('click', () => {
        const form = document.getElementById('customization-form');
        const customization = Object.fromEntries(new FormData(form));
        chrome.storage.local.set({customization: customization}, () => {
            currentStep++;
            showStep(currentStep);
            generateWebsite();
        });
    });

    // Add event listeners for back and next buttons
    document.getElementById('back-btn').addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            showStep(currentStep);
        }
    });

    document.getElementById('next-btn').addEventListener('click', () => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            showStep(currentStep);
            if (steps[currentStep] === 'preview') {
                generateWebsite();
            }
        }
    });

    function generateWebsite() {
        chrome.storage.local.get(['websiteType', 'userDetails', 'customization'], (data) => {
            const previewFrame = document.getElementById('preview-frame');
            const previewDoc = previewFrame.contentDocument;
            
            const htmlContent = generateHTML(data);
            const cssContent = generateCSS(data.customization);
            
            previewDoc.open();
            previewDoc.write(htmlContent);
            previewDoc.close();
            
            const style = previewDoc.createElement('style');
            style.textContent = cssContent;
            previewDoc.head.appendChild(style);

            // Store generated content for deployment
            chrome.storage.local.set({
                generatedHTML: htmlContent,
                generatedCSS: cssContent
            }, () => {
                // Show the preview step
                currentStep = steps.indexOf('preview');
                showStep(currentStep);
                
                // Add a confirmation button to proceed to the publish step
                const confirmBtn = document.createElement('button');
                confirmBtn.textContent = 'Confirm and Proceed';
                confirmBtn.addEventListener('click', () => {
                    currentStep++;
                    showStep(currentStep);
                });
                document.getElementById('preview').appendChild(confirmBtn);
            });
        });
    }

    function generateHTML(data) {
        const { websiteType, userDetails, customization } = data;
        let content = '';

        if (websiteType === 'personal') {
            content = `
                <header>
                    <h1>${userDetails.name}</h1>
                </header>
                <main>
                    <section id="about">
                        <h2>About Me</h2>
                        <p>${userDetails.bio}</p>
                    </section>
                    <section id="contact">
                        <h2>Contact</h2>
                        <p>Email: ${userDetails.contact_email}</p>
                        ${userDetails.contact_phone ? `<p>Phone: ${userDetails.contact_phone}</p>` : ''}
                    </section>
                    <section id="social">
                        <h2>Social Media</h2>
                        <ul>
                            ${userDetails.facebook_link ? `<li><a href="${userDetails.facebook_link}" target="_blank">Facebook</a></li>` : ''}
                            ${userDetails.instagram_link ? `<li><a href="${userDetails.instagram_link}" target="_blank">Instagram</a></li>` : ''}
                            ${userDetails.twitter_link ? `<li><a href="${userDetails.twitter_link}" target="_blank">Twitter</a></li>` : ''}
                            ${userDetails.linkedin_link ? `<li><a href="${userDetails.linkedin_link}" target="_blank">LinkedIn</a></li>` : ''}
                        </ul>
                    </section>
                </main>
            `;
        } else {
            content = `
                <header>
                    <h1>${userDetails.full_name}</h1>
                    <p>${userDetails.years_of_experience} years of experience</p>
                </header>
                <main>
                    <section id="skills">
                        <h2>Skills</h2>
                        <ul>
                            ${userDetails.skills.split(',').map(skill => `<li>${skill.trim()}</li>`).join('')}
                        </ul>
                    </section>
                    ${userDetails.interested_in ? `
                    <section id="interests">
                        <h2>Interested In</h2>
                        <p>${userDetails.interested_in}</p>
                    </section>
                    ` : ''}
                    ${userDetails.companies_worked_for ? `
                    <section id="experience">
                        <h2>Companies Worked For</h2>
                        <ul>
                            ${userDetails.companies_worked_for.split('\n').map(company => `<li>${company.trim()}</li>`).join('')}
                        </ul>
                    </section>
                    ` : ''}
                    <section id="contact">
                        <h2>Contact</h2>
                        <p>Email: ${userDetails.contact_email}</p>
                        ${userDetails.contact_phone ? `<p>Phone: ${userDetails.contact_phone}</p>` : ''}
                    </section>
                    <section id="profiles">
                        <h2>Professional Profiles</h2>
                        <ul>
                            ${userDetails.linkedin_profile ? `<li><a href="${userDetails.linkedin_profile}" target="_blank">LinkedIn</a></li>` : ''}
                            ${userDetails.github_profile ? `<li><a href="${userDetails.github_profile}" target="_blank">GitHub</a></li>` : ''}
                        </ul>
                    </section>
                </main>
            `;
        }

        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>${websiteType === 'personal' ? userDetails.name : userDetails.full_name} - ${websiteType.charAt(0).toUpperCase() + websiteType.slice(1)} Website</title>
            </head>
            <body>
                ${content}
            </body>
            </html>
        `;
    }

    function generateCSS(customization) {
        let bgColor, textColor, accentColor;

        if (customization.color_change === 'yes') {
            switch(customization.color_palette) {
                case 'blue_&_white':
                    bgColor = '#FFFFFF';
                    textColor = '#000080';
                    accentColor = '#4169E1';
                    break;
                case 'black_&_gold':
                    bgColor = '#000000';
                    textColor = '#FFD700';
                    accentColor = '#DAA520';
                    break;
                case 'green_&_gray':
                    bgColor = '#E0E0E0';
                    textColor = '#006400';
                    accentColor = '#2E8B57';
                    break;
                case 'purple_&_beige':
                    bgColor = '#F5F5DC';
                    textColor = '#4B0082';
                    accentColor = '#8A2BE2';
                    break;
                case 'red_&_white':
                    bgColor = '#FFFFFF';
                    textColor = '#8B0000';
                    accentColor = '#FF0000';
                    break;
                default:
                    bgColor = '#FFFFFF';
                    textColor = '#000000';
                    accentColor = customization.custom_accent_color || '#4CAF50';
            }
        } else {
            bgColor = '#FFFFFF';
            textColor = '#000000';
            accentColor = '#4CAF50';
        }

        return `
            body {
                font-family: ${customization.font === 'serif' ? 'Georgia, serif' : customization.font === 'monospace' ? 'monospace' : 'Arial, sans-serif'};
                color: ${textColor};
                background-color: ${bgColor};
                line-height: 1.6;
                margin: 0;
                padding: 0;
            }
            header {
                background-color: ${accentColor};
                color: ${bgColor};
                text-align: center;
                padding: 1rem;
            }
            main {
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
            }
            h1, h2 {
                color: ${accentColor};
            }
            a {
                color: ${accentColor};
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            ${customization.animation === 'on' ? `
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            body * {
                animation: fadeIn 1s ease-in-out;
            }
            ` : ''}
        `;
    }

    document.getElementById('download-btn').addEventListener('click', () => {
        chrome.storage.local.get(['websiteType', 'userDetails', 'generatedHTML', 'generatedCSS'], (data) => {
            const zip = new JSZip();
            zip.file("index.html", data.generatedHTML);
            zip.file("styles.css", data.generatedCSS);
            
            zip.generateAsync({type:"blob"}).then(function(content) {
                const fileName = `${data.websiteType}_website_${data.userDetails.name || data.userDetails.full_name}.zip`.toLowerCase().replace(/\s+/g, '_');
                
                chrome.downloads.download({
                    url: URL.createObjectURL(content),
                    filename: fileName,
                    saveAs: true
                }, (downloadId) => {
                    if (chrome.runtime.lastError) {
                        console.error(chrome.runtime.lastError);
                        alert('Download failed. Please try again.');
                    } else {
                        alert('Download started successfully!');
                    }
                });
            }).catch(function(error) {
                console.error('Error generating zip:', error);
                alert('Failed to generate the website files. Please try again.');
            });
        });
    });

    document.getElementById('publish-btn').addEventListener('click', () => {
        chrome.storage.local.get(['generatedHTML', 'generatedCSS', 'userDetails'], (data) => {
            chrome.runtime.sendMessage({
                action: 'publishToGitHub',
                html: data.generatedHTML,
                css: data.generatedCSS,
                userDetails: data.userDetails
            }, (response) => {
                if (response.success) {
                    alert(`Your website has been published! You can view it at: ${response.url}`);
                } else {
                    alert(`Error: ${response.error}`);
                }
            });
        });
    });

    // Initialize navigation buttons
    updateNavigationButtons();
});
