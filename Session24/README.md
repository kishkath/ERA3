 # Resume Evaluator Chrome Extension

An AI-powered Chrome extension that evaluates resumes against job descriptions using Google's Gemini AI. The extension provides instant feedback on how well a resume matches specific job requirements.

YOUTUBE LINK: https://youtu.be/4f2jlC-9eKw?si=guRPhM73C14TzNJ3

## Features

- **File Upload**: Supports PDF and Word document formats
- **AI-Powered Analysis**: Uses Gemini AI for intelligent resume evaluation
- **Detailed Feedback**: Provides match score and actionable suggestions
- **Real-time Processing**: Quick evaluation with immediate results
- **Comprehensive Logging**: Detailed operation tracking for debugging

## Architecture

The extension consists of three main components:

1. **Frontend (Chrome Extension)**
   - Popup interface for file upload
   - Background script for processing
   - Clean, user-friendly UI

2. **Backend (Python)**
   - Resume text extraction
   - AI-powered evaluation
   - Logging and error handling

3. **AI Integration**
   - Google Gemini AI for intelligent analysis
   - Custom prompt engineering for evaluation
   - JSON response parsing

## Installation

### Prerequisites

- Google Chrome browser
- Python 3.7+
- Google Gemini API key

### Setup Steps

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Gemini API**
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Update the API key in `resume_evaluator.py`

3. **Load the Extension in Chrome**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select the extension directory

## Usage

1. Click the extension icon in Chrome
2. Click "Choose File" to upload a resume (PDF or Word)
3. Wait for the evaluation (typically 2-3 seconds)
4. View your results:
   - Match Score (0-100)
   - Detailed feedback and suggestions

## Default Job Description

The extension evaluates resumes against the following job requirements:

```
We are seeking a talented Software Engineer who will:

• Architect, develop and maintain Python-based applications (3+ years' experience)  
• Design and implement robust, secure RESTful APIs using Flask or FastAPI  
• Build, train and deploy deep learning models with TensorFlow and/or PyTorch  
• Containerize & deploy services and ML workloads on AWS EC2 instances  
• Work with relational (PostgreSQL, MySQL) and/or NoSQL (MongoDB, DynamoDB) databases  
• Optimize database schemas, write efficient SQL queries, and ensure data integrity  
• Monitor, troubleshoot and scale production systems in cloud environments  
• Collaborate closely with data scientists, product owners, and QA to deliver features  
• Write clean, testable code and contribute to code reviews and CI/CD pipelines  

**Qualifications:**  
- Bachelor's degree in Computer Science, Engineering or related field  
- 3+ years' professional Python development  
- Strong grasp of REST API design principles and web security best practices  
- Hands-on experience with deep learning frameworks (TensorFlow, PyTorch)  
- Familiarity with AWS EC2 provisioning, AMIs, security groups, and basic networking  
- Experience with at least one DBMS: schema design, indexing, performance tuning  
- Excellent problem-solving skills and a collaborative mindset  
- Good communication skills and ability to work in an Agile environment
```

## Logging

The extension includes comprehensive logging at multiple levels:

1. **Browser Console**
   - Popup script operations
   - Background script activities
   - User interactions

2. **Python Logs**
   - File processing details
   - AI evaluation steps
   - Error tracking
   - Saved to `resume_evaluator.log`

### Log Format
```
YYYY-MM-DD HH:MM:SS - LEVEL - [FUNCTION] Message
```

## Error Handling

The extension includes robust error handling for:
- Invalid file formats
- File processing errors
- AI API failures
- JSON parsing issues
- Communication errors

## Development

### File Structure
```
resume_evaluator/
├── manifest.json          # Extension configuration
├── popup.html            # User interface
├── popup.js              # Frontend logic
├── background.js         # Background processing
├── resume_evaluator.py   # Python backend
├── requirements.txt      # Python dependencies
└── icon.png             # Extension icon
```

### Key Components

1. **Popup Interface**
   - Clean, modern design
   - File upload handling
   - Result display

2. **Background Script**
   - Message handling
   - File processing coordination
   - Error management

3. **Python Backend**
   - Resume text extraction
   - AI integration
   - Evaluation logic

## Dependencies

### Python
- PyPDF2==3.0.1
- python-docx==1.0.1
- typing-extensions==4.9.0
- google-generativeai==0.3.2

### Chrome Extension
- Manifest V3
- Native Messaging (optional)
- File System Access API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini AI for the evaluation engine
- Chrome Extension API documentation
- Python community for the excellent libraries
