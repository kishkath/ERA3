import PyPDF2
import docx
import json
import google.generativeai as genai
from typing import Dict, Optional
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('resume_evaluator.log')
    ]
)

# Configure Gemini API
genai.configure(api_key='AIzaSyCtxeBmeKjjvhHhUblb_nSvm-fRtVHed7E')  # Replace with your actual API key
model = genai.GenerativeModel('gemini-2.0-flash')

# # Default job description
# DEFAULT_JOB_DESCRIPTION = """We are looking for a Software Engineer with:
# - 3+ years of experience in Python
# - Strong knowledge of web development including REST APIs via Flask
# - A bit of familiary with cloud platforms (AWS/GCP)
# - Excellent problem-solving skills
# - Bachelor's degree in Computer Science or related field
# """

DEFAULT_JOB_DESCRIPTION = """
We are seeking a talented Software Engineer who will:

• Architect, develop and maintain Python-based applications (3+ years’ experience)  
• Design and implement robust, secure RESTful APIs using Flask or FastAPI  
• Build, train and deploy deep learning models with TensorFlow and/or PyTorch  
• Containerize & deploy services and ML workloads on AWS EC2 instances  
• Work with relational (PostgreSQL, MySQL) and/or NoSQL (MongoDB, DynamoDB) databases  
• Optimize database schemas, write efficient SQL queries, and ensure data integrity  
• Monitor, troubleshoot and scale production systems in cloud environments  
• Collaborate closely with data scientists, product owners, and QA to deliver features  
• Write clean, testable code and contribute to code reviews and CI/CD pipelines  

**Qualifications:**  
- Bachelor’s degree in Computer Science, Engineering or related field  
- 3+ years’ professional Python development  
- Strong grasp of REST API design principles and web security best practices  
- Hands-on experience with deep learning frameworks (TensorFlow, PyTorch)  
- Familiarity with AWS EC2 provisioning, AMIs, security groups, and basic networking  
- Experience with at least one DBMS: schema design, indexing, performance tuning  
- Excellent problem-solving skills and a collaborative mindset  
- Good communication skills and ability to work in an Agile environment
"""
def upload_resume(path: str) -> str:
    """
    Loads and extracts text from the uploaded resume file.
    Supports PDF and Word documents.
    
    Args:
        path (str): Path to the resume file
        
    Returns:
        str: Extracted text from the resume
    """
    logging.info(f"[upload_resume] Starting resume upload for file: {path}")
    
    try:
        if path.lower().endswith('.pdf'):
            logging.info("[upload_resume] Processing PDF file")
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for i, page in enumerate(reader.pages):
                    logging.debug(f"[upload_resume] Processing page {i+1}")
                    text += page.extract_text()
                logging.info("[upload_resume] PDF processing completed")
                return text
        elif path.lower().endswith(('.doc', '.docx')):
            logging.info("[upload_resume] Processing Word document")
            doc = docx.Document(path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logging.info("[upload_resume] Word document processing completed")
            return text
        else:
            error_msg = f"Unsupported file format: {path}"
            logging.error(f"[upload_resume] {error_msg}")
            raise ValueError(error_msg)
    except Exception as e:
        logging.error(f"[upload_resume] Error processing file: {str(e)}")
        raise

def evaluate(resume_text: str, job_text: str = DEFAULT_JOB_DESCRIPTION) -> Dict:
    """
    Evaluates the resume against the job description using Gemini AI.
    
    Args:
        resume_text (str): Text extracted from the resume
        job_text (str): Job description text
        
    Returns:
        Dict: Evaluation results containing score and feedback
    """
    logging.info("[evaluate] Starting resume evaluation")
    
    try:
        # Create a prompt for the LLM
        prompt = f"""Please evaluate this resume against the following job description and provide:
        1. A match score from 0-100
        2. Detailed feedback on strengths and areas for improvement

        Job Description:
        {job_text}

        Resume:
        {resume_text}

        Please respond in JSON format with 'score' and 'feedback' fields."""

        logging.info("[evaluate] Sending prompt to Gemini AI")
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        logging.info("[evaluate] Received response from Gemini AI")
        
        # Parse the response
        try:
            result = json.loads(response.text)
            logging.info(f"[evaluate] Successfully parsed response. Score: {result.get('score')}")
            return {
                "score": int(result.get("score", 0)),
                "feedback": result.get("feedback", "No feedback available")
            }
        except json.JSONDecodeError as e:
            logging.error(f"[evaluate] Error parsing JSON response: {str(e)}")
            logging.warning("[evaluate] Falling back to basic evaluation")
            return {
                "score": 50,
                "feedback": "Unable to parse AI response. Please try again."
            }
    except Exception as e:
        logging.error(f"[evaluate] Error during evaluation: {str(e)}")
        return {
            "score": 0,
            "feedback": f"Error during evaluation: {str(e)}"
        }

class ResumeEvaluatorAgent:
    def __init__(self):
        logging.info("[ResumeEvaluatorAgent] Initializing agent")
        self.max_iterations = 3
        self.iteration = 0
        self.last_response = None
        self.iteration_response = []
        
        self.system_prompt = """You are a resume evaluation agent. Respond with EXACTLY ONE of these formats:
1. FUNCTION_CALL: function_name|input
2. FINAL_ANSWER: [result]

where function_name is one of:
1. upload_resume(path) - Uploads and extracts text from resume
2. evaluate(resume_text) - Evaluates resume against job description

DO NOT include multiple responses. Give ONE response at a time."""

    def process(self, query: str) -> str:
        """
        Process the user's query through iterations.
        
        Args:
            query (str): User's input query
            
        Returns:
            str: Final response or next action
        """
        logging.info(f"[process] Starting processing with query: {query[:50]}...")
        
        while self.iteration < self.max_iterations:
            logging.info(f"[process] Starting iteration {self.iteration + 1}")
            
            if self.last_response is None:
                current_query = query
            else:
                current_query = query + "\n\n" + " ".join(self.iteration_response)
                current_query += " What should I do next?"

            # Get model's response (simulated)
            if self.iteration == 0:
                response = "FUNCTION_CALL: upload_resume|" + query
                logging.info("[process] Calling upload_resume function")
            elif self.iteration == 1:
                response = "FUNCTION_CALL: evaluate|" + self.last_response
                logging.info("[process] Calling evaluate function")
            else:
                response = "FINAL_ANSWER: " + json.dumps(self.last_response)
                logging.info("[process] Generating final answer")

            if response.startswith("FUNCTION_CALL:"):
                _, function_info = response.split(":", 1)
                func_name, params = [x.strip() for x in function_info.split("|", 1)]
                logging.info(f"[process] Executing function: {func_name} with params: {params[:50]}...")
                
                if func_name == "upload_resume":
                    self.last_response = upload_resume(params)
                elif func_name == "evaluate":
                    self.last_response = evaluate(params)
                    
                self.iteration_response.append(
                    f"In iteration {self.iteration + 1}, {func_name} was called with {params}, "
                    f"returning {self.last_response}"
                )
                logging.info(f"[process] Function {func_name} completed successfully")
                
            elif response.startswith("FINAL_ANSWER:"):
                logging.info("[process] Process completed successfully")
                return response.split(":", 1)[1].strip()

            self.iteration += 1
            
        logging.warning("[process] Maximum iterations reached without final answer")
        return "Maximum iterations reached without final answer." 

# Register the native messaging host
subprocess.run(["reg", "import", "register_host.reg"])