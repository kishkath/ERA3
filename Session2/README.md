**Session2**::: Python Essentials, Version Control, and Web Development Basics

* The session deals with:
  ----------------------

  • Python Programming for AI: Essential Python syntax and data structures relevant to AI programming.

  • Version Control with Git and GitHub: Basic commands, branching, merging, and collaboration workflows.

  • Web Development Introduction: Basics of HTML, CSS, and JavaScript to create simple web interfaces.
  
  • Setting Up a Simple Web Server: Hosting applications locally to interact with AI models.
  


### Session 2 Assignment: 

🔏 Problem Statement:

--------------------

      1. Use Cursor/GhatGPT for all of this. 
      2. Create a HTML/CSS/JS based front end which has 2 boxes.
            A. one box to select one of the 3 check-boxes (cat, dog, elephant)
            B. one box to upload any file
            C. If one selects cat or dog or elephant in the first box, then show a photo of cat, dog, or elephant (you can store this locally as well)
            D. If one uploads any file, then respond back with its name, file-size, and type
      3. You must use Flask or FasiAPI as the backend to achieve this (pick any of your choice)
      4. Create a video and share the YouTube link a& zip the code and upload the code
          

💡 Define Problem:
------------------
 Create a simple web server (via Flask/FastAPI).
 
🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── Session2
    │ ├── images
    │ │ ├── contains images
    │ ├── static
    │ │ ├── script.js : contains the logic (in java script) for the functionality after selecting checkbox or after upload of image, which reads and displays data.
    │ │ ├── styles.css: containt the logic (in css) for the styling of frontend covering all aspects.
    │ ├── templates
    │ │ ├── index.html: contains the UI logic of front end co-operating with sytling (styles.css) & functionality (script.js)
    │ ├── app.py: contains the python code logic to load the server via flask framework
    │ ├── requirements.txt: contains the required libraries for the environment.
    └── README.md Details about the Process.

  Process:
  -------
  * The step by step process for creating a Server using AI.
      * Set up the environment(conda/virutalenv) and then prepare the scripts using the cursor/chatgpt.
      * Finetune the prompt based on the requirement needed
        
              * Server Usage:
                -------------------
                  A. Select any checkbox, once it is selected the respective image gets displayed along with a short information about the image.
                  B. Upload any image, it displays the image as well as the metadata of the image.



💊 Server Result: 
--------------
 Youtube link: https://www.youtube.com/watch?v=zMdk1e-0Crc

