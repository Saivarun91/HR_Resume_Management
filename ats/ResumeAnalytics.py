from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import ConversationChain
from IPython.display import Markdown, display
from IPython.display import display, Markdown
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv, find_dotenv
from newspaper import Article
from docx import Document
import pytesseract
from typing import List, Dict
import pytesseract
import os
import markdown
from functools import wraps
from newspaper import Article
import re
import json
from PIL import Image
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from json import JSONDecodeError
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ExceptionHandeler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Error] {e}")
            return f"Error: {str(e)}"
    return wrapper

   
class ResumeAnalytics(object):
    def __init__(self, modelname: str = 'models/gemini-2.0-flash', chatmodel = "models/gemma-3-27b-it") ->None:
        load_dotenv(find_dotenv())
        self.__API = os.getenv("GOOGLE_API_KEY")
        if not self.__API:
            raise ValueError("API key not found. Please set the GEMINIAPI environment variable.")
        genai.configure(api_key = self.__API)
        self.outputsFOLDER = "outputs"
        self.model: genai.GenerativeModel = None
        self.chatmodel: ChatGoogleGenerativeAI = None
        for model in genai.list_models():
            if model.name == modelname:
                self.model: genai.GenerativeModel = genai.GenerativeModel(
                    model_name=modelname,
                    generation_config ={"response_mime_type":"application/json"},
                    safety_settings={},
                    tools = None,
                    system_instruction = "You are an expert resume screening assistant. Always return JSON. Be concise."
                )
            elif model.name == chatmodel:
                self.chatmodel = ChatGoogleGenerativeAI(
                    model = modelname,
                    temperature = 0.7,
                    max_output_tokens = 10000,
                    top_k = 40,
                    top_p = 0.95
                )
        logger.info("CHAT/TEXT GENERATION MODELS initialized successfully.")
        if self.model is None or self.chatmodel is None:
            raise ValueError(f"Error in initlizing the models")
        self.chatmemory = ConversationSummaryBufferMemory(
            llm = self.chatmodel,
            max_token_limit=10000,
            return_messages=True,
            memory_key="history"
        )
        self.memory = ConversationSummaryBufferMemory(
            llm=self.chatmodel,
            max_tokens=100000,
            return_messages=True,
            memory_key="history"
        )
        print("[DEBUG] GOOGLE_API_KEY:", self.__API)
    

    
    @ExceptionHandeler 
    @property
    def getAPI(self) -> Any:
        return self.__API
    
    @ExceptionHandeler
    @property
    def getMODEL(self) -> genai.GenerativeModel:
        return self.model
    
    @ExceptionHandeler
    def getResponse(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        else:
            raise ValueError("Invalid response format from the model.")
    
    def datacleaning(self, textfile: str) -> str:
        #cleaning special symbols/characters from the given data to reduce the tokens 
        if not textfile or textfile.strip() == "":
            return ""
        
        text = textfile
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\x00-\x7F]', ' ', text)
        text = re.sub(r'[-–]', ' ', text) 
        text = re.sub(r'(\w)[|](\w)', r'\1, \2', text)
        
        text = text.replace(" - ", "\n- ").replace(":", ":\n")  

        return text.strip()
    
    @ExceptionHandeler
    def documentParser(self, filepath: str) -> dict:
        if not filepath:
            raise ValueError("No input provided.")

        if os.path.exists(filepath):
            ext = os.path.splitext(filepath)[1].lower() if os.path.exists(filepath) else None
        
        if filepath.startswith("http://") or filepath.startswith("https://"):
            scraped_data = ""
            A = Article(filepath)
            A.download()
            A.parse()
            scraped_data += A.text
            if scraped_data:
                print(f"successfully scraped data from given URL (filepath)")
                return {"content": scraped_data, "pages": 1}
            else:
                raise ValueError("couldn't find/ Error in scraping data from the given website")
                
        elif ext in [".pdf", ".docx", ".txt"]:
            if ext == ".pdf":
                loader = PyMuPDFLoader(filepath)
            elif ext == ".docx":
                loader = Docx2txtLoader(filepath)
            elif ext == ".txt":
                loader = TextLoader(filepath)
            else:
                print("Unsupported document format.")
                raise ValueError("Unsupported document format.")

            document = loader.load()
            filecontent: str = " ".join([doc.page_content for doc in document])
            pages = len(document)
            return {
                "content": self.datacleaning(filecontent.strip()) if filecontent else None,
                "pages": pages
            }

        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            image = Image.open(filepath)
            if image.mode != "RGB":
                image = image.convert("RGB")
            filecontent: str = pytesseract.image_to_string(image)
            if not filecontent.strip():
                raise ValueError("No text found in the image.")
            return {
                "content": self.datacleaning(filecontent),
                "pages": 1 
            }
        else:
            print("Invalid file format.")
            raise ValueError("Invalid file format.")

    @ExceptionHandeler
    def resumeanalytics(self, resumepath: str, jobdescription: str, filename: str = "prompt.txt", output_folder: str = None) -> Optional[Dict[str, Any]]:
        resume = self.documentParser(resumepath)
        JobDescription = self.documentParser(jobdescription)

        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        with open(filename, "r", encoding="utf-8") as file:
            prompt = file.read()
        Fprompt = f"{prompt}\nResume: {resume}\nJob Description: {JobDescription}"
        outputs_folder = output_folder or self.outputsFOLDER
        if not os.path.exists(outputs_folder):
            os.makedirs(outputs_folder, exist_ok=True)
        import datetime

        filename = os.path.split(os.path.basename(resumepath))[1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{filename}_{timestamp}.json"
        savePath = os.path.join(outputs_folder, save_filename)
        try:
            response = self.getResponse(Fprompt)
            responseJSON = json.loads(response)
            with open(savePath, "w", encoding="utf-8") as file:
                json.dump(responseJSON, file, indent=4, ensure_ascii=False)
            print(f"JSON file saved: {savePath}")
            
            return responseJSON
        except JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error in resumeanalytics: {e}")
            return None

    @ExceptionHandeler  
    def chatbot(self, query: str) -> str:
        if not query or query.strip() == "":
            return "Please type a message to continue."

        messages = self.memory.chat_memory.messages.copy()
        messages.append(HumanMessage(content=query))

        response = self.chatmodel.invoke(messages).content
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response)

        return markdown.markdown(response)
    
    from langchain_core.prompts import ChatPromptTemplate

    def getCoverLetter(self, resume: str, jd: str, output_folder: str = None) -> str:
        # Step 1: Parse the resume and JD
        resume_text = self.documentParser(resume)
        jd_text = self.documentParser(jd)

        # Step 2: Validate parsed data
        if not isinstance(resume_text, dict) or not resume_text.get("content"):
            logger.error("[getCoverLetter] Resume parsing failed or returned invalid data: %s", resume_text)
            return "Error: Resume content could not be extracted. Please upload a valid file."

        if not isinstance(jd_text, dict) or not jd_text.get("content"):
            logger.error("[getCoverLetter] JD parsing failed or returned invalid data: %s", jd_text)
            return "Error: Job description content could not be extracted. Please upload a valid file."

        # Step 3: Define the prompt using LangChain
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following resume and job description, generate a professional cover letter.
            The cover letter should highlight the candidate's relevant skills and experiences that match the job requirements.
            Write a long cover letter (200 to 500 words) if there’s enough information.

            Resume:
            {RESUME}
            Job Description:
            {JOBDESCRIPTION}

            Format the cover letter as follows:
            1. Start with a professional greeting
            2. Include an opening paragraph expressing interest in the position
            3. Body paragraphs highlighting 2-3 key qualifications
            4. A closing paragraph reiterating interest and providing contact info
            5. End with a professional sign-off

            The title should be in **bold** markdown.
            Tone: Professional, enthusiastic, confident.
            Output only the cover letter — no instructions or summaries.
            """
        )

        # Step 4: Format and send to Gemini
        try:
            formatted_prompt = prompt.format(
                RESUME=resume_text["content"],
                JOBDESCRIPTION=jd_text["content"]
            )

            logger.info("[getCoverLetter] Prompt formatted successfully.")
            response = self.getResponse(formatted_prompt)
            logger.info("[getCoverLetter] Gemini response received.")
        except Exception as e:
            logger.error("[getCoverLetter] Error during prompt formatting or response: %s", e)
            return "Error: Failed to generate cover letter from AI."

        # Step 5: Save cover letter to file
        try:
            outputs_folder = output_folder or self.outputsFOLDER
            if not os.path.exists(outputs_folder):
                os.makedirs(outputs_folder, exist_ok=True)

            outputpath = os.path.join(outputs_folder, "CoverLetter.txt")
            with open(outputpath, "w", encoding="utf-8") as file:
                file.write(response)

            logger.info("[getCoverLetter] Cover letter saved to: %s", outputpath)
        except Exception as e:
            logger.error("[getCoverLetter] Failed to save cover letter: %s", e)

        return response

    @ExceptionHandeler
    def ATSanalytics(self, resume: str, jobdescription: str = None, output_folder: str = None) -> Optional[Dict[str, Any]]:
        resume_data: dict = self.documentParser(resume)
        resumeLength: int = resume_data.get("pages", 0) if resume_data else 0
        JD = None
        if jobdescription:
            JD = self.documentParser(jobdescription)
        if not resume_data or not resume_data.get("content"):
            logger.error("No content found in the resume.")
            return {"error": "Could not extract meaningful content from resume."}
        logger.info("Resume content extracted successfully.")
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
            """
            You are an advanced ATS (Applicant Tracking System) evaluator.

            Your job is to:
            1. Extract key information from the given resume.
            2. Calculate an overall ATS score (0 - 100) based on:
                - Resume Format & Length (10 points)
                - Spelling & Grammar (10 points)
                - Summary or Objective (10 points)
                - Skills: Hard & Soft (10 points)(remove some points if the user misses any of the skills in soft or hard)
                - Work Experience (10 points)
                - Projects (10 points)
                - Certifications (10 points)
                - Education (10 points)
                - Contact Details (10 points)
            3. Penalize for:
                - Missing sections (e.g., no certifications, no contact details)
                - Resume longer than 2 pages (deduct up to 5 points from format score)
            4. Provide specific improvement recommendations.

            === Resume Content ===
            {resume_text}
            Length (in pages): {resumeLength}
            === OUTPUT FORMAT ===
            Return a valid JSON object in this structure:
            {{
                "Extracted Data": {{
                    "Name": "...",
                    "Contact Details": "...",
                    "Summary or Objective": "...",
                    "Skills": {{
                        "Soft Skills": [...],
                        "Hard Skills": [...]
                    }},
                    "Experience": [
                        {{
                            "Title": "...",
                            "Company": "...",
                            "Duration": "...",
                            "Description": "..."
                        }}
                    ],
                    "Projects": [...],
                    "Certifications": [...],
                    "Education": "..."
                }},
                "ATS Score": {{
                    "Total Score": <score_out_of_100>,
                    "Breakdown": {{
                        "Format Score": <score_out_of_10>,
                        "Spelling & Grammar": <score_out_of_10>,
                        "Summary": <score_out_of_10>,
                        "Skills": <score_out_of_10>,
                        "Experience": <score_out_of_10>,
                        "Projects": <score_out_of_10>,
                        "Certifications": <score_out_of_10>,
                        "Education": <score_out_of_10>,
                        "Contact Details": <score_out_of_10>
                    }}
                }},
                "Recommendations": [
                    "...",  // bullet point suggestions for improvement
                    "...",
                    "...",
                    "7 recommendations"
                    - Ensure that each point is big like 2 lines and also highlight the main keywords with bold ** here i will use markdown formar in web app
                ]
            }}
            """
        )
        formatted_prompt = prompt.format(
            resume_text=resume_data["content"],
            resumeLength=resumeLength
        )
        logger.info("ATS Prompt formatted successfully.")
        response = self.getResponse(formatted_prompt)
        logger.info("ATS analytics response received from model.")
        outputs_folder = output_folder or self.outputsFOLDER
        try:
            if not os.path.exists(outputs_folder):
                os.makedirs(outputs_folder, exist_ok = True)
            responseJSON = json.loads(response)
            outputpath = os.path.join(outputs_folder, "ATSanalytics.json")
            with open(outputpath, "w", encoding="utf-8") as file:
                json.dump(responseJSON, file, indent=4, ensure_ascii=False)
            print(f"ATS analytics JSON file saved: {outputpath}")
            return responseJSON
        except JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error in ATSanalytics: {e}")
            return None
    
    def getJobRecommendations(self, resume: str, output_folder: str = None) -> Optional[Dict[str, Any]]:
        try:
            resume_data = self.documentParser(resume)
            if not isinstance(resume_data, dict) or not resume_data.get("content"):
                logger.error("No content found in the resume.")
                return {"error": "Could not extract meaningful content from resume."}

            logger.info("Resume content extracted successfully.")

            prompt_text = (
                "You are an expert career advisor. Based on the resume content below, analyze the candidate's skills, experience, and qualifications. "
                "Identify the top 5 job roles that are most relevant to the resume and assign a relevance score out of 100 for each. "
                "Return the result as a valid JSON object, where each key is a job role and the value is the relevance score (an integer between 0 and 100). "
                "\n\n=== OUTPUT FORMAT EXAMPLE ===\n"
                "{\n"
                '    "ROLEMATCHES": {\n'
                '        "Data Scientist": 85,\n'
                '        "Machine Learning Engineer": 82,\n'
                '        "PCB Designer": 78,\n'
                '        "Data Analyst": 75,\n'
                '        "Software Engineer": 70\n'
                '    }\n'
                "}\n"
                "=== Resume Content ===\n"
                f"{resume_data['content']}"
            )

            logger.info("Job Recommendations Prompt formatted successfully.")
            response = self.getResponse(prompt_text)
            logger.info("Job recommendations response received from model.")

            try:
                responseJSON = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse Gemini JSON response:\n%s", response)
                return {"error": "Gemini returned malformed JSON. Check model output."}

            outputs_folder = output_folder or self.outputsFOLDER
            if not os.path.exists(outputs_folder):
                os.makedirs(outputs_folder, exist_ok=True)

            outputpath = os.path.join(outputs_folder, "JobRecommendations.json")
            with open(outputpath, "w", encoding="utf-8") as file:
                json.dump(responseJSON, file, indent=4, ensure_ascii=False)

            print(f"Job recommendations JSON file saved: {outputpath}")
            return responseJSON

        except Exception as e:
            logger.exception("Error in getJobRecommendations:")
            return {"error": "An internal error occurred while generating recommendations."}

        
    def pdfchatbot(self, Documents: List[str], Query: str) -> str:
        try:
            if not Query or Query.strip() == "":
                return "Please type a query to continue."
            if not hasattr(self, 'chatmemory') or self.chatmemory is None:
                # Reduce the max_tokens to a more reasonable number
                self.chatmemory = ConversationSummaryBufferMemory(
                    llm=self.chatmodel,
                    max_token_limit=4000,  # More reasonable limit
                    return_messages=True,
                    memory_key="history"
                )
                logger.log(logging.INFO, "Chat memory initialized.")
            if Documents and any(doc.strip() != "" for doc in Documents):
                self.corpus = []
                for doc in Documents:
                    tempdata = self.documentParser(doc)
                    if tempdata and tempdata.get("content"):
                        self.corpus.append(tempdata["content"])

                if not self.corpus:
                    return "No content found in the uploaded documents."

                combined_documents = "\n".join(self.corpus)
                self.chatmemory.chat_memory.add_ai_message(
                    "Here are the uploaded documents that you should use to answer future queries:\n\n" + combined_documents
                )
                logger.log(logging.INFO, "Uploaded documents added to memory.")

            elif not hasattr(self, 'corpus') or not self.corpus:
                return "Please upload one or more documents to continue."
            messages = self.chatmemory.chat_memory.messages.copy()
            messages.append(HumanMessage(content=Query))

            response = self.chatmodel.invoke(messages).content
            self.chatmemory.chat_memory.add_user_message(Query)
            self.chatmemory.chat_memory.add_ai_message(response)

            return markdown.markdown(response)

        except Exception as e:
            logger.log(logging.ERROR, f"Error in pdfchatbot: {str(e)}")
            return "An error occurred while processing the documents. Please try again."
        
    def getCustomCoverLetter(self, job_title, company_name, your_name, additional_info, output_folder: str = None):
        model = genai.GenerativeModel(
            model_name="models/gemini-2.0-flash",
            generation_config={"response_mime_type": "text/plain"},
            safety_settings={},
            tools=None,
            system_instruction="You are an expert cover letter generator. Always return plain text."
        )
        prompt = f"""
        Generate a professional cover letter with the following details:
        - Applicant Name: {your_name}
        - Company Name: {company_name}
        - Job Title: {job_title}
        - Additional Information: {additional_info}

        The cover letter should:
        1. Be properly formatted with sender/recipient information
        2. Include a professional salutation
        3. Clearly state the position being applied for
        4. Highlight relevant skills and experiences
        5. Show enthusiasm for the position
        6. Include a professional closing
        give me markdown format for some main points or impoortant information
        Return ONLY the raw text of the cover letter, no JSON formatting.
        """
        
        response = model.generate_content(prompt)
        
        outputs_folder = output_folder or self.outputsFOLDER
        if not os.path.exists(outputs_folder):
            os.makedirs(outputs_folder, exist_ok=True)
        outputpath = os.path.join(outputs_folder, "CustomCoverLetter.txt")
        with open(outputpath, "w", encoding="utf-8") as file:
            file.write(response.text)
        
        logger.info("Custom cover letter generated successfully.")
        return markdown.markdown(response.text)
    
    def generate_enhanced_resume(self, resume_content: str, improvements: list = None, missing_skills: list = None, missing_keywords: list = None, tips: list = None) -> dict:
        """
        Enhance the resume content by adding missing skills, keywords, and applying tips.
        """
        content = resume_content or ''
        # Add missing skills/keywords to skills section
        skills_section = '\n'.join(missing_skills or [])
        keywords_section = '\n'.join(missing_keywords or [])
        tips_section = '\n'.join(tips or [])
        # Compose enhanced fields
        enhanced = {
            'about_me': (content[:300] + '\n' + tips_section).strip(),
            'skills': skills_section + ('\n' + keywords_section if keywords_section else ''),
            'exp1': 'Experience 1 improved',
            'exp2': 'Experience 2 improved',
            'exp3': 'Experience 3 improved',
            'design_tools': 'AutoCAD, SketchUp, Photoshop',
            'achievements': 'Added missing achievements. ' + tips_section,
            'reference': 'Available upon request',
            'raw_content': content
        }
        return enhanced
    

    






