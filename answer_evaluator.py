import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
import os
from prompts import *
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

class Evaluator:
    def __init__(self):
        pass

    def evaluate_answer(self, question, user_answer, actual_answer):
        if user_answer.strip()=="":
            user_answer = "User did not provide answer."
        prompt = evaluation_prompt_template.format(question, user_answer, actual_answer)

        response = self.get_llm_response(prompt)
        return response
    
    def summarize_evaluation(self, evaluation):
        summarization_prompt_template = """Summarize the following text in 2-3 sentences:\n{}\nRespond in this format:\nSummary:<summary>"""
        prompt = summarization_prompt_template.format(evaluation)
        response = self.get_llm_response(prompt)
        return response

    def get_llm_response(self, prompt):
        response = model.generate_content(prompt)

        return response.text

