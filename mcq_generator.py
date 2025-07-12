from pdf_parser import pdf_to_text
from tqdm import tqdm
from typing import Any, Dict, List
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
import os
from prompts import *

api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

class MCQGenerator:
    def __init__(
        self, 
        pdf_paths, 
        separators: List[str] = ["\n\n\n", "\n\n", "\n", " "],
        chunk_size: int = 50000,
        chunk_overlap: int = 0,
        parse_func: str = 'pymupdf'
        ):
        # Load documents
        self.docs = pdf_to_text(
            file_paths=pdf_paths,
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parse_func=parse_func
        ) 

        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        self.response_txt_file_name = '_'.join([pdf_paths[i].split('/')[-1].replace('.pdf','') for i in range(len(pdf_paths))]) + '_llm_response.txt'
        self.mcq_txt_file_name = '_'.join([pdf_paths[i].split('/')[-1].replace('.pdf','') for i in range(len(pdf_paths))]) + '_MCQs.txt'
        

        self.qa_prompt_template = mcq_prompt_template

        #test_output = self.get_llm_response("Say 'Test.'")

    def generate_mcqs(self, num_questions: int = 5):
        qns_per_batch = num_questions // len(self.docs) + 1

        questions, answers = [], []
        resp_texts = []
        contexts = []
        for i in tqdm(range(min(num_questions,len(self.docs))), desc="Generating MCQs"):
            # Generate prompt text
            doc = self.docs[i]
            doc_text = doc.page_content
            doc_text = doc_text.replace('\n','  ')
            if qns_per_batch==1:
                prompt = self.qa_prompt_template.format(content=doc_text, N="single")
            else:
                prompt = self.qa_prompt_template.format(content=doc_text, N=f"set of {qns_per_batch}")
            resp_text = self.get_llm_response(prompt)
            resp_texts.append(resp_text)

            q_lst, ans_lst = self.convert_resp_text_to_mcqs(resp_text)
            context_lst = [doc_text for j in range(qns_per_batch)]
            questions.extend(q_lst)
            answers.extend(ans_lst)
            contexts.extend(context_lst)

        qa_list = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        mcqs = []
        for qa in qa_list:
            mcqs.append(self.get_options(qa))
        return mcqs
    
    def get_llm_response(self, prompt):
        response = model.generate_content(prompt)
        with open("test_output.txt","w") as f:
            f.write(response.text)
        return response.text
    
    def convert_resp_text_to_mcqs(self,resp_text):
        resp_text_lines = resp_text.split('\n')
        lb = "<line_break>"
        resp_text_lines = [line+lb for line in resp_text_lines]
        q_lst, ans_lst = [], []

        q = -1
        question, answer = "", ""
        for line in resp_text_lines:
            if line.strip().replace(lb,"")=="":
                continue
            if line.lower().strip().startswith('question'):
                if answer.strip() != "":
                    ans_lst.append(answer.strip().replace(lb,"\n"))
                question = line.split(":")[1].strip()
                q = 1
            elif line.lower().strip().startswith('answer'):
                if question.strip() != "":
                    q_lst.append(question.strip().replace(lb,"\n"))
                q = 0
                answer = line.split(":")[1].strip()
            elif q==1:
                question += " "
                question += line.strip()
            elif q==0:
                answer += " "
                answer += line.strip()
        if answer.strip()!="":
            ans_lst.append(answer.strip().replace(lb,"\n"))
        return q_lst, ans_lst
    
    def get_options(self, qa):
        q = qa['question']
        a = qa['answer']
        f = 1
        q_lines = q.split("\n")
        options = {}
        question = ""
        for line in q_lines:
            if line.strip().startswith('A)')==False and f==1:
                question += line.strip() + "\n"
            if line.strip().startswith('A)'):
                f = 0
                options['A'] = line.strip()[2:].strip()
            if line.strip().startswith('B)'):
                options['B'] = line.strip()[2:].strip()
            if line.strip().startswith('C)'):
                options['C'] = line.strip()[2:].strip()
            if line.strip().startswith('D)'):
                options['D'] = line.strip()[2:].strip()

        if "A" in a:
            correct_option = 'A'
        elif "B" in a:
            correct_option = "B"
        elif "C" in a:
            correct_option = "C"
        else:
            correct_option = "D"
        mcq = {}
        mcq['question'] = question.strip()
        mcq['answer'] = correct_option + ") " + options[correct_option]
        mcq['options'] = ["A) "+options['A'],"B) "+options['B'],"C) "+options['C'],"D) "+options['D']]
        return mcq
    
    def reformulate_question(self,mcq):
        return None