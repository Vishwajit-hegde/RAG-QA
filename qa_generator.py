### qa_generator.py contains class for converting text chunks into questions and answers. Outputs a pdf files with questions in the beginning and answer in the end.

from pdf_parser import pdf_to_text
from tqdm import tqdm
from typing import Any, Dict, List
import openai

class QAGenerator:
    def __init__(
        self, 
        pdf_paths, 
        anyscale_api_key: str,
        separators: List[str] = ["\n\n\n", "\n\n", "\n", " "],
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        llm_temperature: float = 0.5, 
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
        self.qa_txt_file_name = '_'.join([pdf_paths[i].split('/')[-1].replace('.pdf','') for i in range(len(pdf_paths))]) + '_QAs.txt'

        #LLM configs
        self.llm_endpoint = "https://api.endpoints.anyscale.com/v1"
        self.llm_model_name = llm_model_name
        self.temperature = llm_temperature
        self.anyscale_api_key = anyscale_api_key
        

        self.qa_prompt_template = """Content: {}.\n Based on this content, create a {} question and answer pair to test the knowledge of the user in an exam. Response should be in the follwoing template: \n Q: <question> \n A: <answer>"""

        #test_output = self.get_llm_response("Say 'Test.'")

    def get_questions_answers(self, num_questions: int = 5, save_as_txt = True):

        qns_per_batch = num_questions // len(self.docs) + 1
        
        questions, answers = [], []
        resp_texts = []
        contexts = []
        for i in tqdm(range(min(num_questions,len(self.docs))), desc="Generating questions"):
            # Generate prompt text
            doc = self.docs[i]
            doc_text = doc.page_content
            doc_text = doc_text.replace('\n','  ')
            if qns_per_batch==1:
                prompt = self.qa_prompt_template.format(doc_text, "single")
            else:
                prompt = self.qa_prompt_template.format(doc_text, f"set of {qns_per_batch}")
            resp_text = self.get_llm_response(prompt)
            resp_texts.append(resp_text)

            q_lst, ans_lst = self.convert_resp_text_to_qas(resp_text)
            context_lst = [doc_text for j in range(qns_per_batch)]
            questions.extend(q_lst)
            answers.extend(ans_lst)
            contexts.extend(context_lst)

        qa_list = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

        #qa_list_new = self.select_top_n_questions(qa_list, num_questions)
        output_text = ""
        for i, qa in enumerate(qa_list):
            output_text += 'Q: '
            output_text += qa['question'].strip()
            output_text += '\n'
            output_text += 'A: '
            output_text += qa['answer'].strip()
            output_text += '\n'
            output_text += 'Context: '
            output_text += contexts[i]
            output_text += '\n\n'

        if save_as_txt:
            with open('output_text_files/'+self.qa_txt_file_name,'w',encoding="utf-8") as f:
                f.write(output_text)
        
        output_text = "\nSEPARATOR\n".join(resp_texts)
        with open('output_text_files/'+self.response_txt_file_name,'w',encoding="utf-8") as f:
            f.write(output_text)

        return qa_list
    
    def get_llm_response(self, prompt):
        self.client = openai.OpenAI(base_url = self.llm_endpoint,
                                    api_key = self.anyscale_api_key)
        chat_completion = self.client.chat.completions.create(
        model=self.llm_model_name,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}],
                temperature=self.temperature
        )

        return chat_completion.model_dump()['choices'][0]['message']['content']
    
    def convert_resp_text_to_qas(self, resp_text):
        resp_text_lines = resp_text.split('\n')
        q_lst, ans_lst = [], []
        q = -1
        question, answer = "", ""
        for line in resp_text_lines:
            if line.strip()=="":
                continue
            if line.lower().strip().startswith('q:'):
                if answer.strip() != "":
                    ans_lst.append(answer.strip())
                question = line.strip()[2:]
                q = 1
            elif line.lower().strip().startswith('a:'):
                if question.strip() != "":
                    q_lst.append(question.strip())
                q = 0
                answer = line.strip()[2:]
            elif q==1:
                question += " "
                question += line.strip()
            elif q==0:
                answer += " "
                answer += line.strip()
        if answer.strip()!="":
            ans_lst.append(answer.strip())
        return q_lst, ans_lst
    
    def select_top_n_questions(self, qa_list, n=5):
        total_qas = len(qa_list)
        if n>=total_qas:
            return qa_list
        
        x = total_qas//n + 1
        y = x - 1
        a = n*x - total_qas
        b = n - a

        qa_list_new = []
        prompt_template = """From the following {} question answer pairs, select one pair which is best suited for an exam: \n"""
        for i in tqdm(range(n), desc=f'Selecting best {n} questions'):
            if i<b:
                qa_subset = qa_list[i*x:(i+1)*x]
            else:
                if y==1:
                    qa_list_new.extend(qa_subset)
                    continue
                if i==b:
                    qa_subset = qa_list[i*x:(i+1)*y]
                else:
                    qa_subset = qa_list[i*y:(i+1)*y]
            prompt = prompt_template.format(x)
            for qa in qa_subset:
                question = qa['question']
                answer = qa['answer']
                qa_str = f"Q: {question} \n A: {answer} \n"
                prompt += qa_str

            prompt += "Response should be in the following template: \n Q: <question> \n A: <answer>"
            
            resp_text = self.get_llm_response(prompt)
            q_lst, ans_lst = self.convert_resp_text_to_qas(resp_text)
            qa_list_new.extend([{"question": q, "answer": a} for q, a in zip(q_lst, ans_lst)])

        return qa_list_new







