import streamlit as st
import os
from qa_generator import QAGenerator
from answer_evaluator import Evaluator
api_key = "esecret_35mu1unb6nmdmy5vpyc5jdey8c"

st.title("Generate questions and answers from document")
eval_obj = Evaluator(api_key)
file = st.file_uploader("Upload a PDF document", type='pdf')

if file is not None:
    file_path = "Uploads/" + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    num_questions = st.number_input("Number of questions to create:",min_value=1,max_value=10,value=5)
    if st.button("Generate QAs"):
        qa_obj = QAGenerator(file_path, api_key, llm_model_name= "meta-llama/Llama-2-7b-chat-hf")
        #st.write("Generating Questions...")
        qa_list = qa_obj.get_questions_answers(num_questions)
        questions, user_answers, act_answers = [], [], []
        
    for i, qa in enumerate(qa_list):
        st.write(f"Q{i+1}: {qa['question']}")
        user_answer = st.text_input("Answer:",key=f"answer_{i+1}")
        questions.append(qa['question'])
        act_answers.append(qa['answer'])
        user_answers.append(user_answer)

    if st.button("Evaluate"):
        for i in range(len(questions)):
            response = eval_obj.evaluate_answer(qa['question'],user_answer,qa['answer'])
            st.write(f"Question {i+1} evaluation: "+response)
