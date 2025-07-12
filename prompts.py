mcq_prompt_template = """Your role is an examiner who wants to test the knowledge of the user based on the content. 
You are taksed with testing how well the user understands the content in the document. 
Content: {content}.
Based on this content, create {N} multiple choice question and answer pair with 4 options and single correct answer to test the knowledge of the user in an exam. 
Response should be in the follwoing template:
Question: <question> 
A) <option A> 
B) <option B> 
C) <option C> 
D) <option D> 
Answer: <correct option (whether it is A or B or C or D)>

Note: Do not keep the same correct option for every question. Basically, the correct option can be either A or B or C or D. It should not be same option always."""

qa_prompt_template = """Your role is an examiner who wants to test the knowledge of the user based on the content. 
You are taksed with testing how well the user understands the content in the document. 
Content: {content}.
Based on this content, create {N} question and answer pair to test the knowledge of the user in an exam. 
Response should be in the follwoing template: 
Q: <question> 
A: <answer>"""

evaluation_prompt_template = """Question: {}
User's answer: {}
Actual answer: {}
Respond whether user's answer is correct or wrong based on actual answer to the question and provide explanation. 
Also, mention what the actual answer is before providing the score.
Provide a score out of 2 marks based on the correctness. If the answer is unrelated to the question, respond it is wrong and give 0 marks."""

