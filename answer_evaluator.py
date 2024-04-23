import openai

class Evaluator:
    def __init__(self,anyscale_api_key,llm_model_name="meta-llama/Llama-2-70b-chat-hf",llm_temperature=0.5):
        #LLM configs
        self.llm_endpoint = "https://api.endpoints.anyscale.com/v1"
        self.llm_model_name = llm_model_name
        self.temperature = llm_temperature
        self.anyscale_api_key = anyscale_api_key

    def evaluate_answer(self, question, user_answer, actual_answer):
        if user_answer.strip()=="":
            user_answer = "User did not provide answer."
        evaluation_prompt_template = """Question: {}
        User's answer: {}
        Actual answer: {}
        Respond whether user's answer is correct or wrong based on actual answer to the question and provide explanation. 
        Provide a score out of 2 marks based on the correctness. If the answer is unrelated to the question, respond it is wrong and give 0 marks."""
        prompt = evaluation_prompt_template.format(question, user_answer, actual_answer)

        response = self.get_llm_response(prompt)
        return response
    
    def summarize_evaluation(self, evaluation):
        summarization_prompt_template = """Summarize the following text in 2-3 sentences:\n{}\nRespond in this format:\nSummary:<summary>"""
        prompt = summarization_prompt_template.format(evaluation)
        response = self.get_llm_response(prompt)
        return response

    def get_llm_response(self, prompt):
        client = openai.OpenAI(base_url = self.llm_endpoint,
                                api_key = self.anyscale_api_key)
        chat_completion = client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}],
                        temperature=self.temperature
                )

        return chat_completion.model_dump()['choices'][0]['message']['content']

