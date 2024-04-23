from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from qa_generator import QAGenerator
from answer_evaluator import Evaluator
from mcq_generator import MCQGenerator

with open("api_key.txt", "r") as f:
    api_key = f.read()

UPLOAD_FOLDER = 'Uploads/'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)

eval_obj = Evaluator(api_key)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "Hello123"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    num_questions = int(request.form['num_questions'])
    question_type = request.form.get('question_type', 'verbal')
    if num_questions <= 0:
        return render_template('index.html', error="Number of questions should be greater than 0")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = app.config['UPLOAD_FOLDER'] + filename
        file.save(file_path)
        
        session['question_type'] = question_type
        if question_type=='verbal':
            qa_obj = QAGenerator(file_path, api_key, llm_model_name= "meta-llama/Llama-2-70b-chat-hf")
            session['questions'] = qa_obj.get_questions_answers(num_questions)
        else:
            mcq_obj = MCQGenerator(file_path,api_key,llm_model_name= "meta-llama/Llama-2-70b-chat-hf")
            session['questions'] = mcq_obj.generate_mcqs(num_questions)

        # Pass the first question to the template for display
        return redirect(url_for('show_question', question_index=0))
    else:
        return render_template('index.html', error="Unsupported file type")
    
@app.route('/evaluate', methods=['POST'])
def evaluate():
    user_answer = request.form['user_answer']
    correct_answer = request.form['correct_answer']
    question = request.form['question']
    question_type = session.get('question_type', 'verbal')
    if question_type=='verbal':
        evaluation = eval_obj.evaluate_answer(question,user_answer,correct_answer)
        #evaluation = eval_obj.summarize_evaluation(evaluation)
    else:
        evaluation = correct_answer

    return render_template('evaluation.html', evaluation=evaluation)

@app.route('/next_question', methods=['GET'])
def next_question():
    question_index = int(request.args.get('question_index'))
    next_question_index = question_index + 1 if 'questions' in session and question_index + 1 < len(session['questions']) else None
    if next_question_index is not None:
        return redirect(url_for('show_question', question_index=next_question_index))
    else:
        return "No more questions available"


@app.route('/question/<int:question_index>')
def show_question(question_index):
    # Fetch the next question
    questions = session.get('questions')
    question_type = session.get('question_type', 'verbal')
    if questions is None:
        return redirect(url_for('index')) 
    question = questions[question_index]
    
    # Calculate the next question index
    next_question_index = question_index + 1 if question_index + 1 < len(questions) else None
    
    if question_type == 'multiple_choice':
        return render_template('multiple_choice.html', question=question, question_index=question_index, next_question_index=next_question_index)
    
    return render_template('questions.html', question=question, question_index=question_index, next_question_index=next_question_index)

if __name__ == '__main__':
    app.run(debug=True)