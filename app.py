from flask import Flask, request, render_template, jsonify, send_file, url_for,redirect
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import string

app = Flask(__name__)

# Configuration
MARKDOWN_PATH = "Hackwknd1\data\English For Communication_Special Education_Form_2.md"
VECTOR_STORE_DIR = "./vector_store"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"

qa_chain = None

def create_vector_store(markdown_path, persist_directory=VECTOR_STORE_DIR):
    """Creates a vector store from a markdown file."""
    try:
        with open(markdown_path, 'r', encoding='utf-8') as file:
            text = file.read()
            text = text.replace("\n", " ")

            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))

            # Optionally remove extra spaces
            text = " ".join(text.split())

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=45)
        docs = text_splitter.split_text(text)

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vs = Chroma.from_texts(texts=docs, embedding=embeddings, persist_directory=persist_directory)
        return vs
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


def initialize_qa_chain(vs):
    """Initialize a RetrievalQA chain with a custom prompt and OllamaLLM."""
    try:
        llm = OllamaLLM(model=LLM_MODEL, format="json")
        retriever = vs.as_retriever(search_kwargs={"k": 3})

        custom_prompt = PromptTemplate(
             template=(
            "You are a helpful AI assistant. Use the context below to answer the user's question, Do not provide speculative or fabricated answers,Always ensure clarity and precision in your responses.If the answer is not in the context, respond with: \"I don’t know based on the provided information.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )

        qa_chain_base = load_qa_chain(llm=llm, chain_type="stuff", prompt=custom_prompt)
        retrieval_qa_chain = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=qa_chain_base
        )
        return retrieval_qa_chain
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/learning_style_test')
def learning_style():
    return render_template('learning_style.html')

@app.route('/chap1_quiz')
def quiz1():
    return render_template('quiz_Chap1.html')

@app.route('/chap2_quiz')
def quiz2():
    return render_template('quiz_Chap2.html')

@app.route('/download')
def download_book():
    file_path = 'data/English For Communication_Special Education_Form 2.pdf'
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return f"Error: {e}"

@app.route('/query', methods=['POST'])
def query():
    global qa_chain
    if qa_chain is None:
        return jsonify({'error': 'QA chain not initialized'}), 500

    data = request.get_json(force=True)
    query_text = data.get('query', '')
    if not query_text.strip():
        return jsonify({'error': 'No query provided'}), 400

    response = qa_chain.run(query_text)
    app.logger.info(f"Query: {query_text} | Response: {response}")
    print("LLM Response:", response)
    return jsonify({'response': response})

# Questions with correct answers
# QUIZ_ANSWERS = {
#     "text_question1": "C",
#     "text_question2": "D",
#     "text_question3": "B",
#     "text_question4": "D",
#     "text_question5": "A",
#     "visual_question1": "B",
#     "visual_question2": "C",
#     "visual_question3": "D",
#     "visual_question4": "B",
#     "visual_question5": "A",
#     "audio_question1": "B",
#     "audio_question2": "C",
#     "audio_question3": "B",
#     "audio_question4": "C",
#     "audio_question5": "C"
# }

@app.route("/check-answers", methods=["POST"])
def check_answers():
    return redirect(url_for('index'))

#     Retrieve form data
#     user_answers = request.form.to_dict()
    
#     Initialize scores
#     scores = {"text": 0, "visual": 0, "audio": 0}
    
#     Calculate scores
#     for question, correct_answer in QUIZ_ANSWERS.items():
#         if question in user_answers and user_answers[question] == correct_answer:
#             if question.startswith("text"):
#                 scores["text"] += 1
#             elif question.startswith("visual"):
#                 scores["visual"] += 1
#             elif question.startswith("audio"):
#                 scores["audio"] += 1
    
#     Determine the highest section
#     highest_section = max(scores, key=scores.get)
    
#     Redirect to results page with scores
#     return render_template("result.html", scores=scores, highest_section=highest_section)


if __name__ == '__main__':
    try:
        if os.path.exists(MARKDOWN_PATH):
            print("Using markdown file:", MARKDOWN_PATH)
            vector_store = create_vector_store(MARKDOWN_PATH)
            if vector_store:
                print("Vector store initialized.")
                qa_chain = initialize_qa_chain(vector_store)
                if qa_chain:
                    print("QA chain successfully initialized.")
                else:
                    print("Failed to initialize QA chain.")
            else:
                print("Failed to create vector store.")
        else:
            print(f"Markdown file not found at {MARKDOWN_PATH}. Please provide the file.")

        app.run(debug=True)
    except Exception as e:
        print(f"Critical error during initialization: {e}")
