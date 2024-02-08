from flask import Flask, request, jsonify
# from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)

# CORS(app, origins=['http://localhost:3000'])
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json
        article = data['text']
        print("Received article:", article)  # Debug statement
        max_chunk = 500
        article = article.replace('.', '.<eos>')
        article = article.replace('?', '?<eos>')
        article = article.replace('!', '!<eos>')
        sentences = article.split('<eos>')
        print("Sentences:", sentences)  # Debug statement
        chunks = []
        current_chunk = 0
        for sentence in sentences:
            if len(chunks) == current_chunk + 1:
                if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                    chunks[current_chunk].extend(sentence.split(' '))
                else:
                    current_chunk += 1
                    chunks.append(sentence.split(' '))
            else:
                chunks.append(sentence.split(' '))
        print("Chunks:", chunks)  # Debug statement
        for chunk_id in range(len(chunks)):
            chunks[chunk_id] = ' '.join(chunks[chunk_id])
        print("Processed chunks:", chunks)  # Debug statement

        res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
        print("Summarization results:", res)  # Debug statement
        text = ' '.join([summ['summary_text'] for summ in res])
        print("Summary text:", text)  # Debug statement
        return jsonify({"summary": text})
    
    except Exception as e:
        print("Error:", e)  # Debug statement
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()

