from flask import Flask, request, render_template
import spacy

nlp = spacy.load("el_core_news_sm")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        word = request.form.get('word', '').strip()
        if word:
            doc = nlp(word)
            result = []
            for token in doc:
                result.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'morph': token.morph.to_dict()
                })
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
