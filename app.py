if __name__ == '__main__':
    from torch.utils.model_zoo import load_url
    import os
    import json
    from flask_cors import CORS
    from flask import Flask, request, render_template, jsonify
    from BeheadedInception3 import Model
    from CaptioningNet import CaptModel
    from Predict import *



app = Flask(__name__)
CORS(app, headers=['Content-Type'])


def load_inception(inception_path="models/inception.pth"):
    inception_model = Model(transform_input=True)
    inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

    inception_model.load_state_dict(load_url(inception_url, map_location=torch.device('cpu')))
    inception_model.eval()
    return inception_model


def load_recurrent(vocab, pad_ix,
                   recurrent_path="models/recurrent.pth"):
    recurrent_model = CaptModel(n_tokens=len(vocab), pad_ix=pad_ix)
    recurrent_model.load_state_dict(torch.load(recurrent_path,
                                               map_location=torch.device('cpu')))
    recurrent_model.eval()
    return recurrent_model


def captions_preprocessing(captions):
    start_end(captions)
    word_counts = build_vocab(captions)
    return pad(word_counts)


captions = json.load(open("models/captions_tokenized.json")) #load captions json
word_to_index, vocab, eos_ix, unk_ix, pad_ix = captions_preprocessing(captions) #preprocessing word vocab
inception_model = load_inception() #download inception network
recurrent_model = load_recurrent(vocab, pad_ix) #download recurrent network



@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():
	return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict(vocab=vocab, word_to_index=word_to_index,
            unk_ix=unk_ix, pad_ix=pad_ix,
            inception_model=inception_model,
            recurrent_model=recurrent_model):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "something went wrong!"

        user_file = request.files['file']
        if user_file.filename == '':
            return "file name not found ..."

        else:
            path = os.path.join(user_file.filename)
            captions = []
            for i in range(5):
                capt = ' '.join(generate_caption(image=user_file, vocab=vocab,
                                             word_to_index=word_to_index,
                                             unk_ix=unk_ix, pad_ix=pad_ix,
                                             inception=inception_model,
                                             recurrent=recurrent_model, t=5.)[1:-1])
                captions.append(capt)

            return jsonify({
                "prediction" + "_#_" + str(number + 1): capt for number, capt in enumerate(captions)
            })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
