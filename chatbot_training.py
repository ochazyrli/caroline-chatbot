#import library
from util import JSONParser
import string
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('data_clean.csv')

data = data.reset_index()
names = data['name']
indices = pd.Series(data.index, index=data['name'])

cos_sim = pickle.load(open('cos_sim.pkl', 'rb'))

def get_recommendations_new(input):
    new_rec = data.loc[(data['category'] == input) & (data['rating'] == 4.5) & (data['love'] >= 80000)]
    rec = new_rec['name'].to_numpy()
    return rec[0:5]

def get_recommendations(name):
    idx = indices[name]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    name_indices = [i[0] for i in sim_scores]
    return names.iloc[name_indices][0:5]

def preprocess(chat):
    #konversi ke non capital
    chat = chat.lower()

    #hilangkan tanda baca
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat

def bot_response(chat):
    chat = preprocess(chat)
    res = pipeline.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.2 and len(data[data['name'].str.lower() == chat]) > 0 :
        rekomendasi = get_recommendations(chat.title())
        return 'Carolline merekomendasikan kakak pakai produk ini ', '\n, '.join(rekomendasi.values), None
    elif max_prob < 0.2 and len(data[data['name'].str.lower() == chat]) == 0 :
        return "maaf kak, carolline tidak mengerti", None 
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        return jp.get_response(pred_tag)
        
#load data
path = "intents.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

#preprocess data
# case folding
df['text_input_prep'] = df.text_input.apply(preprocess)

#pemodelan
pipeline = make_pipeline(CountVectorizer(),
                        MultinomialNB())
#train
pipeline.fit(df.text_input_prep, df.intents)

#save model
with open("model_chatbot.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)


if __name__ == "__main__":
    print("[INFO] Anda sudah terhubung dengan Carolline (Beauty Consultant)")
    while True:
        chat = input("Anda >> ")
        res, tag = bot_response(chat)
        print(f"Bot >> {res}")
        if tag == 'Closing':
            break

