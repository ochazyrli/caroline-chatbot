
def get_recommendations(name):
    idx = indices[name]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    name_indices = [i[0] for i in sim_scores]
    return names.iloc[name_indices][0:5]

def get_recommendations_new(input):
    new_rec = data.loc[(data['category'] == input) & (data['rating'] == 4.5) & (data['love'] >= 80000)]
    rec = new_rec['name'].to_numpy()
    return rec[0:5]