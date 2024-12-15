# import pickle
# import numpy as np

# def load_model():
#     with open("best_extra_trees_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     return model

# def make_prediction(model, features):
#     features = np.array([features])
#     prediction = model.predict(features)[0]
#     return prediction