import keras
import pickle
import numpy as np
from more_itertools import chunked
ensemble_goodModels = []
ensemble_badModels = []
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.float32)

for i in range(1, 12):
    model = keras.models.load_model(f"./Models2/goodModel{i + 7}.keras")
    ensemble_goodModels.append(model)

for i in range(1, 12):
    model = keras.models.load_model(f"./Models2/badModel{i}.keras")
    ensemble_badModels.append(model)

def shuffle_data(X, y):
    p = np.random.permutation(len(X))
    return X[p], y[p]

def generate_batches(X, y, batch_size):
    num_batches = 30
    X, y = shuffle_data(X, y)
    image_batches = list(chunked(X, batch_size))[:num_batches]
    label_batches = list(chunked(y, batch_size))[:num_batches]
    batches =[]
    for i in range(len(image_batches)):
        batches.append((image_batches[i], label_batches[i]))

    f = open("batches3.pickle", "wb")
    pickle.dump(batches, f)
    f.close()
    return batches

def update_weights(prediction_matrix, weights, true_labels):
    total = 11
    for i in range(len(prediction_matrix[0])):
        incorrect = 0
        label = true_labels[i]
        pred_lst = []
        for j in range(len(prediction_matrix)):
            prediction = prediction_matrix[j][i]
            if prediction != label:
                incorrect+=1
            pred_lst.append(prediction)
        
        delta = incorrect / total
        for k in range(len(pred_lst)):
            if pred_lst[k] == label:
                weights[k]+= delta
    
    return weights


def get_weights_for_batch(batch, models, res):
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    results = []
    images = batch[0]
    labels = batch[1]
    images = np.array(images)
    for model in models:
        classificationLst = model.predict(images)
        classificationLst = [(item[0] > 0.5).astype(int).item() for item in classificationLst]
        results.append(classificationLst)
    res.append(results)
    model_weights = update_weights(results, weights, labels)
    return model_weights

def normalize_weights(weights):
    minimum = min(weights)
    maximum = max(weights)

    weights = [((weight - minimum)/(maximum - minimum)) for weight in weights]
    return weights

batch_lst = generate_batches(X, y, 300)

w = open("weights_goodModels3.pickle", "wb")
weights_lst = []
res = []
for batch in batch_lst:
    weights = get_weights_for_batch(batch, ensemble_goodModels, res)
    #weights = normalize_weights(weights)
    weights_lst.append(weights)

pickle.dump(weights_lst, w)
w.close()

z = open("prediction_matrix_goodModels3.pickle", "wb")
pickle.dump(res, z)
z.close()



w = open("weights_badModels3.pickle", "wb")
weights_lst = []
res2 = []
for batch in batch_lst:
    weights = get_weights_for_batch(batch, ensemble_badModels, res2)
    #weights = normalize_weights(weights)
    weights_lst.append(weights)

pickle.dump(weights_lst, w)
w.close()

z = open("prediction_matrix_badModels3.pickle", "wb")
pickle.dump(res2, z)
z.close()
