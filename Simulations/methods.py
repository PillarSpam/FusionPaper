import numpy as np
import random as random
#Each class:
#Takes in the model_lst file, and the data_lst object
#model_lst_file: tells which models to use when performing classification
#data_lst: is the batch lst used for testing each method. Each element corresponds to a batch,
#and each batch has x number of task labels (either 0 for true or 1 for false)

class OurMethod():
    def __init__(self, model_lst_file, data_lst, threshold):
        self.file_name = model_lst_file
        self.threshold = threshold
        self.model_data = self.retrieve_model_data()
        self.data_lst = data_lst
        self.classification_lst = self.get_classification_lst()
        self.metric_lst = self.get_metrics()

    def retrieve_model_data(self):
        model_data = np.loadtxt(self.file_name, delimiter=',', dtype=float)
        return model_data

    def get_predictions(self, task_label):
        prediction_lst = []
        for model in self.model_data:
            tp = model[0]
            fn = model[1]
            fp = model[2]
            tn = model[3]
            if task_label == 0:
                rand_val = random.random()
                if rand_val <= tp:
                    prediction_lst.append(0)
                else:
                    prediction_lst.append(1)
            
            elif task_label == 1:
                rand_val = random.random()
                if rand_val <= tn:
                    prediction_lst.append(1)
                else:
                    prediction_lst.append(0)
        return prediction_lst

    def make_classification(self, prediction_lst):
        int0 = prediction_lst.count(0)
        total = len(prediction_lst)
        
        if int0 / total >= self.threshold:
            return 0
        else:
            return 1

    def get_classification_lst(self):
        classification_lst = []
        for batch in self.data_lst:
            batch_classification = []
            for task_label in batch:
                #make a prediction on this task for each model
                prediction_lst_for_task = self.get_predictions(task_label)
                #print("task_label:{} | {}".format(task_label, prediction_lst_for_task))
                #print(len(prediction_lst_for_task))
                classification = self.make_classification(prediction_lst_for_task)
                batch_classification.append(classification)

            classification_lst.append(batch_classification)
        return classification_lst

    def get_metrics(self):
        metric_lst = []
        for batch_ind in range(len(self.data_lst)):
            batch_lst = self.data_lst[batch_ind]
            classification_lst = self.classification_lst[batch_ind]
            tp = fn = fp = tn = 0
            for task_ind in range(len(batch_lst)):
                task_label = batch_lst[task_ind]
                prediction = classification_lst[task_ind]
                if task_label == 0 and prediction == 0:
                    tp += 1
                elif task_label == 0 and prediction == 1:
                    fn += 1
                elif task_label == 1 and prediction == 0:
                    fp += 1
                elif task_label == 1 and prediction == 1:
                    tn += 1
           
            accuracy = 0
            if (tp != 0 or tn != 0 or fp != 0 or fn != 0):
                accuracy = (tn + tp) / (tp + tn + fp + fn)
            precision = 0
            if(tp != 0 or fp != 0):
                precision = tp / (tp + fp)
            recall = 0
            if (tp != 0 or fn != 0):
                recall = tp / (tp + fn)
            f1 = 0
            if (precision != 0 or recall != 0):
                f1 = (2 * precision * recall) / (precision + recall)
            metric_lst.append([accuracy, precision, recall, f1])
        return metric_lst


class MajorityVoteMethod():
    def __init__(self, model_lst_file, data_lst):
        self.file_name = model_lst_file
        self.threshold = 0.5
        self.model_data = self.retrieve_model_data()
        self.data_lst = data_lst
        self.classification_lst = self.get_classification_lst()
        self.metric_lst = self.get_metrics()

    def retrieve_model_data(self):
        model_data = np.loadtxt(self.file_name, delimiter=',', dtype=float)
        return model_data

    def get_predictions(self, task_label):
        prediction_lst = []
        for model in self.model_data:
            tp = model[0]
            fn = model[1]
            fp = model[2]
            tn = model[3]
            #print("tp:{} | tn:{}".format(tp, tn))
            if task_label == 0:
                rand_val = random.uniform(0, 1)
                if rand_val < tp:
                    prediction_lst.append(0)
                else:
                    prediction_lst.append(1)
            
            elif task_label == 1:
                rand_val = random.uniform(0, 1)
                if rand_val < tn:
                    prediction_lst.append(1)
                else:
                    prediction_lst.append(0)
        return prediction_lst

    def make_classification(self, prediction_lst):
        num_positives = 0
        num_negatives = 0
        for prediction in prediction_lst:
            if prediction == 0:
                num_positives+=1
            elif prediction == 1:
                num_negatives+=1
        
        
        if num_negatives > num_positives:
            return 1
        else:
            return 0

    def get_classification_lst(self):
        classification_lst = []
        for batch in self.data_lst:
            batch_classification = []
            for task_label in batch:
                #make a prediction on this task for each model
                prediction_lst_for_task = self.get_predictions(task_label)
                #print("task_label:{} | {}".format(task_label, prediction_lst_for_task))
                #print(len(prediction_lst_for_task))
                classification = self.make_classification(prediction_lst_for_task)
                batch_classification.append(classification)

            classification_lst.append(batch_classification)
        return classification_lst

    def get_metrics(self):
        metric_lst = []
        for batch_ind in range(len(self.data_lst)):
            batch_lst = self.data_lst[batch_ind]
            classification_lst = self.classification_lst[batch_ind]
            tp = fn = fp = tn = 0
            for task_ind in range(len(batch_lst)):
                task_label = batch_lst[task_ind]
                prediction = classification_lst[task_ind]
                if task_label == 0 and prediction == 0:
                    tp += 1
                elif task_label == 0 and prediction == 1:
                    fn += 1
                elif task_label == 1 and prediction == 0:
                    fp += 1
                elif task_label == 1 and prediction == 1:
                    tn += 1
            accuracy = 0
            if (tp != 0 or tn != 0 or fp != 0 or fn != 0):
                accuracy = (tn + tp) / (tp + tn + fp + fn)
            precision = 0
            if(tp != 0 or fp != 0):
                precision = tp / (tp + fp)
            recall = 0
            if (tp != 0 or fn != 0):
                recall = tp / (tp + fn)
            f1 = 0
            if (precision != 0 or recall != 0):
                f1 = (2 * precision * recall) / (precision + recall)
            metric_lst.append([accuracy, precision, recall, f1])
        return metric_lst

class WeightedMajorityVote():
    def __init__(self, model_lst_file, data_lst):
        self.file_name = model_lst_file
        self.threshold = 0.5
        self.model_data = self.retrieve_model_data()
        self.data_lst = data_lst
        self.predcition_lst = self.get_prediction_lst()
        #print(self.predcition_lst)
        self.weights = self.get_weights(self.predcition_lst)
        self.classification_lst = self.get_classification_lst(self.predcition_lst)
        self.metric_lst = self.get_metrics()

    def get_weights_for_batch(self, batch):
        temp_b = [(i % 2, batch[i]) for i in range(len(batch))]
        num_models = len(self.model_data)
        weights = [1 for _ in range(num_models)]
        for task in temp_b:
            #print(task)
            incorrect = 0
            label = task[0]
            model_predictions = task[1]
            
            for prediciton in model_predictions:
                if label != prediciton:
                    incorrect+=1
            
            delta = incorrect / len(model_predictions)
            for i in range(len(weights)):
                if model_predictions[i] == label:
                    weights[i] += delta
            
        return weights
    
    def get_weights(self, prediction_matrix):
        weights_lst = []
        
        for batch in prediction_matrix:
            batch_weights = self.get_weights_for_batch(batch)
            weights_lst.append(batch_weights)
        return weights_lst

    def retrieve_model_data(self):
        model_data = np.loadtxt(self.file_name, delimiter=',', dtype=float)
        return model_data

    def get_predictions(self, task_label):
        prediction_lst = []
        for model in self.model_data:
            tp = model[0]
            fn = model[1]
            fp = model[2]
            tn = model[3]
            #print("tp:{} | tn:{}".format(tp, tn))
            if task_label == 0:
                rand_val = random.uniform(0, 1)
                if rand_val < tp:
                    prediction_lst.append(0)
                else:
                    prediction_lst.append(1)
            
            elif task_label == 1:
                rand_val = random.uniform(0, 1)
                if rand_val < tn:
                    prediction_lst.append(1)
                else:
                    prediction_lst.append(0)
        return prediction_lst

    def make_classification(self, prediction_lst, index):
        weighted_sum_positive = 0
        weighted_sum_negative = 0
        for i, prediction in enumerate(prediction_lst):
            if prediction == 0:
                weighted_sum_positive += self.weights[index][i]
            elif prediction == 1:
                weighted_sum_negative += self.weights[index][i]
        
        
        if weighted_sum_positive > weighted_sum_negative:
            return 0
        else:
            return 1

    def get_prediction_lst(self):
        prediction_lst = []
        for batch in self.data_lst:
            batch_classification = []
            for task_label in batch:
                #make a prediction on this task for each model
                prediction_lst_for_task = self.get_predictions(task_label)
                batch_classification.append(prediction_lst_for_task)
            prediction_lst.append(batch_classification)
        return prediction_lst

    def get_classification_lst(self, prediction_lst):
        classification_lst = []
        for index, batch in enumerate(prediction_lst):
            batch_classification = []
            
            for i, prediction_lst_for_task in enumerate(batch):
                task_label = i % 2
                #print("task_label:{} | {}".format(task_label, prediction_lst_for_task))
                #print(len(prediction_lst_for_task))
                classification = self.make_classification(prediction_lst_for_task, index)
                batch_classification.append(classification)

            classification_lst.append(batch_classification)
        return classification_lst

    def get_metrics(self):
        metric_lst = []
        for batch_ind in range(len(self.data_lst)):
            batch_lst = self.data_lst[batch_ind]
            classification_lst = self.classification_lst[batch_ind]
            tp = fn = fp = tn = 0
            for task_ind in range(len(batch_lst)):
                task_label = batch_lst[task_ind]
                prediction = classification_lst[task_ind]
                if task_label == 0 and prediction == 0:
                    tp += 1
                elif task_label == 0 and prediction == 1:
                    fn += 1
                elif task_label == 1 and prediction == 0:
                    fp += 1
                elif task_label == 1 and prediction == 1:
                    tn += 1
            accuracy = 0
            if (tp != 0 or tn != 0 or fp != 0 or fn != 0):
                accuracy = (tn + tp) / (tp + tn + fp + fn)
            precision = 0
            if(tp != 0 or fp != 0):
                precision = tp / (tp + fp)
            recall = 0
            if (tp != 0 or fn != 0):
                recall = tp / (tp + fn)
            f1 = 0
            if (precision != 0 or recall != 0):
                f1 = (2 * precision * recall) / (precision + recall)
            metric_lst.append([accuracy, precision, recall, f1])
        return metric_lst


class TopKMethod():
    def __init__(self, model_lst_file, data_lst, top_k):
        self.file_name = model_lst_file
        self.threshold = 0.5
        self.top_k = top_k
        self.model_data = self.retrieve_model_data()
        self.model_data = self.get_top_models(self.model_data)
        self.data_lst = data_lst
        self.classification_lst = self.get_classification_lst()
        self.metric_lst = self.get_metrics()

    def retrieve_model_data(self):
        model_data = np.loadtxt(self.file_name, delimiter=',', dtype=float)
        return model_data

    def get_top_models(self, model_data):
        model_lst = []
        for i in range(len(model_data)):
            prob = (model_data[i][0] + model_data[i][-1]) / 2
            model_lst.append((prob, model_data[i]))

        model_lst = sorted(model_lst, key=lambda a: a[0], reverse=True)
        top = [entry[1] for entry in model_lst[:self.top_k]]
        return top

    def get_predictions(self, task_label):
        prediction_lst = []
        for model in self.model_data:
            tp = model[0]
            fn = model[1]
            fp = model[2]
            tn = model[3]
            #print("tp:{} | tn:{}".format(tp, tn))
            if task_label == 0:
                rand_val = random.uniform(0, 1)
                if rand_val < tp:
                    prediction_lst.append(0)
                else:
                    prediction_lst.append(1)
            
            elif task_label == 1:
                rand_val = random.uniform(0, 1)
                if rand_val < tn:
                    prediction_lst.append(1)
                else:
                    prediction_lst.append(0)
        return prediction_lst

    def make_classification(self, prediction_lst):
        num_positives = 0
        num_negatives = 0
        for prediction in prediction_lst:
            if prediction == 0:
                num_positives+=1
            elif prediction == 1:
                num_negatives+=1
        
        
        if num_negatives > num_positives:
            return 1
        else:
            return 0

    def get_classification_lst(self):
        classification_lst = []
        for batch in self.data_lst:
            batch_classification = []
            for task_label in batch:
                #make a prediction on this task for each model
                prediction_lst_for_task = self.get_predictions(task_label)
                #print("task_label:{} | {}".format(task_label, prediction_lst_for_task))
                #print(len(prediction_lst_for_task))
                classification = self.make_classification(prediction_lst_for_task)
                batch_classification.append(classification)

            classification_lst.append(batch_classification)
        return classification_lst

    def get_metrics(self):
        metric_lst = []
        for batch_ind in range(len(self.data_lst)):
            batch_lst = self.data_lst[batch_ind]
            classification_lst = self.classification_lst[batch_ind]
            tp = fn = fp = tn = 0
            for task_ind in range(len(batch_lst)):
                task_label = batch_lst[task_ind]
                prediction = classification_lst[task_ind]
                if task_label == 0 and prediction == 0:
                    tp += 1
                elif task_label == 0 and prediction == 1:
                    fn += 1
                elif task_label == 1 and prediction == 0:
                    fp += 1
                elif task_label == 1 and prediction == 1:
                    tn += 1
            accuracy = 0
            if (tp != 0 or tn != 0 or fp != 0 or fn != 0):
                accuracy = (tn + tp) / (tp + tn + fp + fn)
            precision = 0
            if(tp != 0 or fp != 0):
                precision = tp / (tp + fp)
            recall = 0
            if (tp != 0 or fn != 0):
                recall = tp / (tp + fn)
            f1 = 0
            if (precision != 0 or recall != 0):
                f1 = (2 * precision * recall) / (precision + recall)
            metric_lst.append([accuracy, precision, recall, f1])
        return metric_lst