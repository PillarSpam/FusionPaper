import matplotlib as mpl 
import pickle
import numpy as np
from more_itertools import chunked
import keras
import matplotlib.pyplot as plt
from Experiments.graph import Graph



class Experiment():
    
    def __init__(self, type, batch_data_fn, weights_fn, pred_matr_fn, graph_name):
        self.exp_type = type
        self.batch_lst = self.read_pickle_data(batch_data_fn)
        self.num_batches = len(self.batch_lst)
        self.weights = self.read_pickle_data(weights_fn)
        self.prediction_matrix = self.read_pickle_data(pred_matr_fn)
        self.graph_name = graph_name
        self.top_weak_models = self.get_top_models("matrices_badModels.txt", 3, 1)
        self.top_strong_models = self.get_top_models("matrices_goodModels.txt", 3, 0)
    
    def read_pickle_data(self, file_name):
        file = open(file_name, "rb")
        loaded_data = pickle.load(file)
        file.close()
        return loaded_data[:30]
    
    def get_top_models(self, fname, num_models, type):
        f = open(fname, "r")
        model_metrics = f.readlines()
        if type == 1:
            metric_lst = [item.strip().split(' ') for item in model_metrics]
        elif type == 0:
            metric_lst = [item.strip().split(' ') for item in model_metrics][7:]

        for i in range(len(metric_lst)):
            metric_lst[i] = (i + 1, metric_lst[i])
            sorted_lst = sorted(metric_lst, key=lambda x: int(x[1][0]) + int(x[1][-1]))
            sorted_lst.reverse()
        return sorted_lst[:num_models]
    
    #takes batch in, produces final confusion matrix of entire ensemble (all models in group)
    def get_batch_metrics(self, batch_prediction_lst, threshold, index, method, model_weights):
    
        classed_MV = []
        classed_WMV = []
        labels = self.batch_lst[index][1]
        cutoff = len(batch_prediction_lst) * threshold
        for i in range(len(batch_prediction_lst[0])): #number of tasks in batch
            sum = 0
            weighted_sum_positive = 0
            weighted_sum_negative = 0
            int0 = 0
            int1 = 0
            for j in range(len(batch_prediction_lst)): #number of models (11 in this case)
                pred = batch_prediction_lst[j][i]
                if pred == 0:
                    weighted_sum_positive += model_weights[index][j]
                else:
                    weighted_sum_negative += model_weights[index][j]
                
                if pred == 1:
                    int1 += 1
                elif pred == 0:
                    int0 +=1
            
            if (weighted_sum_positive > weighted_sum_negative):
                classed_WMV.append(0)
            else:
                classed_WMV.append(1)

            
            if (int0 >= cutoff):
                classed_MV.append(0)
                
            else:
                classed_MV.append(1)
            
        
        classed = classed_MV
        if (method == 1):
            classed = classed_WMV

        TP = FN = FP = TN = 0
        for i in range(len(classed)):
            res = classed[i]
            label = labels[i]

            if (res == label and label == 0):
                TP += 1
            elif (res == label and label == 1):
                TN += 1
            elif (res == 1 and label == 0):
                FN += 1
            else:
                FP += 1

        return TP, FN, FP, TN
    
    def get_pred_matrix_top_models(self, prediction_matrix):
        if self.exp_type == 0:
            top_model_indexes = [item[0]-1 for item in self.top_strong_models]
        elif self.exp_type == 1:
            top_model_indexes = [item[0]-1 for item in self.top_weak_models]

        prediction_lst_top_k = []
        for i in range(len(self.prediction_matrix)):
            lst = []
            for j in range(11):
                if j in top_model_indexes:
                    lst.append(self.prediction_matrix[i][j])
            prediction_lst_top_k.append(lst)
        return prediction_lst_top_k

    def generate_metric_lsts(self, full_models, threshold, method, model_weights):
        acc_lst = []
        prec_lst = []
        recall_lst = []
        f1_lst = []
        if full_models:
            prediction_lst = self.prediction_matrix
        else:
            prediction_lst = self.get_pred_matrix_top_models(self.prediction_matrix)

        for i in range(len(prediction_lst)):
            TP, FN, FP, TN = self.get_batch_metrics(prediction_lst[i], threshold, i, method, model_weights)
            accuracy = 0
            if (TP != 0 or TN != 0 or FP != 0 or FN != 0):
                accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = 0
            if(TP != 0 or FP != 0):
                precision = TP / (TP + FP)
            recall = 0
            if (TP != 0 or FN != 0):
                recall = TP / (TP + FN)
            f1 = 0
            if (precision != 0 or recall != 0):
                f1 = (2 * precision * recall) / (precision + recall)

            acc_lst.append(accuracy)
            prec_lst.append(precision)
            recall_lst.append(recall)
            f1_lst.append(f1)
            
        return acc_lst, prec_lst, recall_lst, f1_lst
    
    def get_ips(self, metrics_lst):
        our_method = metrics_lst[0]
        MV = metrics_lst[1]
        WMV = metrics_lst[2]
        top_k = metrics_lst[3]


        mv_imp = wmv_imp = top_k_imp = 0
        for i in range(len(our_method)):
            our_val = our_method[i]
            mv_imp += (our_val - MV[i]) / MV[i]
            wmv_imp += (our_val - WMV[i]) / WMV[i]
            top_k_imp += (our_val - top_k[i]) / top_k[i]

        mv_imp /= len(our_method)
        wmv_imp /= len(our_method)
        top_k_imp /= len(our_method)
        return [mv_imp * 100, wmv_imp * 100, top_k_imp * 100]    

    def write_ips(self, metrics_lst, metric_label):
        file_name = "performance_improvement.txt"
        improvment_percentages = self.get_ips(metrics_lst)
        file = open(file_name, "a")
        file.write(self.graph_name + "\n")
        file.write(metric_label + "\n")
        file.write("Our Method vs MV: " + str(improvment_percentages[0]) + "\n")
        file.write("Our Method vs WMV: " + str(improvment_percentages[1]) + "\n")
        file.write("Our Method vs TopK: " + str(improvment_percentages[2]) + "\n")
        file.write("\n")
        file.close()

    def make_graphs(self, cutoffs, count=0):
        x = [i for i in range(self.num_batches)]
        print(len(x))
        a1, p1, r1, f1 = self.generate_metric_lsts(True, 0.5, 0, self.weights)
        a2, p2, r2, f2 = self.generate_metric_lsts(False, 0.5, 0, self.weights)
        a3, _, _, _ = self.generate_metric_lsts(True, cutoffs[0], 0,self.weights)
        _, p3, _, _ = self.generate_metric_lsts(True, cutoffs[1], 0, self.weights)
        _, _, r3, _ = self.generate_metric_lsts(True, cutoffs[2], 0, self.weights)
        _, _, _, f3 = self.generate_metric_lsts(True, cutoffs[3], 0,self.weights)
        a4, p4, r4, f4 = self.generate_metric_lsts(True, 0.5, 1, self.weights)

        if self.exp_type == 1:
            ident = "WeakLearner"
        elif self.exp_type == 0:
            ident = "StrongLearner"
        

        graph1 = Graph(x, [a1, a2, a3, a4], "Accuracy", "Experiment_{}_{}_Acc_LowSense".format(count, ident))
        self.write_ips([a3, a1, a4, a2], "Accuracy")
        graph2 = Graph(x, [p1, p2, p3, p4], "Precision", "Experiment_{}_{}_Prec_LowSense".format(count, ident))
        self.write_ips([p3, p1, p4, p2], "Precision")
        graph3 = Graph(x, [r1, r2, r3, r4], "Recall", "Experiment_{}_{}_Rec_LowSense".format(count, ident))
        self.write_ips([r3, r1, r4, r2], "Recall")
        graph4 = Graph(x, [f1, f2, f3, f4], "F1 Score", "Experiment_{}_{}_F1_LowSense".format(count, ident))
        self.write_ips([f3, f1, f4, f2], "F1")


exp = Experiment(1, "batches2.pickle", "weights_badModels2.pickle", "prediction_matrix_badModels2.pickle", "weak_learners")
exp2 = Experiment(0, "batches3.pickle", "weights_goodModels3.pickle", "prediction_matrix_goodModels3.pickle", "strong_learners")
step = 0.04
t = 0.2
c = 0

exp.make_graphs([0.45, 0.7, 0.4, 0.45], c)
exp2.make_graphs([0.6, 0.7, 0.4, 0.6], c)
    



