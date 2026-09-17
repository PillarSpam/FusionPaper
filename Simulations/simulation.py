import numpy as np
from Simulations.methods import OurMethod
from Simulations.methods import MajorityVoteMethod
from Simulations.methods import WeightedMajorityVote
from Simulations.methods import TopKMethod
import matplotlib.pyplot as plt

class Simulation():
    def __init__(self, num_models, model_lst_path, threshold):
        self.num_models = num_models
        self.threshold = threshold
        self.data_lst = self.generate_data(1000, 300)
        self.weighted_majority_vote = WeightedMajorityVote(model_lst_path, self.data_lst)
        self.our_method = OurMethod(model_lst_path, self.data_lst, self.threshold)
        self.majority_vote = MajorityVoteMethod(model_lst_path, self.data_lst)
        self.top_k_method = TopKMethod(model_lst_path, self.data_lst, 10)

    
    def generate_data(self, num_batches, num_tasks_per_batch):
        data = []
        #Each x point is a batch
        for batch_num in range(num_batches):
            #For each batch, generate the designated number of tasks per batch
            batch_lst = []
            for task_num in range(num_tasks_per_batch):
                task_label = task_num % 2
                
                #label 0 means True, label 1 means False
                batch_lst.append(task_label)
            data.append(batch_lst)
        return data

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


    def get_metric_lsts(self, method_obj):
        batch_metrics = method_obj.metric_lst
        accuracy_lst = [lst[0] for lst in batch_metrics]
        precision_lst = [lst[1] for lst in batch_metrics]
        recall_lst = [lst[2] for lst in batch_metrics]
        f1_lst = [lst[3] for lst in batch_metrics]
        return [accuracy_lst, precision_lst, recall_lst, f1_lst]

        

    def graph_metrics(self, graph_name):
        m3 = self.get_metric_lsts(self.weighted_majority_vote)
        
        m1 = self.get_metric_lsts(self.our_method)
        #print(len(m1[0]))
        m2 = self.get_metric_lsts(self.majority_vote)
        m4 = self.get_metric_lsts(self.top_k_method)
        
        x_points = [i for i in range(len(m1[0]))]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
        file_name = "performance_improvement.txt"
        for i in range(4):
            print(i)
            improvment_percentages = self.get_ips([m1[i], m2[i], m3[i], m4[i]])
            file = open(file_name, "a")
            file.write(graph_name + "\n")
            file.write(metric_labels[i] + "\n")
            file.write("Our Method vs MV: " + str(improvment_percentages[0]) + "\n")
            file.write("Our Method vs WMV: " + str(improvment_percentages[1]) + "\n")
            file.write("Our Method vs TopK: " + str(improvment_percentages[2]) + "\n")
            file.write("\n")
            file.close()
            plt.plot(x_points, m1[i], label="Our Method")
            plt.plot(x_points, m2[i], label="MV")
            plt.plot(x_points, m3[i], label="WMV")
            plt.plot(x_points, m4[i], label="Top K")
            plt.xlabel('Batch')
            plt.ylabel(metric_labels[i])
            plt.legend(loc="upper right")
            plt.savefig("{}_{}.png".format(graph_name, i), format="png")
            plt.savefig("{}_{}.eps".format(graph_name, i), format="eps")
            plt.close()


#simulation = Simulation(100, "weak_learners", 0.7)
sim1 = Simulation(100, "random_learners", 0.45)
sim2 = Simulation(100, "weak_learners", 0.65)
sim3 = Simulation(100, "strong_learners", 0.52)

'''
for i, batch in enumerate(simulation.data_lst):
    print("batch_num:{} | batch_lst:{} | class_lst:{}".format(i, batch, simulation.our_method.classification_lst[i]))
'''

sim1.graph_metrics("random_learners")
sim2.graph_metrics("weak_learners")
sim3.graph_metrics("strong_learners")