import random as random
import numpy as np
class Models():
    #model_num = 0
    def __init__(self, n):
        self.num_models = n

    def generate_models(self, tp_prob, tn_prob):
        model_lst = []
        for _ in range(self.num_models):
            tp = random.uniform(tp_prob, tp_prob + 0.1)
            fn = 1 - tp
            tn = random.uniform(tn_prob, tn_prob + 0.1)
            fp = 1 - tn
            model_metrics = [tp, fn, fp, tn]
            model_lst.append(model_metrics)
        return np.array(model_lst)

    def true_random(self):
        model_lst = []
        for i in range(self.num_models):
            tp = random.random()
            tn = random.random()
            model_lst.append([tp, 1 - tp, 1 - tn, tn])
        return model_lst

    def generate_random(self):
        model_lst = []
        for i in range(self.num_models):
            if i < self.num_models:
                tp = random.uniform(0.4, 0.6)
                tn = random.uniform(0.05, 0.25)
            '''
            else:
                tp = random.uniform(0.05, 0.25)
                tn = random.uniform(0.7, 0.8)
            '''
            model_lst.append([tp, 1- tp, 1 - tn, tn])
            
        return model_lst
            

    def save_models(self, model_lst, file_name):

        np.savetxt(file_name, model_lst, delimiter=',', fmt='%f')
        return

    def get_average_per_metric(self, model_lst):
        tp_avg = fn_avg = tn_avg = fp_avg = 0
        for model_stats in model_lst:
            tp_avg += model_stats[0]
            fn_avg += model_stats[1]
            fp_avg += model_stats[2]
            tn_avg += model_stats[3]
        return [tp_avg/self.num_models, fn_avg/self.num_models, fp_avg/self.num_models, tn_avg/self.num_models]


model_obj = Models(100)
weak_learners = model_obj.generate_models(0.75, 0.45)
model_obj.save_models(weak_learners, "weak_learners")

random_learners = model_obj.generate_random()
for item in random_learners:
    print(item)

model_obj.save_models(random_learners, "random_learners")

r2 = model_obj.true_random()
model_obj.save_models(r2, "true_random_learners")

strong_learners = model_obj.generate_models(0.75, 0.75)
model_obj.save_models(strong_learners, "strong_learners")

for i in range(0, 5):
    print(strong_learners[i])
#average_values = model_obj.get_average_per_metric(weak_learners)
#print("average_values: {}".format(average_values))
#model_obj.save_models(weak_learners, 0)
