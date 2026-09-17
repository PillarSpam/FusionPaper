import matplotlib.pyplot as plt

class Graph():
    def __init__(self, x, line_data, metric_name, graph_name):
        self.x = x
        self.mv, self.top_k, self.om, self.wmv = line_data
        self.metric_name = metric_name
        self.graph_name = graph_name
        self.create_graph()

    def create_graph(self):
        plt.plot(self.x, self.om, label="Our Method")
        plt.plot(self.x, self.mv, label="MV")
        plt.plot(self.x, self.wmv, label="WMV")
        plt.plot(self.x, self.top_k, label="Top K")
        plt.xlabel("Batch")
        plt.ylabel(self.metric_name)
        plt.legend(loc='upper right')
        #plt.show()
        plt.savefig(self.graph_name + ".png", format='png')
        plt.savefig(self.graph_name + ".eps", format='eps')
        plt.cla()