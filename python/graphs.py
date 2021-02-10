import matplotlib.pyplot as plt
import numpy as np

def similarity_results_grpah(tab_results_base, tab_results_improvement, project_folder, global_metric):
    ind = np.arange(len(tab_results_base)) # the x locations for the groups
    width=0.3
    plt.bar(ind, tab_results_base, width, color='y', label='Shortest path')
    plt.bar(ind, tab_results_improvement, width,bottom=tab_results_base, color='g', label="Modified path")
    x = np.arange(len(tab_results_base))
    plt.xticks(x, ["Mapbox", "Clusters", "Global", "Clusters + NN"])
    plt.legend(loc='upper right')
    plt.ylabel('Similarity between the path generated and the observation (%)')
    plt.yticks(np.arange(0, 110, step=10))

    if(global_metric):
        plt.savefig("files/"+project_folder+"/images/similarity_results_global.png")
    else:
        plt.savefig("files/"+project_folder+"/images/similarity_results.png")


def NN_results_graph(tab_results_NN, project_folder, global_metric):   
    plt.figure()
    ind = np.arange(len(tab_results_NN[0])) # the x locations for the groups
    width=0.3
    plt.bar(ind, tab_results_NN[0], width, color='y', label='Shortest path')
    plt.bar(ind, tab_results_NN[1], width,bottom=tab_results_NN[0], color='g', label="Modified path")
    x = np.arange(len(tab_results_NN[0]))
    plt.legend(loc='upper right')
    plt.xticks(x, ["Good predictions", "Bad predictions", "Total"])
    plt.ylabel('Similarity between the path generated and the observation (%)')
    plt.yticks(np.arange(0, 110, step=10))
    if(global_metric):
        plt.savefig("files/"+project_folder+"/images/NN_results_global.png")
    else:
        plt.savefig("files/"+project_folder+"/images/NN_results.png")