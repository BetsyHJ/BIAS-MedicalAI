import matplotlib.pyplot as plt
import numpy as np

def read_out(path):
    fp = open(path, 'r')
    indices, auc_results, auc_masked_results, deltas = [], [], [], []
    for line in fp.readlines():
        s = line.strip().split()
        if (len(s) == 4) and (s[0].isdigit()):
            idx, auc, auc_masked, delta = int(s[0]), float(s[1]), float(s[2]), float(s[3])
            assert idx == len(indices)
            indices.append(idx)
            auc_results.append(auc)
            auc_masked_results.append(auc_masked)
            deltas.append(delta)
    fp.close()
    return {'idx': indices, 'auc': auc_results, 'auc_masked': auc_masked_results, 'delta': deltas}

def his_plot(deltas, bins=100):
    plt.hist(deltas, bins=bins)
    plt.xlabel('delta = auc - auc_masked')
    plt.ylabel('number')
    plt.savefig("generate_plot1.pdf", format='pdf', bbox_inches="tight")
    plt.show()

# path = "out/resnet_1221_NIH_random_lr1e-4_rot20_analysis.out" # on 5k image
# results_dict = read_out(path)
# a1 = np.array(results_dict['delta']).argsort()

# path = "out/resnet_1221_NIH_random_lr1e-4_rot20_analysis3.out" # on 1k images
path = "out/resnet_1221_NIH_random_lr1e-4_rot20_analysis_kmeans_k100_Ate.out" # on 1k images
results_dict = read_out(path)
a2 = np.array(results_dict['delta']).argsort()
a2 = a2[::-1]
print(results_dict['delta'][a2[0]], a2[0])
# print(", ".join([str(x) for x in a2[-10:]]))

# print(set(a1[-50:]) & set(a2[-50:]))

his_plot(results_dict['delta'], bins=50)