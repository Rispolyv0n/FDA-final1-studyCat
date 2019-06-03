import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
    
def mean_accs(data, attr):
    """Visualize the distribution of mean accuracies over users.
    Args:
      data: List of dicts, must be filtered just to have accuracies.
      attr: String. The feature of interest - e.g. user, teacher, school.
    """
    accs = {}
    for x in data:
        if x[attr] not in accs.keys():
            accs[x[attr]] = []
        accs[x[attr]].append(x['accuracy'])
    sns.distplot([np.mean(v) for v in accs.values()])
    plt.xlabel('mean accuracy')
    plt.ylabel('density')
    plt.show()



