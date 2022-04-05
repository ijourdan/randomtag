#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS


# Generación del corpus
class Data_Gen():
    def __init__(self, data, tags):
        self.data = data
        self.targets = tags
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, indx):
        return self.data[indx], self.targets[indx]

def corpus_generator(quant_tag = 4, quant_data = 1000, dimension = 2, test_proportion = 0.2, gap=4):
    """

    :param quant_tag: 4 cantidad de etiquetas
    :param quant_data: 1000 cantidad de datos del corpus
    :param dimension: 2 dimensiones del problema
    :param gap: 4 separación entre clusters
    :param test_proportion:
    :return: training, training_tags, test, test_tags
    """

    centers = torch.randn(size=(quant_tag,dimension))*gap
    data = torch.randn(size=( quant_data,dimension))
    data_tags = torch.randint(low=0, high=quant_tag, size=(quant_data,))

    for i in range(quant_data):
        data[i,:] += centers[data_tags[i].item(),:]

    limit_index_for_test = torch.floor(torch.tensor((1-test_proportion)*quant_data)).long().item()
    training = data[:limit_index_for_test,:]
    training_tags = data_tags[:limit_index_for_test]
    test = data[limit_index_for_test:,:]
    test_tags = data_tags[limit_index_for_test:]

    return training, training_tags, test, test_tags

def shuffle_tags(tags, proba=0.0):
    """
    Cambia los labels de tags_in con una probabilidad dada por proba
    :param tags_in: tensor de tags (labels)
    :param proba: probabilidad de cambiar un label
    :return:
    """
    # Se toman dos elementos, uno en línea, y el otro se elige al azar.
    # Si resulta que rand < p entonces se intercambia la calse con el elegido aleatoriamente. .

    new_tag = (torch.zeros(size=tags.size())-1).long()
    index = torch.randperm(len(new_tag))
    eligible_index = new_tag == -1
    i = 0
    while eligible_index.sum() > 1:
        if torch.rand((1,)).item() < proba:  # se intercambian
            new_tag[index[2 * i]] = tags[index[2 * i + 1]]
            new_tag[index[2 * i + 1]] = tags[index[ 2 * i]]
        else: # no se intercambian
            new_tag[index[2 * i]] = tags[index[2 * i]]
            new_tag[index[2 * i + 1]] = tags[index[2 * i + 1]]
        eligible_index = new_tag == -1
        i += 1

    if eligible_index.sum() > 1:
        new_tag[eligible_index] = tags[eligible_index]
    return new_tag


def plt_data_2d(data,data_tags):
    keys = list(TABLEAU_COLORS.keys())
    fig = plt.figure(figsize=(9,6))
    for i in range(data_tags.max().item()+1):
        index = data_tags == i
        plt.plot(data[index,0],data[index,1], '*' ,color=TABLEAU_COLORS[keys[i]])
    plt.show()
    
# Scheduler not used
def lr_schedule(epoch):
    """
    Modificamos el learning rate para que no sea tan rápido cuando el aprendizaje está avanzado, pero que no sea lento
    en los estadíos iniciales.
    :param epoch:
    :return: learning rate
    """
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

# Neural Network definition

class Net(nn.Module):
    def __init__(self, num_classes=4, num_input=2):
        super(Net, self).__init__()
        self.fc3 = nn.Linear(num_input,128)
        self.fc2 = nn.Linear(128,84)
        self.fc =  nn.Linear(84,num_classes)
    def forward(self, x):
        out = self.fc3(x)
        out = self.fc2(out)
        out = self.fc(out)
        return out







