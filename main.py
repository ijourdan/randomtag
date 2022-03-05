import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS


#%%
# Generación del corpus
def corpus_generator(quant_tag = 4, quant_data = 1000, dimension = 2, test_proportion = 0.2):
    """

    :param quant_tag: 4 cantidad de etiquetas
    :param quant_data: 1000 cantidad de datos del corpus
    :param dimension: 2 dimensiones del problema
    :param test_proportion:
    :return: training, training_tags, test, test_tags
    """

    centers = torch.randn(size=(dimension,quant_tag))*2
    data = torch.randn(size=(dimension, quant_data))
    data_tags = torch.randint(low=0, high=quant_tag, size=(1,quant_data))

    for i in range(quant_data):
        data[:,i] += centers[:,data_tags[0,i].item()]

    limit_index_for_test = torch.floor(torch.tensor((1-test_proportion)*quant_data)).int().item()
    training = data[:,:limit_index_for_test]
    training_tags = data_tags[:,:limit_index_for_test]
    test = data[:,limit_index_for_test:]
    test_tags = data_tags[:,limit_index_for_test:]

    return training, training_tags, test, test_tags

def plt_data_2d(data,data_tags):
    keys = list(TABLEAU_COLORS.keys())
    # fig = plt.figure(figsize=(12,8))
    for i in range(data_tags.max().item()+1):
        index = data_tags == i
        plt.plot(data[:,index[0]][0],data[:,index[0]][1], '*' ,color=TABLEAU_COLORS[keys[i]])


#%%

# Vamos a generar dos modelos:
# 1-  plantea una probabilidad de error de asignacion de tags
# 2- plantea que los tags son distribuciones de probabilidades,
#    y el valor del tag es una realización de la distribucion
#    de probabilidad.


training, training_tags, test, tags = corpus_generator()
# 1.
tag_list = [ i for i in range(tags.max().item()+1)]
p = 0.1  # probabilidad que cambie el índice.
aux = torch.rand(size=tags.size()) < p  # Los que cambian







#%%

