import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from node2vec import Node2Vec





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load nodes details
    with open("./fb-pages-food/fb-pages-food.nodes") as f:
        fb_nodes = f.read().splitlines()

        # load edges (or links)
    with open("./fb-pages-food/fb-pages-food.edges") as f:
        fb_links = f.read().splitlines()

    #print(len(fb_nodes), len(fb_links))

    # captture nodes in 2 separate lists
    node_list_1 = []
    node_list_2 = []

    for i in tqdm(fb_links):
        node_list_1.append(i.split(',')[0])
        node_list_2.append(i.split(',')[1])

    fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})
    #print(fb_df)
    # create graph
    G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())

    # combine all nodes in a list
    node_list = node_list_1 + node_list_2

    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))

    # build adjacency matrix
    adj_G = nx.to_numpy_matrix(G, nodelist=node_list)
    #print(adj_G.shape)

    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset, adj_G.shape[1]):
            if i != j:
                if nx.shortest_path_length(G, str(i), str(j)) <= 2:
                    if adj_G[i, j] == 0:
                        all_unconnected_pairs.append([node_list[i], node_list[j]])

        offset = offset + 1

    print("unconnected node pairs= ",len(all_unconnected_pairs))
    # node pairs will act as negative samples during the training of the link prediction model.
    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1': node_1_unlinked,
                         'node_2': node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0

    initial_node_count = len(G.nodes)

    fb_df_temp = fb_df.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(fb_df.index.values):

        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            fb_df_temp = fb_df_temp.drop(index=i)

    print("links that can be drop= ",len(omissible_links_index))

    # create dataframe of removable edges
    fb_df_ghost = fb_df.loc[omissible_links_index]

    # add the target variable 'link'
    fb_df_ghost['link'] = 1

    data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)
    print("Data links= ",data['link'].value_counts())

    # drop removable edges
    fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

    # Generate walks
    node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

    # train node2vec model
    n2w_model = node2vec.fit(window=7, min_count=1)

    compute_features = [(n2w_model[str(i)] + n2w_model[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]
    #split our data into two parts, 70 percent training set and 30 percent testing test
    xtrain, xtest, ytrain, ytest = train_test_split(np.array(compute_features), data['link'],
                                                    test_size=0.3,
                                                    random_state=35)
    # Fitting LogisticRegression classifier to the training set
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(xtrain, ytrain)
    # Predicting the test set result
    predictions = lr.predict_proba(xtest)
    # Test accuracy of the result
    print("Accuracy= ",roc_auc_score(ytest, predictions[:, 1]))

    # plot graph
    plt.figure(figsize=(10, 10))

    pos = nx.random_layout(G, seed=23)
    nx.draw(G, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7)

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
