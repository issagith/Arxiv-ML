import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import pandas as pd

def nonDirected_graph(file, filter_weight=None, samples=None):
    ''' Create a Graph from
    file: collaboration csv with 4 columns: author1, author2, weight, category
    filter_weight: if we want to display only authors that have more than filter_weight collaborations
    samples: if the csv is too heavy, use only some samples taken randomly from the dataset
    '''
    df = pd.read_csv(file)

    # eliminate lines which weight isn't relevant
    if filter_weight is not None:
        df_filtered = df[df["weight"] > filter_weight]
        df = df_filtered
    
    # take randomly n samples
    if samples is not None:
        df_sample = df.sample(n=samples, random_state=42)  # random_state for reproducibility
        df = df_sample

    # Create non-directed graph
    G = nx.Graph()

    # Add weight and category as attributes to the edges
    for _, row in df.iterrows():
        G.add_edge(row["author1"], row["author2"], weight=row["weight"], category=row["category"])
        
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    return G
G_science= nonDirected_graph("science_collaboration_cat&wight.csv", filter_weight=0)

def plot_communities_trend(G, min_res=0.1, max_res=2.0, step=0.1):
    """
    Trace la courbe du nombre de communautés détectées par Louvain
    en fonction de la résolution γ allant de min_res à max_res (inclus),
    avec un pas de step.

    Parameters
    ----------
    G : networkx.Graph
        Le graphe sur lequel lancer Louvain.
    min_res : float
        Résolution de départ (par ex. 0.1).
    max_res : float
        Résolution de fin (par ex. 2.0).
    step : float
        Pas entre chaque valeur de résolution (par ex. 0.1).

    Returns
    -------
    dict
        Dictionnaire {résolution: nombre_de_communautés}.
    """
    # Génère la liste de résolutions
    n_steps = int(round((max_res - min_res) / step)) + 1
    resolutions = resolutions = [0.2, 0.6, 1, 1.5, 2]

    # Calcule le nombre de communautés pour chaque résolution
    res_to_count = {}
    for γ in resolutions:
        partition = community_louvain.best_partition(G, resolution=γ)
        res_to_count[γ] = len(set(partition.values()))

    # Trace la tendance
    plt.figure(figsize=(8, 5))
    plt.plot(resolutions, [res_to_count[γ] for γ in resolutions], marker='o')
    plt.xlabel("Résolution γ")
    plt.ylabel("Nombre de communautés")
    plt.title("Tendance du nombre de communautés vs. résolution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return res_to_count

trend = plot_communities_trend(G_science, min_res=0.1, max_res=2.0, step=0.1)
print(trend)