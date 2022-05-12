############### PROJET 5 #################

import time
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.cm as cm
from sklearn import (preprocessing,
                     manifold,
                     decomposition)
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from sklearn.metrics.cluster import adjusted_rand_score
from math import radians, cos, sin, asin, sqrt
import warnings

def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : a list containing two values :
                                - the dataframe for the data
                                - a brief description of the file

        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
                            - a column "Description"    : a brief description of the file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    'Description': files_descriptions})

    presentation_df.index += 1

    return presentation_df

#------------------------------------------

def describe_columns(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : list of dataframes
        
        Returns
        ---------------
        resume_data : pandas dataframe :
                            
    '''

    liste_indices = []
    liste_colonnes = []
    liste_types = []
    liste_uniques = []

    for filename, file_data in source_files.items():
        for column in file_data.columns:
            liste_indices.append(filename)
            liste_colonnes.append(column)
            liste_types.append(file_data[column].dtype)
            liste_uniques.append(file_data[column].nunique())
 
    resume_data = pd.DataFrame([liste_indices, 
                                liste_colonnes, 
                                liste_types, 
                                liste_uniques]
                              ).T
    resume_data.columns=['Nom fichier', 'Nom colonne', 'Type', 'valeurs uniques']

    return resume_data

#------------------------------------------

def get_missing_values_percent_per(data):
    '''
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value
        of a given column

        Parameters
        ----------------
        data                : pandas dataframe
                              The dataframe to be analyzed

        Returns
        ---------------
        missing_percent_df  : A pandas dataframe containing:
                                - a column "column"
                                - a column "Percent Missing" containing the percentage of
                                  missing value for each value of column
    '''

    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})
    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']
    missing_percent_df['Total'] = 100

    return missing_percent_df


#------------------------------------------

def plot_percentage_missing_values_for(data, long, larg):
    '''
        Plots the proportions of filled / missing values for each unique value
        in column as a horizontal bar chart.

        Parameters
        ----------------
        data : pandas dataframe with:
                - a column column
                - a column "Percent Filled"
                - a column "Percent Missing"
                - a column "Total"

       long : int
            The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    data_to_plot = get_missing_values_percent_per(data)\
                     .sort_values("Percent Filled").reset_index()

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    handle_plot_1 = sns.barplot(x="Total", y="index", data=data_to_plot,
                                label="non renseignées", color="thistle", alpha=0.3)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(), size=TICK_SIZE)
    _, ylabels = plt.yticks()
    handle_plot_1.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    handle_plot_2 = sns.barplot(x="Percent Filled", y="index", data=data_to_plot,
                                label="renseignées", color="darkviolet")

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(), size=TICK_SIZE)
    handle_plot_2.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    axis.legend(bbox_to_anchor=(1.04, 0), loc="lower left",
                borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    axis.set(ylabel="Colonnes", xlabel="Pourcentage de valeurs (%)")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                               pos: '{:2d}'.format(int(x)) + '%'))
    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def plot_correlation_circle(pcs, data, long, larg):
    '''
        Plots 2 distplots horizontally in a single figure

        Parameters
        ----------------
        pcs     : PCA components
                  Components from a PCA

        data    : pandas dataframe
                  The original data used for the PCA
                  before any treatment

        long            : int
                          The length of the figure for the plot

        larg            : int
                          The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    TITLE_PAD = 70

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[0, :], pcs[1, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, data.columns[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')

    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[2, :], pcs[3, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, data.columns[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    
    plt.xlabel('Composante 3')
    plt.ylabel('Composante 4')
    

    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[4, :], pcs[5, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, data.columns[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    
    plt.xlabel('Composante 5')
    plt.ylabel('Composante 6')
    
    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

#------------------------------------------

def haversine_distance(lat1, lng1, lat2, lng2, degrees=True):
    r = 3956 # rayon de la Terre en miles
    
    if degrees:
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    
    # Formule Haversine
    dlng = lng2 - lng1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    d = 2 * r * asin(sqrt(a))  

    return d

#------------------------------------------

def fit_plot(algorithms, data, long, larg, title):
    '''
        For each given algorithm :
        - fit them to the data on 3 iterations
        - Calculate the mean silhouette and adjusted rand scores
        - Gets the calculation time

        The function then plots the identified clusters for each algorithm.

        Parameters
        ----------------
        algorithms : dictionary with
                        - names and type of input as keys
                        - instantiated algorithms as values

        - data     : pandas dataframe
                     Contains the data to fit the algos on

        - long     : int
                     length of the plot figure

        - larg     : int
                     width of the plot figure

        - title    : string
                     title of the plot figure

        Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient,
                      the adjusted Rnad score, the number of clusters,
                      the calculation time for each algorithm in algorithms
    '''

    scores_time = pd.DataFrame(columns=["Algorithme", "iter",
                                        "silhouette", "ARI",
                                        "Nb Clusters", "Time"])

    # Constants for the plot
    TITLE_SIZE = 45
    TITLE_PAD = 1.05
    SUBTITLE_SIZE = 25
    TICK_SIZE = 25
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30

    nb_rows = int(len(algorithms)/2) if int(len(algorithms)/2) > 2 else 2
    nb_cols = 2

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(larg, long))
    fig.suptitle(title, fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = column = 0
    ITER = 3 # constant

    for algoname, algo in algorithms.items():

        cluster_labels = {}

        for i in range(ITER):
            if algoname == "Dummy":
                start_time = time.time()
                cluster_labels[i] = assign_random_clusters(data, algo)
                elapsed_time = time.time() - start_time
            else:
                start_time = time.time()
                algo.fit(data)
                elapsed_time = time.time() - start_time
                cluster_labels[i] = algo.labels_

        for i in range(ITER):
            j = i+1

            if i == 2:
                j = 0

            scores_time.loc[len(scores_time)] = [algoname, i,
                                                 silhouette_score(data,
                                                                  cluster_labels[i],
                                                                  metric="euclidean"),
                                                 adjusted_rand_score(cluster_labels[i],
                                                                     cluster_labels[j]),
                                                 len(set(cluster_labels[i])),
                                                 elapsed_time]

        # plot
        #if nb_rows > 1:
        axis = axes[row, column]
        #else:
        #    axis = axes

        data_to_plot = data.copy()
        data_to_plot["cluster_labels"] = cluster_labels[ITER-1]
        plot_handle = sns.scatterplot(x="tsne-pca-one", y="tsne-pca-two",
                                      data=data_to_plot, hue="cluster_labels",
                                      palette=sns\
                                      .color_palette("hls",
                                                     data_to_plot["cluster_labels"].nunique()),
                                      legend="full", alpha=0.3, ax=axis)

        plt.tight_layout()

        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.3, hspace=0.4)

        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()
        axis.spines['left'].set_position(('outward', 10))
        axis.spines['bottom'].set_position(('outward', 10))

        axis.set_xlabel('tsne-pca-one', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        axis.set_ylabel('tsne-pca-two', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)

        plot_handle.set_yticklabels(plot_handle.get_yticks(), size=TICK_SIZE)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)

        scores = (r'$Silh={:.2f}$' + '\n' + r'$ARI={:.2f}$')\
                 .format(scores_time[scores_time["Algorithme"] == algoname]["silhouette"].mean(),
                         scores_time[scores_time["Algorithme"] == algoname]["ARI"].mean())

        axis.legend([extra], [scores], loc='upper left', fontsize=LEGEND_SIZE)
        title = algoname + '\n Évaluation en {:.2f} secondes'.format(elapsed_time)
        axis.set_title(title, fontsize=SUBTITLE_SIZE, fontweight="bold")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

    return scores_time

#------------------------------------------

def fit_plot2(algorithms, data, dftsne, long, larg, title):
    '''
        For each given algorithm :
        - fit them to the data on 3 iterations
        - Calculate the mean silhouette and adjusted rand scores
        - Gets the calculation time

        The function then plots the identified clusters for each algorithm.

        Parameters
        ----------------
        algorithms : dictionary with
                        - names and type of input as keys
                        - instantiated algorithms as values

        - data     : pandas dataframe
                     Contains the data to fit the algos on

        - long     : int
                     length of the plot figure

        - larg     : int
                     width of the plot figure

        - title    : string
                     title of the plot figure

        Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient,
                      the adjusted Rnad score, the number of clusters,
                      the calculation time for each algorithm in algorithms
    '''

    scores_time = pd.DataFrame(columns=["Algorithme", "silhouette", "Nb Clusters", "Time"])

    # Constants for the plot
    TITLE_SIZE = 45
    TITLE_PAD = 1.05
    SUBTITLE_SIZE = 25
    TICK_SIZE = 25
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30

    nb_rows = int(len(algorithms)/2) if int(len(algorithms)/2) > 2 else 2
    nb_cols = 2

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(larg, long))
    fig.suptitle(title, fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = column = 0
    ITER = 1 # constant

    for algoname, algo in algorithms.items():

        cluster_labels = {}

        for i in range(ITER):
            if algoname == "Dummy":
                start_time = time.time()
                cluster_labels[i] = assign_random_clusters(data, algo)
                elapsed_time = time.time() - start_time
            else:
                start_time = time.time()
                algo.fit(data)
                elapsed_time = time.time() - start_time
                cluster_labels[i] = algo.labels_

        for i in range(ITER):
            j = i+1

            if i == 2:
                j = 0

            scores_time.loc[len(scores_time)] = [algoname, 
                                                 silhouette_score(data,
                                                                  cluster_labels[i],
                                                                  metric="euclidean"),
                                                 len(set(cluster_labels[i])),
                                                 elapsed_time]

        # plot
        #if nb_rows > 1:
        axis = axes[row, column]
        #else:
        #    axis = axes

        data_to_plot = dftsne.copy()
        data_to_plot["cluster_labels"] = cluster_labels[ITER-1]
        plot_handle = sns.scatterplot(x="tsne-pca-one", y="tsne-pca-two",
                                      data=data_to_plot, hue="cluster_labels",
                                      palette=sns\
                                      .color_palette("hls",
                                                     data_to_plot["cluster_labels"].nunique()),
                                      legend="full", alpha=0.3, ax=axis)

        plt.tight_layout()

        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.3, hspace=0.4)

        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()
        axis.spines['left'].set_position(('outward', 10))
        axis.spines['bottom'].set_position(('outward', 10))

        axis.set_xlabel('tsne-pca-one', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        axis.set_ylabel('tsne-pca-two', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)

        plot_handle.set_yticklabels(plot_handle.get_yticks(), size=TICK_SIZE)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)

        scores = (r'$Silh={:.2f}$' + '\n' + r'$ARI={:.2f}$')\
                 .format(scores_time[scores_time["Algorithme"] == algoname]["silhouette"].mean(),
                         scores_time[scores_time["Algorithme"] == algoname]["ARI"].mean())

        axis.legend([extra], [scores], loc='upper left', fontsize=LEGEND_SIZE)
        title = algoname + '\n Évaluation en {:.2f} secondes'.format(elapsed_time)
        axis.set_title(title, fontsize=SUBTITLE_SIZE, fontweight="bold")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

    return scores_time

#------------------------------------------

def fit_plot3(algorithm, data, dftsne, long, larg, title):
    '''
        For each given algorithm :
        - fit them to the data
        - Calculate the mean silhouette
        - Gets the calculation time

        The function then plots the identified clusters for each algorithm.

        Parameters
        ----------------
        - algorithm : dictionary with
                        - name and type of input as keys
                        - instantiated algorithm as values

        - data     : pandas dataframe
                     Contains the data to fit the algo on
                     
        - dftnse   : pandas dataframe
                     Contains 2D tsne data

        - long     : int
                     length of the plot figure

        - larg     : int
                     width of the plot figure

        - title    : string
                     title of the plot figure

        Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient, the number of clusters,
                      the calculation time of the algorithm
    '''

    scores_time = pd.DataFrame(columns=["Algorithme", "silhouette", "Nb Clusters", "Time"])

    # Constants for the plot
    TITLE_SIZE = 40
    TITLE_PAD = 1.05
    LABEL_SIZE = 30
    LABEL_PAD = 20

    for algoname, algo in algorithm.items():
        start_time = time.time()
        cluster_labels = algo.fit_predict(data)
        elapsed_time = time.time() - start_time

        scores_time.loc[len(scores_time)] = [algoname, silhouette_score(data,
                                                                          cluster_labels,
                                                                          metric="euclidean"),
                                                         len(set(cluster_labels)),
                                                         elapsed_time]

        data_to_plot = dftsne.copy()
        data_to_plot["cluster_labels"] = cluster_labels
    
        f, (ax1, ax2) = plt.subplots(ncols=2,
                                     sharey=False,
                                     figsize=(long, larg))

        f.suptitle(title, fontsize=TITLE_SIZE, y=TITLE_PAD)
        
        ax1 = plt.subplot(1, 2, 1)

        handle_plot_1 = sns.scatterplot(x="pca-one", y="pca-two",
                                        data=data_to_plot,
                                        hue="cluster_labels",
                                        palette=sns\
                                        .color_palette("hls",data_to_plot["cluster_labels"].nunique()),
                                        legend="full",
                                        alpha=0.3,
                                        ax=ax1)

        handle_plot_1.set_xlabel("PCA 1",
                                 fontsize=LABEL_SIZE,
                                 labelpad=LABEL_PAD,
                                 fontweight="bold")

        handle_plot_1.set_ylabel("PCA 2",
                                 fontsize=LABEL_SIZE,
                                 labelpad=LABEL_PAD,
                                 fontweight="bold")


        ax2 = plt.subplot(1, 2, 2)
        handle_plot_2 = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                                        data=data_to_plot,
                                        hue="cluster_labels",
                                        palette=sns\
                                        .color_palette("hls",data_to_plot["cluster_labels"].nunique()),
                                        legend="full",
                                        alpha=0.3,
                                        ax=ax2)

        handle_plot_2.set_xlabel("t-SNE 1",
                                 fontsize=LABEL_SIZE,
                                 labelpad=LABEL_PAD,
                                 fontweight="bold")

        handle_plot_2.set_ylabel("t-SNE 2",
                                 fontsize=LABEL_SIZE,
                                 labelpad=LABEL_PAD,
                                 fontweight="bold")

    
    
    return scores_time    
    
#------------------------------------------

def getScores(algorithm, data):
    '''
        For each given algorithm :
        - fit them to the data
        - Calculate the mean silhouette
        - Gets the calculation time

        The function then plots the identified clusters for each algorithm.

        Parameters
        ----------------
        - algorithm : dictionary with
                        - name and type of input as keys
                        - instantiated algorithm as values

        - data     : pandas dataframe
                     Contains the data to fit the algo on
                     
         Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient, the number of clusters,
                      the calculation time of the algorithm
    '''

    scores_time = pd.DataFrame(columns=["Algorithme", "silhouette", "Nb Clusters", "Time"])

    for algoname, algo in algorithm.items():
        start_time = time.time()
        cluster_labels = algo.fit_predict(data)
        elapsed_time = time.time() - start_time

        scores_time.loc[len(scores_time)] = [algoname, silhouette_score(data,
                                                                          cluster_labels,
                                                                          metric="euclidean"),
                                                         len(set(cluster_labels)),
                                                         elapsed_time]
    
    return scores_time    
    
#------------------------------------------


def plotCorrelationHeatMap(data, corr_method, long, larg):
    '''
        heatmap des coefficients de corrélation entre les colonnes quantitatives
        
        ----------------
        - data : une dataframe contenant les données
        - corr : méthode de correlation ("pearson" or "spearman")
        - long : int
                 longueur de la figure
        
        - larg : int
                 largeur de la figure 
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5
    
    corr = data.corr(method = corr_method)

    f, ax = plt.subplots(figsize=(long, larg))
                
    f.suptitle("COEFFICIENT DE CORRÉLATION DE " + corr_method.upper(), fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax,
                    annot=corr, annot_kws={"fontsize":20}, fmt=".2f")

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(data.columns.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(data.index.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
                
#------------------------------------------

def plotEblow(data, long, larg):
    
    plt.figure(figsize=(long, larg))
    Sum_of_squared_distances = []

    K = range(1, 11)

    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)

        Sum_of_squared_distances.append(km.inertia_) 
    
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k', fontsize=20,
               labelpad=30)
    plt.ylabel('Sum_of_squared_distances',
               fontsize=20,
               labelpad=30)
    plt.title('Elbow Method For Optimal k',
              fontsize=30, pad=30)

    plt.show()

#------------------------------------------

def plotSilhouetteVisualizer(data, range_n_clusters):
    
    X = data

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

#------------------------------------------

def display_scree_plot(pca):
    '''
        Plots the scree plot for the given pca
        components.

        ----------------
        - pca : A PCA object
                The result of a PCA decomposition

        Returns
        ---------------
        _
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    LABEL_SIZE = 20
    LABEL_PAD = 30

    plt.subplots(figsize=(10, 8))

    scree = pca.explained_variance_ratio_ * 100

    plt.bar(np.arange(len(scree))+1, scree)

    plt.plot(np.arange(len(scree))+1,
             scree.cumsum(), c="red", marker='o')

    plt.xlabel("Rang de l'axe d'inertie",
               fontsize=LABEL_SIZE,
               labelpad=LABEL_PAD)

    plt.ylabel("% d'inertie",
               fontsize=LABEL_SIZE,
               labelpad=LABEL_PAD)

    plt.title("Eboulis des valeurs propres",
              fontsize=TITLE_SIZE)

    plt.show(block=False)

#------------------------------------------

def plot_radars(data, group):

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), 
                        index=data.index,
                        columns=data.columns).reset_index()
    
    fig = go.Figure()

    for k in data[group]:
        fig.add_trace(go.Scatterpolar(
            r=data[data[group]==k].iloc[:,1:].values.reshape(-1),
            theta=data.columns[1:],
            fill='toself',
            name='Cluster '+str(k)
        ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
        showlegend=True,
        title={
            'text': "<b>Projection des moyennes par variable des clusters</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="black",
        title_font_size=26)

    fig.show()
#------------------------------------------    
    
def clustering_eval(preprocessor, model, data, metric, elbow=True, mds=False, sil=False, KBest=None):
    
    if((elbow==True) & (mds==True)):
        ncols=3
    elif((elbow==False) | (mds==False)):
        ncols=2
    else:
        ncols=1
        
    fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=False, sharey=False, figsize=(24,8))
    
    ax=0
    if(elbow==True):
        # Elbow visualizer
        kmeans_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("kelbowvisualizer", KElbowVisualizer(model,K=(4,12), metric=metric, ax=axes[ax]))])
        kmeans_visualizer.fit(data)
        KBest = kmeans_visualizer.named_steps['kelbowvisualizer'].elbow_value_
        kmeans_visualizer.named_steps['kelbowvisualizer'].finalize()
        ax+=1
    
    # Set best K
    K = KBest
    model.set_params(n_clusters=K)

    # Silhouette Visualizer
    if(sil==True):
        silhouette_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("silhouettevisualizer", SilhouetteVisualizer(model, ax=axes[ax]))])
        silhouette_visualizer.fit(data)
        silhouette_visualizer.named_steps['silhouettevisualizer'].finalize()
        ax+=1
    
    # Intercluster distance Map with best k
    if(mds==True):
        distance_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("distancevisualizer", InterclusterDistance(model, ax=axes[ax]))])
        distance_visualizer.fit(data)
        distance_visualizer.named_steps['distancevisualizer'].finalize()
    
    return K
    plt.show()    
    
#------------------------------------------    

def stabilityCheck(nb_iter, K, model, data, labels):
    
    scores_time = pd.DataFrame(columns=["Iteration",
                                        "silhouette", "ARI", "Time"])
    
    for i in range(nb_iter):
        t0 = time.time()
        estimator = make_pipeline(MinMaxScaler(), model).fit(data)
        fit_time = time.time() - t0
        
        scores_time.loc[len(scores_time)] = [i,silhouette_score(data,
                                                                  estimator[1].labels_,
                                                                  metric="euclidean"),
                                                 adjusted_rand_score(labels,
                                                                     estimator[1].labels_),
                                                 fit_time]
    return scores_time

#------------------------------------------    

def create_dataset(dpath="datas/", initial=False, period=2):
    """Cleaning and feature engineering on complete Olist data 
        for preparation of unsupervised classification (K-Means).

    Parameters
    ----------
    dpath : str
        Path to the directory containing the data.
    initial : boolean
        Defines whether the created dataset is the initial dataset.
    period : int
        Increment period in months after initial dataset.
    """
    start_time = time.time()
    #print("Création du dataset en cours ...")
    
    # Root path
    root_path = dpath
    
    # Load datasets
    customers = pd.read_csv(root_path + "olist_customers_dataset.csv")
    geolocation = pd.read_csv(root_path + "olist_geolocation_dataset.csv")
    orders = pd.read_csv(root_path + "olist_orders_dataset.csv")
    order_items = pd.read_csv(root_path + "olist_order_items_dataset.csv")
    order_payments = pd.read_csv(root_path + "olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv(root_path + "olist_order_reviews_dataset.csv")
    products = pd.read_csv(root_path + "olist_products_dataset.csv")
    categories_en = pd.read_csv(root_path + "product_category_name_translation.csv")
    
    # Group location 
    geolocation = geolocation.groupby(["geolocation_state"]).agg({
            "geolocation_lat": "mean",
            "geolocation_lng": "mean"})
    
    # Merge datasets
    # Orders
    orders.drop(["order_approved_at",
                 "order_delivered_carrier_date", 
                 "order_estimated_delivery_date"],
                axis=1, inplace=True)

    order_items.drop(["seller_id",
                      "shipping_limit_date"],
                     axis=1, inplace=True)
    order_items = pd.merge(order_items, orders,
                           how="left",
                           on="order_id")
    
    datetime_cols = ["order_purchase_timestamp", 
                     "order_delivered_customer_date"]
    for col in datetime_cols:
        order_items[col] = order_items[col].astype('datetime64[ns]')
        
    # order Month
    order_items["sale_month"] = order_items['order_purchase_timestamp'].dt.month
    
    # Select orders on period
    start=order_items["order_purchase_timestamp"].min()
    if(initial == True):
        period = 12
    else:
        period = 12+period
    stop=start + pd.DateOffset(months=period)
        
    order_items = order_items[(order_items["order_purchase_timestamp"]>=start)
                              & (order_items["order_purchase_timestamp"]<stop)]
    
    # List of orders on period
    period_orders = order_items.order_id.unique()
    
    # Calculate other features on period
    order_payments = order_payments[order_payments["order_id"].isin(period_orders)]
    order_items = pd.merge(order_items, 
                           order_payments.groupby(by="order_id").agg(
                               {"payment_sequential": 'count',
                                "payment_installments": 'sum'}),
                           how="left",
                           on="order_id")
    order_items = order_items.rename(columns={
        "payment_sequential": "nb_payment_sequential",
        "payment_installments": "sum_payment_installments"})
    
    order_reviews = order_reviews[order_reviews["order_id"].isin(period_orders)]
    order_items = pd.merge(order_items,
                           order_reviews.groupby("order_id").agg({
                               "review_score": "mean"}),
                           how="left",
                           on="order_id")
    
    # Delivery time
    order_items["delivery_delta_days"] = (order_items.order_delivered_customer_date
                                          - order_items.order_purchase_timestamp)\
                                         .dt.round('1d').dt.days
    order_items.drop("order_delivered_customer_date", axis=1, inplace=True)
    
    # Products
    products = pd.merge(products, categories_en,
                    how="left",
                    on="product_category_name")

    del_features_list = ["product_category_name", "product_weight_g",
                         "product_length_cm", "product_height_cm",
                         "product_width_cm", "product_name_lenght", 
                         "product_description_lenght", "product_photos_qty"]
    products.drop(del_features_list, axis=1, inplace=True)
    products = products.rename(columns={"product_category_name_english":
                                        "product_category_name"})
        
    products['product_category'] = np.where((products['product_category_name'].str.contains("fashio|luggage")==True),
                                    'fashion_clothing_accessories',
                            np.where((products['product_category_name'].str.contains("health|beauty|perfum")==True),
                                     'health_beauty',
                            np.where((products['product_category_name'].str.contains("toy|baby|diaper")==True),
                                     'toys_baby',
                            np.where((products['product_category_name'].str.contains("book|cd|dvd|media")==True),
                                     'books_cds_media',
                            np.where((products['product_category_name'].str.contains("grocer|food|drink")==True), 
                                     'groceries_food_drink',
                            np.where((products['product_category_name'].str.contains("phon|compu|tablet|electro|consol")==True), 
                                     'technology',
                            np.where((products['product_category_name'].str.contains("home|furnitur|garden|bath|house|applianc")==True), 
                                     'home_furniture',
                            np.where((products['product_category_name'].str.contains("flow|gift|stuff")==True),
                                     'flowers_gifts',
                            np.where((products['product_category_name'].str.contains("sport")==True),
                                     'sport',
                                     'other')))))))))
    products.drop("product_category_name", axis=1, inplace=True)

    order_items = pd.merge(order_items, products, 
                           how="left",
                           on="product_id")
    
    # Encode categories column
    order_items = pd.get_dummies(order_items, columns=["product_category"], prefix="", prefix_sep="")
    
    # Customers
    order_items = pd.merge(order_items, customers[["customer_id",
                                                   "customer_unique_id",
                                                   "customer_state"]],
                           on="customer_id",
                           how="left")
    
    # Group datas by unique customers
    data = order_items.groupby(["customer_unique_id"]).agg(
        nb_orders=pd.NamedAgg(column="order_id", aggfunc="nunique"),
        total_items=pd.NamedAgg(column="order_item_id", aggfunc="count"),
        total_spend=pd.NamedAgg(column="price", aggfunc="sum"),
        total_freight=pd.NamedAgg(column="freight_value", aggfunc="sum"),
        mean_payment_sequential=pd.NamedAgg(column="nb_payment_sequential", aggfunc="mean"),
        mean_payment_installments=pd.NamedAgg(column="sum_payment_installments", aggfunc="mean"),
        mean_review_score=pd.NamedAgg(column="review_score", aggfunc="mean"),
        mean_delivery_days=pd.NamedAgg(column="delivery_delta_days", aggfunc="mean"),
        books_cds_media=pd.NamedAgg(column="books_cds_media", aggfunc="sum"),
        fashion_clothing_accessories=pd.NamedAgg(column="fashion_clothing_accessories", aggfunc="sum"),
        flowers_gifts=pd.NamedAgg(column="flowers_gifts", aggfunc="sum"),
        groceries_food_drink=pd.NamedAgg(column="groceries_food_drink", aggfunc="sum"),
        health_beauty=pd.NamedAgg(column="health_beauty", aggfunc="sum"),
        home_furniture=pd.NamedAgg(column="home_furniture", aggfunc="sum"),
        other=pd.NamedAgg(column="other", aggfunc="sum"),
        sport=pd.NamedAgg(column="sport", aggfunc="sum"),
        technology=pd.NamedAgg(column="technology", aggfunc="sum"),
        toys_baby=pd.NamedAgg(column="toys_baby", aggfunc="sum"),
        customer_state=pd.NamedAgg(column="customer_state", aggfunc="max"),
        first_order=pd.NamedAgg(column="order_purchase_timestamp", aggfunc="min"),
        last_order=pd.NamedAgg(column="order_purchase_timestamp", aggfunc="max"),
        favorite_sale_month=pd.NamedAgg(column="sale_month", 
                                        aggfunc=lambda x:x.value_counts().index[0]))
    
    # Final feature engineering
    # Categories items ratio
    cat_features = data.columns[7:17]
    for c in cat_features:
        data[c] = data[c] / data["total_items"]
    
    # Mean delay between 2 orders
    data["order_mean_delay"] = [(y[1] - y[0]).round('1d').days if y[1] != y[0]
                                else (stop - y[0]).round('1d').days
                                for x,y in data[["first_order","last_order"]].iterrows()]
    data["order_mean_delay"] = data["order_mean_delay"] / data["nb_orders"]
    data.drop(["first_order", "last_order"], axis=1, inplace=True)
    
    # Freight ratio and total price
    data["freight_ratio"] = (round(data["total_freight"] / (data["total_spend"] + data["total_freight"]),2))
    data["total_spend"] = (data["total_spend"] + data["total_freight"])
    data.drop("total_freight", axis=1, inplace=True)
    
    # Add Haversine distance of customer state
    # Haversine distance
    olist_lat = -25.43045
    olist_lon = -49.29207
        
    geolocation['haversine_distance'] = [haversine_distance(olist_lat, olist_lon, x, y)
                                         for x, y in zip(geolocation.geolocation_lat,
                                                         geolocation.geolocation_lng)]
    data = pd.merge(data.reset_index(), geolocation[["haversine_distance"]],
                    how="left",
                    left_on="customer_state",
                    right_on="geolocation_state")
    data.drop(["customer_state"], axis=1, inplace=True)
    data.set_index("customer_unique_id", inplace=True)
    
    # complete missing values
    features_to_fill = data.isnull().sum()
    features_to_fill = list(features_to_fill[features_to_fill.values > 0].index)
    
    #print(54*"_")
    #print("Features complétées avec la valeur la plus fréquente :")
    #print(54*"_")
    for f in features_to_fill:
        data[f] = data[f].fillna(data[f].mode()[0])
      #  print(f,"\t", data[f].mode()[0])
    #print(54*"_")
    
    end_time = time.time()
    #print("Durée d'execution du Feature engineering : {:.2f}s".format(end_time - start_time))
    
    return data

#------------------------------------------

def plotTSNE(X_scaled, kmeans_labels):
    '''
        For each given algorithm :
        - fit them to the data
        - Calculate the mean silhouette
        - Gets the calculation time

        The function then plots the identified clusters for each algorithm.

        Parameters
        ----------------
        - algorithm : dictionary with
                        - name and type of input as keys
                        - instantiated algorithm as values

        - data     : pandas dataframe
                     Contains the data to fit the algo on
                     
        - dftnse   : pandas dataframe
                     Contains 2D tsne data

        - long     : int
                     length of the plot figure

        - larg     : int
                     width of the plot figure

        - title    : string
                     title of the plot figure

        Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient, the number of clusters,
                      the calculation time of the algorithm
    '''


    TITLE_SIZE = 40
    TITLE_PAD = 1.05
    LABEL_SIZE = 30
    LABEL_PAD = 20
    
    tsne = manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X_scaled)

    data_to_plot = pd.DataFrame()
    
    data_to_plot["tsne-2d-one"] = tsne_results[:,0]
    data_to_plot["tsne-2d-two"] = tsne_results[:,1]
    data_to_plot["kmeans_label"] = kmeans_labels
    
    fig = plt.figure(figsize=(10, 10))
    
    plt.title("Mise en évidence des clusters t-SNE", fontsize=TITLE_SIZE)

    handle_plot_2 = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                                    data=data_to_plot,
                                    hue="kmeans_label",
                                    palette=sns\
                                      .color_palette("hls",
                                                     data_to_plot["kmeans_label"].nunique()),
                                    legend="full")

    handle_plot_2.set_xlabel("t-SNE 1",
                             fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD,
                             fontweight="bold")

    handle_plot_2.set_ylabel("t-SNE 2",
                             fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD,
                             fontweight="bold")    
#------------------------------------------