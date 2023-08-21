from sklearn.utils.random import sample_without_replacement
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

def get_kmeans_score(umap_df, label_col, verb=True):
    umap_unlabel = umap_df.drop(label_col, axis=1)
    km = KMeans(n_clusters=29, n_init=100, max_iter=10000, init="random")
    km_arr = km.fit_predict(umap_unlabel)
    umap_df["km"] = km_arr
    if verb == True:
        print(
            "Homogeneity: ", metrics.homogeneity_score(umap_df["type"], umap_df["km"])
        )
        print(
            "Completeness: ", metrics.completeness_score(umap_df["type"], umap_df["km"])
        )
        print("V-measure: ", metrics.v_measure_score(umap_df["type"], umap_df["km"]))
    return umap_df

def get_kmeans_dict(umap_df, label_col):
    kmeans_dict = {}
    umap_df = get_kmeans_score(umap_df, "type", verb=False)
    kmeans_dict["Homogeneity"] = metrics.homogeneity_score(
        umap_df["type"], umap_df["km"]
    )
    kmeans_dict["Completeness"] = metrics.completeness_score(
        umap_df["type"], umap_df["km"]
    )
    kmeans_dict["V-Measure"] = metrics.v_measure_score(umap_df["type"], umap_df["km"])
    return kmeans_dict

    umap_df_dict[i] = [shap_umap_df, random_umap_df, edger_umap_df, full_umap_df]

def get_random_gene_df(gene_df, n_genes, label_col="type"):
    labels = gene_df.loc[:, label_col]
    unlab_df = gene_df.drop(label_col, axis=1)
    index_set = sample_without_replacement(gene_df.shape[1], n_genes)
    gene_arr_all = gene_df.columns
    gene_arr_rand = gene_arr_all[index_set]
    gene_df_rand = gene_df[gene_arr_rand]
    gene_df_rand["type"] = labels
    return gene_df_rand

def get_p_value(alt_hypo, test_stat_null, num_samples):
    """
    Params:
        @null_hypo <list> : list of bootstrap samples from the null hypothesis
        @test_hypo <float> : single statistic we are doing the test on
    """
    alt_hypo = np.array(alt_hypo)
    p_value = 1-(len(alt_hypo[alt_hypo >= test_stat_null])/len(alt_hypo))
    if p_value == 0.0:
        p_value = 1/num_samples
    return p_value