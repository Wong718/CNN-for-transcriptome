from constant import colour_map
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def plot_umap(
        df: pd.DataFrame,
        plot_title: str,
        output_dir: str,
        colour_map=colour_map,
        label_col: str = "true_label",
        seed: int = 42,
        save_plot=True
) -> None:
    """Take in a dataframe with a label column and display UMAP beautifully

    Args:
        - df:              input DataFrame, must have a labels column defaulted to be true_label
        - plot_title:      used to label the UMAP plot
        - output_dir:      directory to save output in
        - run_mode:        either to run UMAP on CPU or GPU
        - save_as:         output format. Can only take .html or .svg
        - label_col:       title of the labels columns

    """
    unlabeled_df = df.drop(label_col, axis=1)

    fit = umap.UMAP(n_components=2, random_state=seed)

    u = fit.fit_transform(unlabeled_df)

    df_subset = pd.DataFrame(df[label_col], index=df.index)
    df_subset["umap-2d-one"] = u[:, 0]
    df_subset["umap-2d-two"] = u[:, 1]

    if save_plot == True:
        figure(num=None, figsize=(14, 8), dpi=80, facecolor="w", edgecolor="k")
        fig = plt.scatter(
            x=df_subset["umap-2d-one"],
            y=df_subset["umap-2d-two"],
            c=df_subset[label_col].apply(lambda x: colour_map[x]),
            s=1,
        )
        plt.axis("off")
        file_path = output_dir + f"{plot_title}.svg"
        plt.savefig(file_path)
    return df_subset