# CNN-for-transcriptome
The CNN model for transcriptome data mining. 

> These are the codes for the **Comparative analysis of tissue-specific genes in maize based on machine learning models: CNN performs technically best, LightGBM performs biologically sound** project
> 

## Environment requirement

### Environment

- Python 3
- TensorFlow 2.11
- Jupyter notebook

GPU is recommended for boosting the CNN model training.


## Usage

- All code should be downloaded in batch and placed in the same folder.

### 1.  cnn (****Training 2D CNN on Maize transcriptome data****)

- Input file: Expression matrix for training in .csv form
    
    The expression matrix should be transferred into the approprate shape: The first column is the sample ID, the middle columns are the expression of genes among all samples and the last column is the tissue types of the samples.
    
- Output: *model_topology.json, model_topology.json*

In this notebook, the CNN model located in the modelling folder would be invoked to train the expression matrix. The accuracy and loss of the training and validation set are visualized in the notebook respectively and the F1 score was calculated.

### 2. shap (Runing SHAP model to interprete the CNN model)

- input file:
    - Expression matrix for training in .csv form
    - The .json files generated in the last step
- Output: *shap_scores.pkl, ranks.pkl, shap_genes.pkl*

In this notebook, the SHAP model would be used to evalue the performance of each gene and interprete the CNN model. The weight and architecture of the CNN model are loaded into the SHAP explainer to initialize it with the training expression matrix. Following initialization, the test data set is fed into the SHAP explainer, yielding an array of SHAP values for each sample. Then the genes are filtered and ranked due to their SHAP values within each sample. The filtered gene set with their SHAP values are saved in the shap_genes.pkl.

### 3. ****Cluster analysis comparison****

- input file:
    - Expression matrix for training in .csv form
    - The gene sets generated by different methods
- Output: figures in .svg form and *shap_kmeans.pkl*

In this notebook, the V-measure method with k-means clustering would be used to evalue the performance of each gene set. The k-means clustering is implemented for each gene set individually and then their V-measure value was calculated due to the clustering results. In addition, 5 (optional) random selected gene sets are also processed in the same way. The results of clustering and V-measure are visualized in the notebook and saved as the .svg files.

### 4. ****Cluster analysis t-test****

- input file:
    - Expression matrix for training in .csv form
    - *shap_kmeans.pkl* generated in the last step
- Output: *Fig_t.svg and rand_gauss.pkl*

In this notebook, the result of V-measure method of SHAP gene set would be evalued by t-test. 100 random gene samplings are carried out using various random k-means initializations. The mean of each k-means sample distribution is used to create a null distribution. The 'true' test statistic would be the mean values from the k-means sampling of tissue-specific genes identification technique. The probability of picking SHAP genes at random is then evaluated using a one tail student's t-test and be visualized in the notebook.
