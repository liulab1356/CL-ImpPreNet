## Dependencies
python == 3.8.10 pytorch == 1.11.0 numpy == 1.22.4 pandas == 1.4.3 scikit-learn == 1.1.2 

## Datasets
We made our experiments on the MIMIC-III and eICU datasets. In order to access the datasets, please refer to  https://mimic.physionet.org/gettingstarted/access/ and https://eicu-crd.mit.edu/gettingstarted/access/.

## Main Entrance
main.py contains both training code and evaluation code. 

## Ablation Studies
We present two variants of our approach as follows:  
$Ours_{\alpha}$ (A variation of our approach that does not perform graph analysis-based patient stratification modeling) :  
Remove the Similarity, GCN, and InfoAgg classes  
$Ours_{\beta}$ (A variation of our approach in which we omit the contrastive learning component) :  
Remove the CLLoss class
