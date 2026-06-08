# Studying PV neuron reprogramming with Cellrank

How to run:
1. Put clustered_anndata.h5ad in the "Data" directory
2. Download [list of human transcription factors](https://guolab.wchscu.cn/AnimalTFDB4_static/download/TF_list_final/Homo_sapiens_TF) and put it in the "Data" directory.
3. Create a Conda environment from "environment.yml".
4. Run the Cellrank analysis with "Cellrank scVelo.ipynb"
5. Identify driver genes with "Finding drivers.ipynb".

The repository also contains notebooks for Cellrank analyses with SDEvelo and UniTVelo, as well as a notebook for creating a force-directed layout.
 
