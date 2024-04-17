This code is for the paper "MNGT: A Multi-scale Nested Graph Transformer for Classification of High-Resolution Chest X-ray Images with Limited Annotations"

Abstract: The classification of high-resolution Chest X-ray (CXR) images is crucial for accurate diagnosis and treatment planning of various lung conditions. 
In this paper, we introduce a Multi-scale Nested Graph Transformer (MNGT) architecture specifically designed for the efficient analysis of such images. 
Our methodology involves segmenting high-resolution CXR images into numerous squares, subsequently applying a graph Transformer with a variable attention scope to the further subdivided image patches. 
This strategy enables the model to capture features from a local to global perspective, thereby ensuring the retention of crucial details pertaining to small lesion targets, 
all while maintaining the integrity of vital information. Next, high-resolution and low-resolution images are seamlessly fused using a cross-attention-based graph Transformer. 
Moreover, to further enhance the model's semantic parsing ability and processing efficiency, graph pooling is employed within the model pipeline to aggregate the patches and form semantic regions.
By incorporating inductive bias, our method mitigates the risk of overfitting in scenarios with limited labeled data, thereby enhancing the model's generalization capabilities. 
Through extensive experiments on three types of high-resolution CXR images, we demonstrate the superiority of our architecture, surpassing other models in terms of both accuracy and F1-score. 
Furthermore, our ablation study highlights the efficiency of our designed architecture.
