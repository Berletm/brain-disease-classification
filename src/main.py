import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main() -> None:
    train_embeds = np.loadtxt("../embeddings/train_embed.txt")
    train_labels = np.loadtxt("../embeddings/train_label.txt", dtype=int)
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeds)
    
    val_embeds = np.loadtxt("../embeddings/val_embed.txt")
    val_labels = np.loadtxt("../embeddings/val_label.txt", dtype=int)
    
    test_embeds = np.loadtxt("../embeddings/test_embed.txt")
    test_labels = np.loadtxt("../embeddings/test_label.txt", dtype=int)

    other_embeds = np.concatenate([val_embeds, test_embeds], axis=0)
    other_labels = np.concatenate([val_labels, test_labels], axis=0)
    
    other_scaled = scaler.transform(other_embeds)
    
    
    pca = KernelPCA(n_components=2, kernel="rbf", random_state=42)
    pca.fit(train_scaled)
    kernel_pca_train = pca.transform(train_scaled)
    kernel_pca_other = pca.transform(other_scaled)
    
    pca = KernelPCA(n_components=2, kernel="linear", random_state=42)
    pca.fit(train_scaled)
    pca_train = pca.transform(train_scaled)
    pca_other = pca.transform(other_scaled)
    
    umap = UMAP(n_components=2, random_state=42)
    umap.fit(train_scaled)
    umap_train = umap.transform(train_scaled)
    umap_other = umap.transform(other_scaled)
    
    tsne = TSNE(n_components=2, random_state=42)
    all_scaled = np.concatenate([train_scaled, other_scaled], axis=0)
    tsne_all = tsne.fit_transform(all_scaled)
    
    n_train = train_scaled.shape[0]
    tsne_train = tsne_all[:n_train]
    tsne_other = tsne_all[n_train:]
    
    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    
    methods_data = {
        'PCA': (pca_train, pca_other),
        'kernel PCA (rbf)': (kernel_pca_train, kernel_pca_other),
        't-SNE': (tsne_train, tsne_other),
        'UMAP': (umap_train, umap_other)
    }
    
    for i, (method_name, (train_proj, other_proj)) in enumerate(methods_data.items()):
        df_train = pd.DataFrame({
            'x': train_proj[:, 0], 'y': train_proj[:, 1],
            'label': train_labels, 'dataset': 'Train'
        })
        df_other = pd.DataFrame({
            'x': other_proj[:, 0], 'y': other_proj[:, 1],
            'label': other_labels, 'dataset': 'Val/Test'
        })
        df = pd.concat([df_train, df_other], ignore_index=True)
        
        sns.scatterplot(
            data=df, x='x', y='y', 
            hue='label', style='dataset',
            palette='tab10', markers=['o', 'X'], 
            s=60, alpha=0.7, ax=ax[i]
        )
        
        ax[i].set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        if i < 3:
            ax[i].get_legend().remove()
        else:
            sns.move_legend(ax[i], "center left", bbox_to_anchor=(1.02, 0.5), title='Класс / Выборка')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()