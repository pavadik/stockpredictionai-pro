import pandas as pd
from sklearn.decomposition import PCA

def eigen_portfolio(df: pd.DataFrame, n_components: int = 12) -> pd.DataFrame:
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(df.values)
    cols = [f"eig_{i+1:02d}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, index=df.index, columns=cols)


from typing import Tuple

def fit_eigen(df: pd.DataFrame, n_components: int = 12) -> Tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(df.values)
    cols = [f"eig_{i+1:02d}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, index=df.index, columns=cols), pca

def transform_eigen(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    comps = pca.transform(df.values)
    cols = [f"eig_{i+1:02d}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, index=df.index, columns=cols)
