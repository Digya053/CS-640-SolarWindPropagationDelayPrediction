import numpy as np
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

class SolarWindDataset(Dataset):
  """
  A class to construct SolarWindDataset. Each dataset is normalized and dimensionality is reduced.
  Attributes
  ----------
    X: matrix
      Matrix consisting of features
    Y: array
      Corresponding target
  """
  def __init__(self, X, Y, transform=None):
    self.X = X
    self.Y = Y
    self.transform = transform

  def __len__(self):
    """Gives length of data."""
    return len(self.X)

  def __getitem__(self, index):
    """Get item of corresponding index.
    Parameters:
    -----------
      index: Integer
        Index of a dataset 
    """
    X = self.X[index]
    y = self.Y[index]
    X_scaled = StandardScaler().fit_transform(X)
    pca_feat = PCA(n_components=5)
    pca_seq = PCA(n_components = 1)
    important_features = pca_feat.fit_transform(X_scaled)
    important_sequences = pca_seq.fit_transform(np.transpose(important_features))

    if self.transform:
      x = normalize(np.transpose(important_sequences), axis=1, norm='l1')
      x = self.transform(x)
    return (x, y)