from scipy.stats import pearsonr, entropy, spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error
import numpy as np

def pcc(x1,x2):
    return pearsonr(x1,x2)[0]

def spcc(x1,x2):
    return spearmanr(x1,x2)[0]

def rmse(x1,x2):
    return mean_squared_error(x1,x2,squared=False)

def mae(x1,x2):
    return np.mean(np.abs(x1-x2))

def js(x1,x2):
    return jensenshannon(x1,x2)

def kl(x1,x2):
    entropy(x1, x2)

def ssim(im1,im2,M=1):
    im1, im2 = im1/im1.max(), im2/im2.max()
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim