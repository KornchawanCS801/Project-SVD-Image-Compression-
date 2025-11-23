import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1) Compute SVD from eigen-decomposition of A^T A
def svd_from_eigendecomp(A: np.ndarray, eps: float = 1e-12):
    # A is an m x n matrix (grayscale image)
    AtA = A.T @ A                       # n x n
    lam, V = np.linalg.eigh(AtA)       # eigenvalues & eigenvectors of A^T A

    # singular values = sqrt(eigenvalues)
    sigma = np.sqrt(np.clip(lam, 0, None))

    # sort from largest to smallest
    idx = np.argsort(sigma)[::-1]
    sigma = sigma[idx]
    V = V[:, idx]

    # keep only singular values bigger than eps
    r = int(np.sum(sigma > eps))
    sigma_r = sigma[:r]
    V_r = V[:, :r]

    # U = A V / sigma
    U_r = A @ V_r
    U_r = U_r / sigma_r[None, :]       # divide each column by sigma_i

    # normalize columns of U
    U_r = U_r / np.linalg.norm(U_r, axis=0, keepdims=True)

    return U_r, sigma_r, V_r

# 2) Rank–k approximation  A_k = sum_{i=1..k} sigma_i u_i v_i^T
def reconstruct_rank_k(U_r, sigma_r, V_r, k: int):
    k_eff = int(min(k, len(sigma_r)))   # in case k is bigger than rank
    if k_eff == 0:
        return np.zeros((U_r.shape[0], V_r.shape[0]), dtype=np.float64), 0

    Ak = np.zeros((U_r.shape[0], V_r.shape[0]), dtype=np.float64)
    for i in range(k_eff):
        ui = U_r[:, i][:, None]        # m x 1
        vi = V_r[:, i][None, :]        # 1 x n
        Ak += sigma_r[i] * (ui @ vi)   # add rank-1 term

    return Ak, k_eff


# 3) Read image in grayscale
img = cv2.imread("rdr2.jpg", cv2.IMREAD_GRAYSCALE)   # any image name is fine
if img is None:
    raise FileNotFoundError("Cannot find 'rdr2.jpg'")

A = img.astype(np.float64)

# 4) Try several k values
k_list = [1, 5, 10, 100]
compressed_imgs = []

U_r, sigma_r, V_r = svd_from_eigendecomp(A)

for k in k_list:
    Ak, k_eff = reconstruct_rank_k(U_r, sigma_r, V_r, k)
    Ak = np.clip(Ak, 0, 255).astype(np.uint8)
    compressed_imgs.append((k_eff, Ak))

# 5) Show original + compressed images
n_cols = 1 + len(k_list)
plt.figure(figsize=(3 * n_cols, 4))

# original
plt.subplot(1, n_cols, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

# each k
for i, (k_eff, Ak) in enumerate(compressed_imgs, start=2):
    plt.subplot(1, n_cols, i)
    plt.imshow(Ak, cmap="gray")
    plt.title(f"k = {k_eff}")
    plt.axis("off")

plt.tight_layout()
plt.show()