# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# def svd_from_eigendecomp(A: np.ndarray, eps: float = 1e-12):
#     AtA = A.T @ A
#     lam, V = np.linalg.eigh(AtA)         
#     sigma = np.sqrt(np.clip(lam, 0, None))

#     idx = np.argsort(sigma)[::-1]
#     sigma = sigma[idx]
#     V = V[:, idx]

#     r = int(np.sum(sigma > eps))
#     sigma_r = sigma[:r]
#     V_r = V[:, :r]

#     U_r = A @ V_r
#     U_r = U_r / sigma_r[None, :]        

#     U_r = U_r / np.linalg.norm(U_r, axis=0, keepdims=True)
#     return U_r, sigma_r, V_r
# def reconstruct_rank_k(U_r, sigma_r, V_r, k: int):
#     k_eff = int(min(k, len(sigma_r)))
#     if k_eff == 0:
#         return np.zeros((U_r.shape[0], V_r.shape[0]), dtype=np.float64), 0

#     Ak = np.zeros((U_r.shape[0], V_r.shape[0]), dtype=np.float64)
#     for i in range(k_eff):
#         ui = U_r[:, i][:, None]       
#         vi = V_r[:, i][None, :]     
#         Ak += sigma_r[i] * (ui @ vi)  

#     return Ak, k_eff
# img = cv2.imread("rdr2.jpg")   
# if img is None:
#     raise FileNotFoundError("Can't find picture")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
# k_list = [1, 2 , 4, 100]   
# compressed_list = []

# for k in k_list:
#     comp = np.zeros_like(img, dtype=np.float64)
#     for c in range(3):
#         A = img[:, :, c]
#         U_r, sigma_r, V_r = svd_from_eigendecomp(A)
#         Ak, k_eff = reconstruct_rank_k(U_r, sigma_r, V_r, k)
#         comp[:, :, c] = Ak

#     comp = np.clip(comp, 0, 255).astype(np.uint8)
#     compressed_list.append((k, k_eff, comp))

# n_cols = 1 + len(k_list)
# plt.figure(figsize=(3 * n_cols, 4))

# plt.subplot(1, n_cols, 1)
# plt.imshow(img.astype(np.uint8))
# plt.title("Original")
# plt.axis("off")


# for i, (k, k_eff, comp) in enumerate(compressed_list, start=2):
#     plt.subplot(1, n_cols, i)
#     plt.imshow(comp)
#     plt.title(f"k = {k_eff}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# def svd_from_eigendecomp(A: np.ndarray, eps: float = 1e-12):
#     AtA = A.T @ A
#     lam, V = np.linalg.eigh(AtA)

    
#     sigma = np.sqrt(np.clip(lam, 0, None))

  
#     idx = np.argsort(sigma)[::-1]
#     sigma = sigma[idx]
#     V = V[:, idx]

    
#     r = int(np.sum(sigma > eps))
#     sigma_r = sigma[:r]
#     V_r = V[:, :r]

   
#     U_r = A @ V_r
#     U_r = U_r / sigma_r[None, :]

   
#     U_r = U_r / np.linalg.norm(U_r, axis=0, keepdims=True)

#     return U_r, sigma_r, V_r

# def reconstruct_rank_k(U_r, sigma_r, V_r, k: int):
#     k_eff = int(min(k, len(sigma_r)))
#     if k_eff == 0:
#         return np.zeros((U_r.shape[0], V_r.shape[0]), dtype=np.float64), 0

#     Ak = np.zeros((U_r.shape[0], V_r.shape[0]), dtype=np.float64)

   
#     for i in range(k_eff):
#         ui = U_r[:, i][:, None]   
#         vi = V_r[:, i][None, :]   
#         Ak += sigma_r[i] * (ui @ vi)

#     return Ak, k_eff

# img = cv2.imread("rdr2.jpg")
# if img is None:
#     raise FileNotFoundError("Can't find picture")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)


# k_list = [1, 2, 4, 100]
# compressed_list = []

# for k in k_list:
#     comp = np.zeros_like(img, dtype=np.float64)

   
#     for c in range(3):
#         A = img[:, :, c]
#         U_r, sigma_r, V_r = svd_from_eigendecomp(A)
#         Ak, k_eff = reconstruct_rank_k(U_r, sigma_r, V_r, k)
#         comp[:, :, c] = Ak

#     comp = np.clip(comp, 0, 255).astype(np.uint8)
#     compressed_list.append((k, k_eff, comp))

# n_cols = 1 + len(k_list)
# plt.figure(figsize=(3 * n_cols, 4))

# plt.subplot(1, n_cols, 1)
# plt.imshow(img.astype(np.uint8))
# plt.title("Original")
# plt.axis("off")

# for i, (k, k_eff, comp) in enumerate(compressed_list, start=2):
#     plt.subplot(1, n_cols, i)
#     plt.imshow(comp)
#     plt.title(f"k = {k_eff}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -------------------------------------------
# 1) หาค่า U, sigma, V จาก eigen ของ A^T A
# -------------------------------------------
# def svd_from_eigendecomp(A: np.ndarray, eps: float = 1e-12):
#     AtA = A.T @ A
#     lam, V = np.linalg.eigh(AtA)  # eigenvalues / eigenvectors

#     sigma = np.sqrt(np.clip(lam, 0, None))  # sigma = sqrt(lambda)

#     # เรียงจากมาก -> น้อย
#     idx = np.argsort(sigma)[::-1]
#     sigma = sigma[idx]
#     V = V[:, idx]

#     # ลด rank ตามค่าที่มีจริง
#     r = int(np.sum(sigma > eps))
#     sigma_r = sigma[:r]
#     V_r = V[:, :r]

#     # U = A V / sigma
#     U_r = A @ V_r
#     U_r = U_r / sigma_r[None, :]
#     U_r = U_r / np.linalg.norm(U_r, axis=0, keepdims=True)

#     return U_r, sigma_r, V_r

# # -------------------------------------------
# # 2) สร้างเมทริกซ์ rank-k: A_k = Σ σ_i u_i v_i^T
# # -------------------------------------------
# def reconstruct_rank_k(U_r, sigma_r, V_r, k: int):
#     k_eff = int(min(k, len(sigma_r)))
#     if k_eff == 0:
#         return np.zeros((U_r.shape[0], V_r.shape[0])), 0

#     Ak = np.zeros((U_r.shape[0], V_r.shape[0]))

#     for i in range(k_eff):
#         ui = U_r[:, i][:, None]
#         vi = V_r[:, i][None, :]
#         Ak += sigma_r[i] * (ui @ vi)

#     return Ak, k_eff

# # -------------------------------------------
# # 3) โหลดภาพ + ทำ grayscale
# # -------------------------------------------
# img = cv2.imread("rdr2.jpg", cv2.IMREAD_GRAYSCALE)
# if img is None:
#     raise FileNotFoundError("Can't find image")

# A = img.astype(np.float64)

# # จะลอง k หลายค่า
# k_list = [1, 5, 20, 50, 100]

# compressed_list = []

# # -------------------------------------------
# # 4) ทำ SVD 1 channel (ภาพขาวดำ)
# # -------------------------------------------
# U_r, sigma_r, V_r = svd_from_eigendecomp(A)

# for k in k_list:
#     Ak, k_eff = reconstruct_rank_k(U_r, sigma_r, V_r, k)
#     Ak = np.clip(Ak, 0, 255).astype(np.uint8)
#     compressed_list.append((k, Ak))

# # -------------------------------------------
# # 5) แสดงผลเปรียบเทียบหลาย k
# # -------------------------------------------
# n_cols = 1 + len(k_list)
# plt.figure(figsize=(3 * n_cols, 4))

# # Original
# plt.subplot(1, n_cols, 1)
# plt.imshow(A, cmap="gray")
# plt.title("Original")
# plt.axis("off")

# # Compressed
# for i, (k, Ak) in enumerate(compressed_list, start=2):
#     plt.subplot(1, n_cols, i)
#     plt.imshow(Ak, cmap="gray")
#     plt.title(f"k = {k}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()
