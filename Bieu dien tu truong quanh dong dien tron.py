import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Nhập tham số từ bàn phím
# -----------------------
I = float(10.0) #float(input("Nhập cường độ dòng điện I (A): "))
direction = input("Nhập chiều dòng điện ('o' = cùng chiều kim đồng hồ, 'ro' = ngược chiều kim đồng hồ): ").strip().lower()
R = float(0.03) #float(input("Nhập bán kính vòng dây R (m): "))
N = int(50)#int(input("Nhập số đoạn chia nhỏ vòng dây N: "))
L = float(0.05)#float(input("Nhập nửa chiều dài trục hiển thị L (m): "))
res = 6#int(input("Nhập độ phân giải lưới (ví dụ: 6 → 6×6×6 điểm): "))

mu0 = 4*np.pi*1e-7  # Hằng số từ thẩm

# -----------------------
# Chia nhỏ vòng dây
# -----------------------
phi = np.linspace(0, 2*np.pi, N, endpoint=False)
x = R * np.cos(phi)
y = R * np.sin(phi)
z = np.zeros_like(phi)

dphi = 2*np.pi / N
dx = -R * np.sin(phi) * dphi
dy =  R * np.cos(phi) * dphi
dz = np.zeros_like(phi)

# Điều chỉnh chiều dòng điện
if direction == "o":  # cùng chiều kim đồng hồ (nhìn từ +z)
    dx = -dx
    dy = -dy

# -----------------------
# Tạo lưới điểm trong không gian
# -----------------------
xs = np.linspace(-L, L, res)
ys = np.linspace(-L, L, res)
zs = np.linspace(-L, L, res)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
Bz = np.zeros_like(Z)

# -----------------------
# Tính từ trường tại từng điểm (Biot–Savart)
# -----------------------
for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
    r_vec = np.stack((X - xi, Y - yi, Z - zi), axis=-1)
    r_mag = np.linalg.norm(r_vec, axis=-1)
    r_mag[r_mag == 0] = np.inf  # tránh chia 0
    dL = np.array([dlx, dly, dlz])
    cross = np.cross(np.broadcast_to(dL, r_vec.shape), r_vec)
    dB = mu0 * I / (4*np.pi) * cross / (r_mag[..., None]**3)
    Bx += dB[..., 0]
    By += dB[..., 1]
    Bz += dB[..., 2]

# -----------------------
# Chuẩn tỉ lệ hiển thị (như trước)
# -----------------------
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
max_B = np.max(B_mag)
scale = L * 0.3 / max_B  # tỉ lệ phóng đại chung

# -----------------------
# Vẽ kết quả
# -----------------------

    
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Phân bố từ trường quanh vòng dây (Biot–Savart)')

# Vẽ vòng dây
ax.plot(x, y, z, color='blue', lw=2, label='Vòng dây mang dòng I')
ax.plot(x, y ,z, color='black', lw=2, label='Vectơ từ trường B')
ax.plot(x, y ,z, color='orange', lw=2, label='Vectơ độ dài dL')
scale_dL = 0.2 * R
# Vẽ các vectơ dL
for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
    ax.quiver(xi, yi, zi, dlx, dly, dlz,
              length=scale_dL, color='orange', arrow_length_ratio=0.4, normalize=True)
# Vẽ vectơ từ trường
step = max(1, res//5)  # giảm mật độ để dễ nhìn
for i in range(0, res, step):
    for j in range(0, res, step):
        for k in range(0, res, step):
            bx, by, bz = Bx[i,j,k], By[i,j,k], Bz[i,j,k]
            ax.quiver(X[i,j,k], Y[i,j,k], Z[i,j,k],
                      bx*scale, by*scale, bz*scale,
                      color='black', length=1, normalize=False, arrow_length_ratio=0.4)
# Cài đặt tỉ lệ trục
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, L)
try:
    ax.set_box_aspect([1,1,1])
except Exception:
    pass

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.tight_layout()
plt.show()
