import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Nhập tham số từ bàn phím
# -----------------------
I = float(input("Nhập cường độ dòng điện I (A): "))
direction = input("Nhập chiều dòng điện ('o' = cùng chiều kim đồng hồ, 'ro' = ngược chiều kim đồng hồ): ").strip().lower()
R = float(input("Nhập bán kính vòng dây R (m): "))
N = int(input("Nhập số đoạn chia nhỏ N: "))
xP = float(input("Nhập tọa độ điểm cần tính x (m): "))
yP = float(input("Nhập tọa độ điểm cần tính y (m): "))
zP = float(input("Nhập tọa độ điểm cần tính z (m): "))
L = float(input("Nhập nửa chiều dài trục hiển thị L (m): "))

mu0 = 4*np.pi*1e-7  # Hằng số từ thẩm
P = np.array([xP, yP, zP])

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

# -----------------------
# Điều chỉnh chiều dòng điện
# -----------------------
if direction == "o":  # cùng chiều kim đồng hồ (nhìn từ +z)
    dx = -dx
    dy = -dy
# -----------------------
# Tính dB của từng đoạn tại điểm P
# -----------------------
dB_list = []
for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
    r_vec = P - np.array([xi, yi, zi])
    r = np.linalg.norm(r_vec)
    if r == 0:
        continue
    dL = np.array([dlx, dly, dlz])
    dB = mu0 * I / (4*np.pi) * np.cross(dL, r_vec) / (r**3)
    dB_list.append(dB)

dB_list = np.array(dB_list)
B_tot_T = np.sum(dB_list, axis=0)
B_tot_uT = B_tot_T * 1e6  # microtesla

# -----------------------
# Xác định hệ số phóng đại chung
# -----------------------
max_B = max(np.linalg.norm(B_tot_T), np.max(np.linalg.norm(dB_list, axis=1)))
scale = L * 1 / max_B  # phóng đại tất cả vectơ sao cho vừa khung hình

# -----------------------
# Vẽ 3D
# -----------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Mô phỏng từ trường vòng dây (Biot–Savart)')

# Vẽ vòng dây
ax.plot(x, y, z, color='blue', lw=2, label='Vòng dây')

# Vẽ dℓ (vectơ tiếp tuyến)
for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
    ax.quiver(xi, yi, zi, dlx, dly, dlz,length=R*0.2, color='orange', arrow_length_ratio=0.4, normalize=True)


# Vẽ B tổng tại P (đen, cùng tỷ lệ)
ax.quiver(xP, yP, zP,
          B_tot_T[0]*scale, B_tot_T[1]*scale, B_tot_T[2]*scale,
          color='black', linewidth=3, arrow_length_ratio=0.2, label='B tổng')

# Vẽ điểm quan sát
ax.scatter([xP], [yP], [zP], color='green', s=60, label='Điểm quan sát P')

# Trục toạ độ có tỉ lệ đều nhau
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

# Hiển thị giá trị B
text = f"Bx = {B_tot_uT[0]:.4f} µT\nBy = {B_tot_uT[1]:.4f} µT\nBz = {B_tot_uT[2]:.4f} µT\n|B| = {np.linalg.norm(B_tot_uT):.4f} µT"
ax.text2D(0.02, 0.95, text, transform=ax.transAxes, fontsize=10, va='top')

ax.legend()
plt.tight_layout()
plt.show()
