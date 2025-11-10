import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

mu0 = 4*np.pi*1e-7  # Hằng số từ thẩm
# ============================================================
# 🔹 GIAO DIỆN TKINTER
# ============================================================
def start_gui():
    root = tk.Tk()
    root.title("Mô phỏng từ trường vòng dây (Biot–Savart)")
    root.geometry("400x400")

    ttk.Label(root, text="Cường độ dòng I (A):").pack()
    entry_I = ttk.Entry(root); entry_I.insert(0, "5"); entry_I.pack()

    ttk.Label(root, text="Chiều dòng ('o' hoặc 'ro'):").pack()
    entry_dir = ttk.Entry(root); entry_dir.insert(0, "ro"); entry_dir.pack()

    ttk.Label(root, text="Bán kính vòng dây R (m):").pack()
    entry_R = ttk.Entry(root); entry_R.insert(0, "0.05"); entry_R.pack()

    ttk.Label(root, text="Số đoạn chia nhỏ N:").pack()
    entry_N = ttk.Entry(root); entry_N.insert(0, "200"); entry_N.pack()

    ttk.Label(root, text="Giới hạn không gian hiển thị L (m):").pack()
    entry_L = ttk.Entry(root); entry_L.insert(0, "0.1"); entry_L.pack()

    ttk.Label(root, text="Độ phân giải lưới (res):").pack()
    entry_res = ttk.Entry(root); entry_res.insert(0, "6"); entry_res.pack()

    ttk.Label(root, text="Tọa độ điểm quan sát (x,y,z) (vd: 0,0,0.05):").pack()
    entry_obs = ttk.Entry(root); entry_obs.insert(0, "0,0,0.05"); entry_obs.pack()

    def run_single():
        I = float(entry_I.get())
        direction = entry_dir.get()
        R = float(entry_R.get())
        N = int(entry_N.get())
        L = float(entry_L.get())
        obs_point = tuple(map(float, entry_obs.get().split(',')))
        simulate_single_point(I, direction, R, N, obs_point,L)

    def run_distribution():
        I = float(entry_I.get())
        direction = entry_dir.get()
        R = float(entry_R.get())
        N = int(entry_N.get())
        L = float(entry_L.get())
        res = int(entry_res.get())
        simulate_field_distribution(I, direction, R, N, L, res)

    ttk.Button(root, text="🔴 Mô phỏng từ trường tại 1 điểm", command=run_single).pack(pady=10)
    ttk.Button(root, text="🟢 Mô phỏng phân bố từ trường", command=run_distribution).pack(pady=10)
    ttk.Button(root, text="Thoát", command=root.destroy).pack(pady=10)

    root.mainloop()

# ============================================================
# 🔹 HÀM MÔ PHỎNG 1: TỪ TRƯỜNG TẠI MỘT ĐIỂM QUAN SÁT
# ============================================================
def simulate_single_point(I, direction, R, N, obs_point,L):
    phi = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = np.zeros_like(phi)

    dphi = 2*np.pi / N
    dx = -R * np.sin(phi) * dphi
    dy =  R * np.cos(phi) * dphi
    dz = np.zeros_like(phi)

    # Chiều dòng điện
    if direction == 'o':  
        dx, dy = -dx, -dy

    B = np.zeros(3)
    for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
        r_vec = np.array(obs_point) - np.array([xi, yi, zi])
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0: 
            continue
        dL = np.array([dlx, dly, dlz])
        dB = mu0 * I / (4*np.pi) * np.cross(dL, r_vec) / (r_mag**3)
        B += dB
    xP, yP, zP = obs_point
    

    B_micro = B * 1e6  # đổi sang microTesla


    # Vẽ hình
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Thí nghiệm 1: Từ trường tại một điểm quan sát')
    ax.plot(x, y, z, color='blue', lw=2, label='Vòng dây mang dòng I')
    # Vẽ vectơ B tại điểm quan sát
    ax.quiver(*obs_point, *B, color='purple', length=R*0.5, normalize=True)
    ax.text2D(0.05, 0.9, f"B = ({B_micro[0]:.3f}, {B_micro[1]:.3f}, {B_micro[2]:.3f}) μT",
              transform=ax.transAxes, color='black', fontsize=11)
    # Vẽ dℓ (vectơ tiếp tuyến)
    for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
        ax.quiver(xi, yi, zi, dlx, dly, dlz, length=R*0.2, color='orange', arrow_length_ratio=0.4, normalize=True)
    

    ax.scatter([xP], [yP], [zP], color='green', s=60, label='Điểm quan sát P')
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
    plt.show()

# ============================================================
# 🔹 HÀM MÔ PHỎNG 2: PHÂN BỐ TỪ TRƯỜNG TRONG KHÔNG GIAN
# ============================================================
def simulate_field_distribution(I, direction, R, N, L, res):
    phi = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = np.zeros_like(phi)

    dphi = 2*np.pi / N
    dx = -R * np.sin(phi) * dphi
    dy =  R * np.cos(phi) * dphi
    dz = np.zeros_like(phi)

    if direction == 'o':
        dx, dy = -dx, -dy

    xs = np.linspace(-L, L, res)
    ys = np.linspace(-L, L, res)
    zs = np.linspace(-L, L, res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(Z)
    

    for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
        r_vec = np.stack((X - xi, Y - yi, Z - zi), axis=-1)
        r_mag = np.linalg.norm(r_vec, axis=-1)
        r_mag[r_mag == 0] = np.inf
        dL = np.array([dlx, dly, dlz])
        cross = np.cross(np.broadcast_to(dL, r_vec.shape), r_vec)
        dB = mu0 * I / (4*np.pi) * cross / (r_mag[..., None]**3)
        Bx += dB[..., 0]
        By += dB[..., 1]
        Bz += dB[..., 2]

    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    max_B = np.max(B_mag)
    scale = L * 0.3 / max_B

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Thí nghiệm 2: Phân bố từ trường quanh vòng dây')
    ax.plot(x, y, z, color='blue', lw=2, label='Vòng dây')
    ax.plot(x, y ,z, color='black', lw=2, label='Vectơ từ trường B')
    step = max(1, res//5)
    for i in range(0, res, step):
        for j in range(0, res, step):
            for k in range(0, res, step):
                bx, by, bz = Bx[i,j,k], By[i,j,k], Bz[i,j,k]
                ax.quiver(X[i,j,k], Y[i,j,k], Z[i,j,k],bx*scale, by*scale, bz*scale,color='black', length=1, normalize=False, arrow_length_ratio=0.4)
    for xi, yi, zi, dlx, dly, dlz in zip(x, y, z, dx, dy, dz):
        ax.quiver(xi, yi, zi, dlx, dly, dlz,length=R*0.2, color='orange', arrow_length_ratio=0.4, normalize=True)
    
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.tight_layout()
    plt.show()



# ============================================================
# CHẠY GIAO DIỆN
# ============================================================
if __name__ == "__main__":
    start_gui()
