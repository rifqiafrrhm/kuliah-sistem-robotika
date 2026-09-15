"""
================================================================================
 SIMULASI 3D NU-SPIDERCAM
 Berdasarkan: Bai et al. (2019), "NU-Spidercam: A large-scale, cable-driven,
 integrated sensing and robotic system for advanced phenotyping, remote
 sensing, and agronomic research", Computers and Electronics in Agriculture.

 Simulasi ini memodelkan:
   - Lapangan inti (core imaging area) 60 x 67 m, dibagi 128 zona (4.6 x 6.1 m)
   - 4 tiang di sudut lapangan, tinggi 27 m
   - Platform sensor digantung oleh 8 kabel Kevlar (2 kabel per tiang)
   - Ketinggian platform 0-9 m, kecepatan maksimum 2 m/s
   - Akurasi posisi +-5 cm (disimulasikan sbg noise kecil)
   - Mode gerak "stop-measure-go": platform berpindah ke waypoint, berhenti,
     stabil (~5 detik), lalu "mengambil data" sensor (multispektral, termal,
     LiDAR, spektrometer) sebelum melanjutkan ke titik berikutnya.

 Output: animasi 3D (matplotlib) + file GIF hasil animasi.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from matplotlib import animation
from dataclasses import dataclass, field


# ------------------------------------------------------------------------
# 1. PARAMETER FASILITAS (diambil dari Tabel/teks paper)
# ------------------------------------------------------------------------
FIELD_LENGTH_X = 67.0        # m  (sisi lapangan arah X)
FIELD_LENGTH_Y = 60.0        # m  (sisi lapangan arah Y)
POLE_HEIGHT = 27.0           # m  (tinggi tiang)
PLATFORM_Z_MIN, PLATFORM_Z_MAX = 0.0, 9.0   # m, rentang ketinggian platform
MAX_SPEED = 2.0              # m/s, kecepatan maksimum platform
POSITION_ACCURACY = 0.05     # m, akurasi posisi (+-5 cm)
ZONE_DX, ZONE_DY = 6.1, 4.6  # m, ukuran satu zona (128 zona total)
N_ZONES_X = int(round(FIELD_LENGTH_X / ZONE_DX))   # ~ 11
N_ZONES_Y = int(round(FIELD_LENGTH_Y / ZONE_DY))   # ~ 13
STABILIZE_TIME = 1.5         # s (dipersingkat utk animasi; asli 3-10 s)
FLIGHT_HEIGHT = 5.0          # m, tinggi terbang platform saat 2018 (5 m)

# Posisi 4 tiang di sudut lapangan (x, y, z=tinggi tiang)
POLES = np.array([
    [0.0,            0.0,            POLE_HEIGHT],
    [FIELD_LENGTH_X, 0.0,            POLE_HEIGHT],
    [FIELD_LENGTH_X, FIELD_LENGTH_Y, POLE_HEIGHT],
    [0.0,            FIELD_LENGTH_Y, POLE_HEIGHT],
])


# ------------------------------------------------------------------------
# 2. STRUKTUR DATA
# ------------------------------------------------------------------------
@dataclass
class SensorReading:
    """Data yang disimulasikan mengikuti sensor onboard NU-Spidercam
    (Tabel 1 paper): kamera multispektral, kamera termal, LiDAR, spektrometer."""
    ndvi: float = 0.0
    canopy_temp_C: float = 0.0
    soil_temp_C: float = 0.0
    canopy_height_m: float = 0.0
    canopy_cover_pct: float = 0.0


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    pan: float = 0.0
    tilt: float = -90.0     # nadir (menghadap ke bawah), sesuai paper
    label: str = ""


@dataclass
class Zone:
    """Merepresentasikan satu dari 128 zona (4.6 x 6.1 m) pada lapangan."""
    row: int
    col: int
    cx: float   # koordinat pusat zona (x)
    cy: float   # koordinat pusat zona (y)


# ------------------------------------------------------------------------
# 3. PEMBUATAN PETA ZONA & WAY-POINT MAP (analog file .txt pada paper)
# ------------------------------------------------------------------------
def build_zone_grid():
    """Membagi lapangan inti menjadi grid zona (analog 128 zona di paper)."""
    zones = []
    for r in range(N_ZONES_Y):
        for c in range(N_ZONES_X):
            cx = (c + 0.5) * ZONE_DX
            cy = (r + 0.5) * ZONE_DY
            zones.append(Zone(row=r, col=c, cx=cx, cy=cy))
    return zones


def build_serpentine_waypoints(zones, n_selected=24, height=FLIGHT_HEIGHT):
    """Membuat 'Way-Point Map' berpola serpentine (naik-turun berselang-seling
    per baris), seperti mode task-planning pada software LabVIEW di paper."""
    zones_by_row = {}
    for z in zones:
        zones_by_row.setdefault(z.row, []).append(z)

    waypoints = []
    rows_sorted = sorted(zones_by_row.keys())
    for i, r in enumerate(rows_sorted):
        row_zones = sorted(zones_by_row[r], key=lambda z: z.col)
        if i % 2 == 1:
            row_zones = row_zones[::-1]     # serpentine: baris genap dibalik
        for z in row_zones:
            waypoints.append(Waypoint(x=z.cx, y=z.cy, z=height,
                                       label=f"Z{z.row:02d}-{z.col:02d}"))
    # ambil subset agar animasi tidak terlalu panjang
    step = max(1, len(waypoints) // n_selected)
    return waypoints[::step][:n_selected]


# ------------------------------------------------------------------------
# 4. MODEL KINEMATIKA KABEL (cable-driven positioning)
# ------------------------------------------------------------------------
def cable_lengths(platform_pos, poles=POLES):
    """Menghitung panjang tiap kabel dari 4 tiang ke platform (2 kabel per
    tiang -> 8 kabel Kevlar seperti disebut di paper, disini disederhanakan
    2 titik jangkar per tiang dengan sedikit offset horizontal)."""
    lengths = []
    anchors = []
    offset = 0.4  # m, jarak antar 2 titik jangkar kabel pada tiang yang sama
    for p in poles:
        for sign in (-1, 1):
            anchor = p + np.array([sign * offset, sign * offset, 0.0])
            anchors.append(anchor)
            lengths.append(np.linalg.norm(anchor - platform_pos))
    return np.array(lengths), np.array(anchors)


def move_platform(start, end, speed=MAX_SPEED, max_steps=12):
    """Interpolasi linier lintasan platform antara dua titik. Waktu tempuh
    riil (dist/speed, sesuai kecepatan maks 2 m/s pada paper) dihitung untuk
    keperluan log, namun jumlah frame animasi dibatasi (max_steps) supaya
    animasi tetap ringan berapa pun jarak antar plot."""
    start, end = np.array(start), np.array(end)
    dist = np.linalg.norm(end - start)
    travel_time_s = dist / speed
    n_steps = max(2, min(max_steps, int(np.ceil(dist / 3.0)) + 2))
    path = []
    for s in range(n_steps + 1):
        t = s / n_steps
        pos = start + t * (end - start)
        noise = np.random.uniform(-POSITION_ACCURACY, POSITION_ACCURACY, size=3)
        path.append(pos + noise * (0 if s in (0, n_steps) else 1))
    return path, travel_time_s


def simulate_sensors(waypoint, day_fraction):
    """Mensimulasikan pembacaan sensor multispektral/termal/LiDAR pada satu
    plot, dengan pola diurnal kasar mengikuti Fig. 11/12 di paper (suhu tanah
    > suhu kanopi, puncak sekitar tengah hari)."""
    ndvi = np.clip(0.3 + 0.5 * np.sin(np.pi * day_fraction) + np.random.normal(0, 0.02), 0, 1)
    canopy_cover = np.clip(35 + 60 * day_fraction + np.random.normal(0, 2), 0, 100)
    canopy_temp = 26 + 8 * np.sin(np.pi * day_fraction) + np.random.normal(0, 0.5)
    soil_temp = canopy_temp + 6 + np.random.normal(0, 0.5)      # tanah > kanopi
    canopy_height = 0.4 + 0.6 * day_fraction + np.random.normal(0, 0.01)
    return SensorReading(ndvi=ndvi, canopy_temp_C=canopy_temp, soil_temp_C=soil_temp,
                          canopy_height_m=canopy_height, canopy_cover_pct=canopy_cover)


# ------------------------------------------------------------------------
# 5. VISUALISASI & ANIMASI 3D
# ------------------------------------------------------------------------
def run_simulation_and_animate(save_gif_path="/mnt/user-data/outputs/spidercam_simulation.gif"):
    np.random.seed(42)

    zones = build_zone_grid()
    waypoints = build_serpentine_waypoints(zones, n_selected=10, height=FLIGHT_HEIGHT)

    # Bangun full path (stop-measure-go): gerak + jeda stabilisasi di tiap titik
    full_path = []
    events = []   # ('move'/'stabilize', waypoint_index)
    current = np.array([waypoints[0].x, waypoints[0].y, 0.0])
    total_travel_time = 0.0
    n_stab_frames = 4   # frame jeda "stabilize" per plot (ilustratif)
    for i, wp in enumerate(waypoints):
        target = np.array([wp.x, wp.y, wp.z])
        seg, t_travel = move_platform(current, target)
        total_travel_time += t_travel + STABILIZE_TIME
        for p in seg:
            full_path.append(p)
            events.append(("move", i))
        for _ in range(n_stab_frames):
            full_path.append(target)
            events.append(("stabilize", i))
        current = target

    full_path = np.array(full_path)
    log = []   # log data sensor (analog folder data hasil pengukuran)

    # ---------------- Setup figure ----------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((FIELD_LENGTH_X, FIELD_LENGTH_Y, POLE_HEIGHT * 0.6))

    def draw_static_scene():
        ax.clear()
        ax.set_xlim(0, FIELD_LENGTH_X)
        ax.set_ylim(0, FIELD_LENGTH_Y)
        ax.set_zlim(0, POLE_HEIGHT)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z / Tinggi (m)")
        ax.set_title("Simulasi 3D NU-Spidercam\n(Cable-driven Field Phenotyping Platform)")

        # tiang
        for p in POLES:
            ax.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color="dimgray", lw=4)
            ax.scatter(*p, color="black", s=25)

        # garis batas lapangan inti
        gx = [0, FIELD_LENGTH_X, FIELD_LENGTH_X, 0, 0]
        gy = [0, 0, FIELD_LENGTH_Y, FIELD_LENGTH_Y, 0]
        ax.plot(gx, gy, [0] * 5, color="saddlebrown", lw=1.5)

        # grid zona (128 zona) sebagai garis tipis di permukaan tanah
        for c in range(N_ZONES_X + 1):
            x = c * ZONE_DX
            if x <= FIELD_LENGTH_X:
                ax.plot([x, x], [0, FIELD_LENGTH_Y], [0, 0], color="peru", lw=0.3)
        for r in range(N_ZONES_Y + 1):
            y = r * ZONE_DY
            if y <= FIELD_LENGTH_Y:
                ax.plot([0, FIELD_LENGTH_X], [y, y], [0, 0], color="peru", lw=0.3)

        # jejak lintasan (trajectory) yang sudah dilalui
        ax.plot(full_path[:, 0], full_path[:, 1], full_path[:, 2],
                color="royalblue", lw=0.6, alpha=0.4)

    draw_static_scene()

    platform_dot = ax.scatter([], [], [], color="red", s=80, label="Sensor platform")
    cable_lines = [ax.plot([], [], [], color="gold", lw=1)[0] for _ in range(8)]
    status_text = fig.text(0.02, 0.95, "", fontsize=10, family="monospace",
                            va="top", ha="left")

    def update(frame):
        pos = full_path[frame]
        mode, wp_idx = events[frame]
        wp = waypoints[wp_idx]

        # update posisi platform
        platform_dot._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # update 8 kabel dari tiang ke platform
        _, anchors = cable_lengths(pos)
        for line, anchor in zip(cable_lines, anchors):
            line.set_data([anchor[0], pos[0]], [anchor[1], pos[1]])
            line.set_3d_properties([anchor[2], pos[2]])

        # bila platform sedang "stabilize" di akhir segmen -> ambil data sensor
        info_line = ""
        is_first_stabilize_frame = mode == "stabilize" and events[frame - 1][0] == "move"
        if mode == "stabilize":
            day_fraction = wp_idx / max(1, len(waypoints) - 1)
            if is_first_stabilize_frame:
                reading = simulate_sensors(wp, day_fraction)
                log.append((wp.label, reading))
            reading = log[-1][1]
            info_line = (f"[MEASURE] Plot {wp.label} | NDVI={reading.ndvi:.2f} "
                         f"| Tkanopi={reading.canopy_temp_C:.1f}C "
                         f"| Ttanah={reading.soil_temp_C:.1f}C "
                         f"| Tinggi={reading.canopy_height_m:.2f}m "
                         f"| Cover={reading.canopy_cover_pct:.0f}%")
        else:
            info_line = f"[MOVE] -> menuju plot {wp.label} ..."

        status_text.set_text(
            "NU-SPIDERCAM SIMULATION\n"
            f"Mode      : {'stop-measure-go'}\n"
            f"Posisi    : X={pos[0]:5.2f} Y={pos[1]:5.2f} Z={pos[2]:4.2f} m\n"
            f"Kecepatan maks : {MAX_SPEED} m/s | Akurasi: +-{POSITION_ACCURACY*100:.0f} cm\n"
            f"{info_line}"
        )
        return [platform_dot, *cable_lines, status_text]

    anim = animation.FuncAnimation(fig, update, frames=len(full_path),
                                    interval=40, blit=False, repeat=True)

    try:
        anim.save(save_gif_path, writer="pillow", fps=20)
        print(f"Animasi tersimpan di: {save_gif_path}")
    except Exception as e:
        print(f"Gagal menyimpan GIF ({e}); menampilkan plot langsung sebagai gantinya.")
        plt.show()

    return log


# ------------------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------------------
if __name__ == "__main__":
    print("Menjalankan simulasi 3D NU-Spidercam ...")
    print(f"Lapangan inti : {FIELD_LENGTH_X} x {FIELD_LENGTH_Y} m")
    print(f"Jumlah zona   : {N_ZONES_X} x {N_ZONES_Y} = {N_ZONES_X*N_ZONES_Y} (paper: 128)")
    print(f"Tinggi tiang  : {POLE_HEIGHT} m | Rentang tinggi platform: "
          f"{PLATFORM_Z_MIN}-{PLATFORM_Z_MAX} m")
    print(f"Kecepatan maks platform: {MAX_SPEED} m/s\n")

    hasil_log = run_simulation_and_animate()

    print("\nContoh hasil pengukuran (5 plot pertama):")
    for label, reading in hasil_log[:5]:
        print(f"  {label}: NDVI={reading.ndvi:.2f}, "
              f"Tkanopi={reading.canopy_temp_C:.1f}C, "
              f"Ttanah={reading.soil_temp_C:.1f}C, "
              f"Tinggi={reading.canopy_height_m:.2f}m, "
              f"Cover={reading.canopy_cover_pct:.0f}%")
