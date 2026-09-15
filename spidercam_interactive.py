"""
================================================================================
 SIMULASI 3D INTERAKTIF - NU-SPIDERCAM
 Berdasarkan: Bai et al. (2019), "NU-Spidercam: A large-scale, cable-driven,
 integrated sensing and robotic system for advanced phenotyping, remote
 sensing, and agronomic research", Computers and Electronics in Agriculture.

 TUJUAN PROGRAM
 ---------------
 Program ini BUKAN animasi otomatis / GIF, melainkan simulasi yang bisa
 Anda KONTROL SENDIRI secara langsung (real-time) lewat keyboard, supaya
 Anda bisa memahami cara kerja NU-Spidercam meskipun tidak punya alatnya:

   1. Platform sensor digantung oleh 8 kabel Kevlar dari 4 tiang di sudut
      lapangan (tinggi tiang 27 m). Saat Anda menggerakkan platform,
      kabel-kabel ini otomatis menyesuaikan panjang & sudutnya - inilah
      prinsip "cable-driven positioning" yang dijelaskan di paper.
   2. Platform bisa bergerak di area X-Y lapangan (60 x 67 m) dan naik-turun
      pada rentang tinggi 0-9 m (sesuai spesifikasi asli).
   3. Unit pan/tilt di bagian bawah platform bisa diarahkan (nadir/lurus ke
      bawah, atau miring), meniru kemampuan "multi-angle imaging" yang
      disebut di paper (Bagian 2.2 & Future Development).
   4. Jejak pandang kamera (field of view) diproyeksikan ke tanah sebagai
      kotak kuning - berubah posisi & ukurannya sesuai tinggi dan sudut
      kamera. Ini membantu memahami area yang sebenarnya "dipotret" oleh
      kamera multispektral pada satu waktu.
   5. Tekan tombol "ukur" untuk mensimulasikan pengambilan data sensor
      (NDVI, suhu kanopi/tanah, tinggi kanopi) di plot yang sedang dibidik,
      meniru mode "stop-measure-go" pada paper.

 KONTROL KEYBOARD
 -----------------
   W / Panah Atas     -> platform gerak +Y   (maju)
   S / Panah Bawah     -> platform gerak -Y   (mundur)
   A / Panah Kiri      -> platform gerak -X   (kiri)
   D / Panah Kanan     -> platform gerak +X   (kanan)
   R                   -> platform naik   (+Z)
   F                   -> platform turun  (-Z)
   J                   -> kamera PAN kiri
   L                   -> kamera PAN kanan
   I                   -> kamera TILT naik (mendekati horizontal)
   K                   -> kamera TILT turun (mendekati nadir/lurus bawah)
   M                   -> ambil data sensor (measure) di titik yang dibidik
   H                   -> tampilkan/hilangkan panel bantuan
   Q / Esc             -> keluar program

 Jalankan dengan:
     python spidercam_interactive.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registrasi proyeksi 3D)
from dataclasses import dataclass


# ------------------------------------------------------------------------
# 1. PARAMETER FASILITAS (sesuai Tabel & teks pada paper)
# ------------------------------------------------------------------------
FIELD_LENGTH_X = 67.0        # m, sisi lapangan arah X
FIELD_LENGTH_Y = 60.0        # m, sisi lapangan arah Y
POLE_HEIGHT = 27.0           # m, tinggi tiang penyangga kabel
PLATFORM_Z_MIN, PLATFORM_Z_MAX = 0.0, 9.0   # m, rentang tinggi platform
POSITION_ACCURACY = 0.05     # m, akurasi posisi real alat (+-5 cm)
ZONE_DX, ZONE_DY = 6.1, 4.6  # m, ukuran 1 zona/plot (~128 zona total di paper)
N_ZONES_X = int(round(FIELD_LENGTH_X / ZONE_DX))
N_ZONES_Y = int(round(FIELD_LENGTH_Y / ZONE_DY))
DEFAULT_HEIGHT = 5.0         # m, tinggi terbang platform (setting 2018)

# Field of View kamera multispektral (Tabel 1 paper): 44.9 deg (H) x 34.0 deg (V)
CAM_FOV_H = 44.9
CAM_FOV_V = 34.0

STEP_XY = 1.5     # m, langkah gerak per tekan tombol
STEP_Z = 0.3      # m, langkah naik/turun per tekan tombol
STEP_ANGLE = 5.0  # derajat, langkah pan/tilt per tekan tombol

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
    """Data tersimulasi mengikuti sensor onboard NU-Spidercam (Tabel 1
    paper): kamera multispektral, kamera termal, LiDAR."""
    ndvi: float
    canopy_temp_C: float
    soil_temp_C: float
    canopy_height_m: float
    canopy_cover_pct: float


class PlatformState:
    """Menyimpan posisi (x, y, z) dan orientasi kamera (pan, tilt) platform
    sensor NU-Spidercam."""

    def __init__(self):
        self.x = FIELD_LENGTH_X / 2
        self.y = FIELD_LENGTH_Y / 2
        self.z = DEFAULT_HEIGHT
        self.pan = 0.0     # derajat, rotasi horizontal arah kamera
        self.tilt = -90.0  # derajat, -90 = nadir (lurus ke bawah, spt paper)

    def position(self):
        return np.array([self.x, self.y, self.z])

    def clamp(self):
        self.x = float(np.clip(self.x, 0.0, FIELD_LENGTH_X))
        self.y = float(np.clip(self.y, 0.0, FIELD_LENGTH_Y))
        self.z = float(np.clip(self.z, PLATFORM_Z_MIN, PLATFORM_Z_MAX))
        # tilt dibatasi -90 (nadir) sampai 0 (horizontal), sesuai kemampuan
        # pan/tilt unit yang disebut di paper untuk "multi-angle imaging"
        self.tilt = float(np.clip(self.tilt, -90.0, 0.0))
        self.pan = float(self.pan % 360.0)


# ------------------------------------------------------------------------
# 3. KINEMATIKA KABEL (cable-driven positioning)
# ------------------------------------------------------------------------
def cable_anchors_and_lengths(pos, poles=POLES, offset=0.4):
    """Menghitung titik jangkar & panjang tiap kabel dari 4 tiang ke
    platform. Disederhanakan: tiap tiang punya 2 titik jangkar (offset kecil
    horizontal) sehingga totalnya 8 kabel, sesuai jumlah kabel Kevlar pada
    NU-Spidercam asli."""
    anchors, lengths = [], []
    for p in poles:
        for sign in (-1, 1):
            anchor = p + np.array([sign * offset, sign * offset, 0.0])
            anchors.append(anchor)
            lengths.append(np.linalg.norm(anchor - pos))
    return np.array(anchors), np.array(lengths)


# ------------------------------------------------------------------------
# 4. GEOMETRI KAMERA: menghitung titik bidik & jejak FOV di tanah
# ------------------------------------------------------------------------
def camera_look_at_ground(state):
    """Menghitung titik di tanah (z=0) yang sedang dibidik kamera, beserta
    ukuran perkiraan jejak pandang (footprint) berdasarkan tinggi platform,
    sudut pan/tilt, dan FOV kamera multispektral (44.9 x 34.0 derajat).
    Mengembalikan None jika kamera mengarah ke atas (tidak menyentuh tanah).
    """
    pan_rad = np.radians(state.pan)
    tilt_rad = np.radians(state.tilt)   # -90 deg = lurus ke bawah

    # vektor arah kamera sebelum rotasi pan (menghadap +X, dimiringkan tilt)
    dir_local = np.array([np.cos(tilt_rad), 0.0, np.sin(tilt_rad)])
    # rotasi horizontal (pan) mengelilingi sumbu Z
    dx = dir_local[0] * np.cos(pan_rad) - dir_local[1] * np.sin(pan_rad)
    dy = dir_local[0] * np.sin(pan_rad) + dir_local[1] * np.cos(pan_rad)
    dz = dir_local[2]

    pos = state.position()
    if dz >= -1e-6:       # kamera horizontal/ke atas -> tidak kena tanah
        return None

    t = -pos[2] / dz               # jarak sepanjang arah pandang sampai z=0
    ground_point = pos + t * np.array([dx, dy, dz])
    slant_range = t                # dir_local adalah vektor satuan

    half_w = slant_range * np.tan(np.radians(CAM_FOV_H / 2))
    half_h = slant_range * np.tan(np.radians(CAM_FOV_V / 2))

    # kotak jejak FOV (sebelum rotasi pan) lalu dirotasi sesuai arah pan
    corners_local = np.array([
        [-half_w, -half_h], [half_w, -half_h],
        [half_w,  half_h], [-half_w,  half_h],
    ])
    cos_p, sin_p = np.cos(pan_rad), np.sin(pan_rad)
    rot = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
    corners_world = corners_local @ rot.T + ground_point[:2]

    return ground_point, corners_world, slant_range


def zone_label_for(x, y):
    col = int(np.clip(x // ZONE_DX, 0, N_ZONES_X - 1))
    row = int(np.clip(y // ZONE_DY, 0, N_ZONES_Y - 1))
    return f"Z{row:02d}-{col:02d}"


# ------------------------------------------------------------------------
# 5. SIMULASI PEMBACAAN SENSOR (analog data yang direkam onboard computer)
# ------------------------------------------------------------------------
def simulate_sensor_reading(ground_xy, rng):
    """Menghasilkan data sensor pura-pura namun realistis dari segi rentang
    nilai (NDVI 0-1, suhu tanah > suhu kanopi seperti temuan Fig. 11/12 di
    paper, tinggi kanopi 0.4-1.0 m seperti Fig. 9)."""
    # gunakan posisi sebagai "benih" spasial supaya nilai konsisten di plot
    # yang sama tiap kali diukur ulang
    spatial_seed = int((ground_xy[0] * 100 + ground_xy[1]) % 10000)
    local_rng = np.random.RandomState(spatial_seed)
    growth = local_rng.uniform(0.2, 0.95)   # tahap pertumbuhan semu

    ndvi = float(np.clip(0.25 + 0.6 * growth + rng.normal(0, 0.02), 0, 1))
    canopy_cover = float(np.clip(30 + 65 * growth + rng.normal(0, 2), 0, 100))
    canopy_temp = float(27 + 6 * growth + rng.normal(0, 0.4))
    soil_temp = float(canopy_temp + 5.5 + rng.normal(0, 0.4))   # tanah > kanopi
    canopy_height = float(0.35 + 0.7 * growth + rng.normal(0, 0.01))

    return SensorReading(ndvi=ndvi, canopy_temp_C=canopy_temp, soil_temp_C=soil_temp,
                          canopy_height_m=canopy_height, canopy_cover_pct=canopy_cover)


# ------------------------------------------------------------------------
# 6. PROGRAM UTAMA - JENDELA 3D INTERAKTIF
# ------------------------------------------------------------------------
def main():
    rng = np.random.RandomState(0)
    state = PlatformState()
    measurement_log = []
    show_help = [True]   # pakai list agar bisa diubah dalam closure

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((FIELD_LENGTH_X, FIELD_LENGTH_Y, POLE_HEIGHT * 0.55))

    def draw_field():
        ax.clear()
        ax.set_xlim(0, FIELD_LENGTH_X)
        ax.set_ylim(0, FIELD_LENGTH_Y)
        ax.set_zlim(0, POLE_HEIGHT)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Tinggi Z (m)")


        # 4 tiang penyangga
        for p in POLES:
            ax.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color="dimgray", lw=4)
            ax.scatter(*p, color="black", s=30)

        # batas lapangan inti
        gx = [0, FIELD_LENGTH_X, FIELD_LENGTH_X, 0, 0]
        gy = [0, 0, FIELD_LENGTH_Y, FIELD_LENGTH_Y, 0]
        ax.plot(gx, gy, [0] * 5, color="saddlebrown", lw=1.5)

        # grid ~128 zona/plot
        for c in range(N_ZONES_X + 1):
            x = min(c * ZONE_DX, FIELD_LENGTH_X)
            ax.plot([x, x], [0, FIELD_LENGTH_Y], [0, 0], color="peru", lw=0.3)
        for r in range(N_ZONES_Y + 1):
            y = min(r * ZONE_DY, FIELD_LENGTH_Y)
            ax.plot([0, FIELD_LENGTH_X], [y, y], [0, 0], color="peru", lw=0.3)

    draw_field()

    platform_dot = ax.scatter([], [], [], color="red", s=110, depthshade=False)
    cable_lines = [ax.plot([], [], [], color="gold", lw=1)[0] for _ in range(8)]
    sight_line, = ax.plot([], [], [], color="cyan", lw=1.2, linestyle="--")
    fov_line, = ax.plot([], [], [], color="lime", lw=2)
    measured_points = ax.scatter([], [], [], color="magenta", s=25, label="Titik terukur")

    info_text = fig.text(0.015, 0.97, "", fontsize=9.5, family="monospace",
                          va="top", ha="left")
    help_text = fig.text(0.015, 0.02, "", fontsize=8.5, family="monospace",
                          va="bottom", ha="left", color="navy")

    HELP_STRING = (
        "KONTROL: [W/A/S/D atau panah] gerak XY | [R/F] naik/turun Z | "
        "[J/L] pan kiri/kanan | [I/K] tilt naik/turun\n"
        "[M] ambil data sensor di titik bidik | [H] toggle bantuan | [Q/Esc] keluar"
    )

    def redraw(last_action=""):
        pos = state.position()

        platform_dot._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        anchors, _ = cable_anchors_and_lengths(pos)
        for line, anchor in zip(cable_lines, anchors):
            line.set_data([anchor[0], pos[0]], [anchor[1], pos[1]])
            line.set_3d_properties([anchor[2], pos[2]])

        result = camera_look_at_ground(state)
        if result is not None:
            ground_point, fov_corners, slant_range = result
            sight_line.set_data([pos[0], ground_point[0]], [pos[1], ground_point[1]])
            sight_line.set_3d_properties([pos[2], ground_point[2]])

            fx = list(fov_corners[:, 0]) + [fov_corners[0, 0]]
            fy = list(fov_corners[:, 1]) + [fov_corners[0, 1]]
            fov_line.set_data(fx, fy)
            fov_line.set_3d_properties([0] * len(fx))
            plot_label = zone_label_for(ground_point[0], ground_point[1])
            bidik_info = (f"Titik bidik : X={ground_point[0]:5.2f} Y={ground_point[1]:5.2f} "
                          f"(plot {plot_label}) | jarak pandang={slant_range:4.2f} m")
        else:
            sight_line.set_data([], [])
            sight_line.set_3d_properties([])
            fov_line.set_data([], [])
            fov_line.set_3d_properties([])
            bidik_info = "Titik bidik : (kamera tidak mengarah ke tanah)"

        if measurement_log:
            xs = [m[1][0] for m in measurement_log]
            ys = [m[1][1] for m in measurement_log]
            zs = [0.02 for _ in measurement_log]
            measured_points._offsets3d = (xs, ys, zs)

        info_text.set_text(
            "=== NU-SPIDERCAM: SIMULASI INTERAKTIF ===\n"
            f"Posisi platform : X={pos[0]:5.2f}  Y={pos[1]:5.2f}  Z={pos[2]:4.2f} m "
            f"(rentang asli 0-9 m)\n"
            f"Kamera          : pan={state.pan:5.1f} deg | tilt={state.tilt:5.1f} deg "
            f"(-90=nadir lurus bawah)\n"
            f"{bidik_info}\n"
            f"Jumlah titik terukur: {len(measurement_log)}\n"
            f"{last_action}"
        )
        help_text.set_text(HELP_STRING if show_help[0] else "(tekan H untuk bantuan)")
        fig.canvas.draw_idle()

    def on_key(event):
        key = (event.key or "").lower()
        msg = ""

        if key in ("w", "up"):
            state.y += STEP_XY
        elif key in ("s", "down"):
            state.y -= STEP_XY
        elif key in ("a", "left"):
            state.x -= STEP_XY
        elif key in ("d", "right"):
            state.x += STEP_XY
        elif key == "r":
            state.z += STEP_Z
        elif key == "f":
            state.z -= STEP_Z
        elif key == "j":
            state.pan -= STEP_ANGLE
        elif key == "l":
            state.pan += STEP_ANGLE
        elif key == "i":
            state.tilt += STEP_ANGLE     # menuju horizontal (0 deg)
        elif key == "k":
            state.tilt -= STEP_ANGLE     # menuju nadir (-90 deg)
        elif key == "h":
            show_help[0] = not show_help[0]
        elif key == "m":
            result = camera_look_at_ground(state)
            if result is not None:
                ground_point, _, _ = result
                reading = simulate_sensor_reading(ground_point[:2], rng)
                label = zone_label_for(ground_point[0], ground_point[1])
                measurement_log.append((label, ground_point[:2].copy(), reading))
                msg = (f"[UKUR] Plot {label}: NDVI={reading.ndvi:.2f}  "
                       f"Tkanopi={reading.canopy_temp_C:.1f}C  "
                       f"Ttanah={reading.soil_temp_C:.1f}C  "
                       f"Tinggi={reading.canopy_height_m:.2f}m  "
                       f"Cover={reading.canopy_cover_pct:.0f}%")
                print(msg)
            else:
                msg = "[UKUR] Gagal - kamera tidak sedang membidik tanah."
        elif key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return   # tombol lain diabaikan

        state.clamp()
        redraw(msg)

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()

    print("=" * 70)
    print(" SIMULASI INTERAKTIF NU-SPIDERCAM SIAP DIJALANKAN")
    print(" Klik jendela plot 3D lalu gunakan keyboard untuk mengendalikan:")
    print("   W/A/S/D / panah  : gerak platform di bidang X-Y")
    print("   R / F            : naik / turun (Z)")
    print("   J / L            : pan kamera kiri / kanan")
    print("   I / K            : tilt kamera naik / turun")
    print("   M                : ambil data sensor di titik yang dibidik")
    print("   H                : tampilkan/sembunyikan bantuan")
    print("   Q / Esc          : keluar")
    print("=" * 70)

    plt.show()   # jendela interaktif - blocking sampai ditutup

    if measurement_log:
        print("\nRingkasan seluruh pengukuran pada sesi ini:")
        for label, _, reading in measurement_log:
            print(f"  {label}: NDVI={reading.ndvi:.2f}, "
                  f"Tkanopi={reading.canopy_temp_C:.1f}C, "
                  f"Ttanah={reading.soil_temp_C:.1f}C, "
                  f"Tinggi={reading.canopy_height_m:.2f}m, "
                  f"Cover={reading.canopy_cover_pct:.0f}%")


if __name__ == "__main__":
    main()
