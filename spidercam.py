import tkinter as tk
import math


class SpiderCam:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulasi Spidercam")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        # ==================================================
        # UKURAN AREA SPIDERCAM
        # ==================================================

        self.area_width = 800
        self.area_height = 500

        # ==================================================
        # POSISI 4 MOTOR
        # ==================================================

        self.motors = {
            "M1": (80, 80),
            "M2": (720, 80),
            "M3": (80, 420),
            "M4": (720, 420)
        }

        # ==================================================
        # POSISI AWAL KAMERA
        # ==================================================

        self.camera_x = 400
        self.camera_y = 250
        self.camera_z = 100

        # Kecepatan gerakan kamera
        self.speed = 10

        # ==================================================
        # FRAME UTAMA
        # ==================================================

        main_frame = tk.Frame(root)
        main_frame.pack(pady=10)

        # ==================================================
        # CANVAS UNTUK GAMBAR SPIDERCAM
        # ==================================================

        self.canvas = tk.Canvas(
            main_frame,
            width=self.area_width,
            height=self.area_height,
            bg="white",
            highlightthickness=2,
            highlightbackground="black"
        )

        self.canvas.grid(row=0, column=0, padx=10)

        # ==================================================
        # PANEL INFORMASI
        # ==================================================

        info_frame = tk.Frame(main_frame, width=180)
        info_frame.grid(row=0, column=1, padx=10, sticky="n")

        title = tk.Label(
            info_frame,
            text="SIMULASI\nSPIDERCAM",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        # Posisi kamera
        self.position_label = tk.Label(
            info_frame,
            text="",
            font=("Arial", 11),
            justify="left"
        )
        self.position_label.pack(pady=10)

        # Informasi kabel
        self.cable_label = tk.Label(
            info_frame,
            text="",
            font=("Arial", 10),
            justify="left"
        )
        self.cable_label.pack(pady=10)

        # ==================================================
        # TOMBOL KONTROL
        # ==================================================

        control_title = tk.Label(
            info_frame,
            text="KONTROL",
            font=("Arial", 12, "bold")
        )
        control_title.pack(pady=(20, 5))

        tk.Label(
            info_frame,
            text="W = Maju\nS = Mundur\nA = Kiri\nD = Kanan\nQ = Naik\nE = Turun",
            font=("Arial", 10),
            justify="left"
        ).pack()

        # Tombol reset
        reset_button = tk.Button(
            info_frame,
            text="RESET",
            width=12,
            command=self.reset
        )
        reset_button.pack(pady=20)

        # ==================================================
        # EVENT KEYBOARD
        # ==================================================

        root.bind("<KeyPress>", self.key_pressed)

        # Gambar awal
        self.draw()

    # ======================================================
    # MENGHITUNG JARAK KABEL
    # ======================================================

    def calculate_distance(self, motor_x, motor_y):

        dx = self.camera_x - motor_x
        dy = self.camera_y - motor_y

        # Karena ada koordinat Z,
        # digunakan rumus jarak 3 dimensi
        distance = math.sqrt(
            dx ** 2 +
            dy ** 2 +
            self.camera_z ** 2
        )

        return distance

    # ======================================================
    # MENGGAMBAR SPIDERCAM
    # ======================================================

    def draw(self):

        self.canvas.delete("all")

        # --------------------------------------------------
        # GAMBAR AREA
        # --------------------------------------------------

        self.canvas.create_rectangle(
            20, 20,
            self.area_width - 20,
            self.area_height - 20,
            outline="gray",
            width=2
        )

        # --------------------------------------------------
        # GAMBAR 4 MOTOR
        # --------------------------------------------------

        for name, (x, y) in self.motors.items():

            # Lingkaran motor
            self.canvas.create_oval(
                x - 20,
                y - 20,
                x + 20,
                y + 20,
                fill="gray",
                outline="black",
                width=2
            )

            # Label motor
            self.canvas.create_text(
                x,
                y - 35,
                text=name,
                font=("Arial", 11, "bold")
            )

        # --------------------------------------------------
        # GAMBAR KABEL
        # --------------------------------------------------

        for name, (x, y) in self.motors.items():

            self.canvas.create_line(
                x,
                y,
                self.camera_x,
                self.camera_y,
                fill="black",
                width=2
            )

        # --------------------------------------------------
        # GAMBAR KAMERA
        # --------------------------------------------------

        camera_size = 25

        self.canvas.create_oval(
            self.camera_x - camera_size,
            self.camera_y - camera_size,
            self.camera_x + camera_size,
            self.camera_y + camera_size,
            fill="black"
        )

        # Lensa kamera
        self.canvas.create_oval(
            self.camera_x - 10,
            self.camera_y - 10,
            self.camera_x + 10,
            self.camera_y + 10,
            fill="white"
        )

        self.canvas.create_text(
            self.camera_x,
            self.camera_y + 40,
            text="KAMERA",
            font=("Arial", 11, "bold")
        )

        # --------------------------------------------------
        # UPDATE INFORMASI
        # --------------------------------------------------

        self.update_information()

    # ======================================================
    # UPDATE INFORMASI
    # ======================================================

    def update_information(self):

        self.position_label.config(
            text=(
                "POSISI KAMERA\n\n"
                f"X : {self.camera_x:.1f}\n"
                f"Y : {self.camera_y:.1f}\n"
                f"Z : {self.camera_z:.1f}"
            )
        )

        cable_text = "PANJANG KABEL\n\n"

        for name, (x, y) in self.motors.items():

            distance = self.calculate_distance(x, y)

            cable_text += f"{name} : {distance:.1f}\n"

        self.cable_label.config(text=cable_text)

    # ======================================================
    # KONTROL KEYBOARD
    # ======================================================

    def key_pressed(self, event):

        key = event.keysym.lower()

        # -----------------------------------------------
        # GERAK KIRI
        # -----------------------------------------------

        if key == "a":
            self.camera_x -= self.speed

        # -----------------------------------------------
        # GERAK KANAN
        # -----------------------------------------------

        elif key == "d":
            self.camera_x += self.speed

        # -----------------------------------------------
        # GERAK MAJU
        # -----------------------------------------------

        elif key == "w":
            self.camera_y -= self.speed

        # -----------------------------------------------
        # GERAK MUNDUR
        # -----------------------------------------------

        elif key == "s":
            self.camera_y += self.speed

        # -----------------------------------------------
        # NAIK
        # -----------------------------------------------

        elif key == "q":
            self.camera_z += self.speed

        # -----------------------------------------------
        # TURUN
        # -----------------------------------------------

        elif key == "e":
            self.camera_z -= self.speed

        # -----------------------------------------------
        # BATAS Z
        # -----------------------------------------------

        if self.camera_z < 10:
            self.camera_z = 10

        if self.camera_z > 300:
            self.camera_z = 300

        # -----------------------------------------------
        # BATAS X
        # -----------------------------------------------

        if self.camera_x < 50:
            self.camera_x = 50

        if self.camera_x > self.area_width - 50:
            self.camera_x = self.area_width - 50

        # -----------------------------------------------
        # BATAS Y
        # -----------------------------------------------

        if self.camera_y < 50:
            self.camera_y = 50

        if self.camera_y > self.area_height - 50:
            self.camera_y = self.area_height - 50

        # Gambar ulang
        self.draw()

    # ======================================================
    # RESET POSISI
    # ======================================================

    def reset(self):

        self.camera_x = 400
        self.camera_y = 250
        self.camera_z = 100

        self.draw()


# ==========================================================
# PROGRAM UTAMA
# ==========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = SpiderCam(root)

    root.mainloop()