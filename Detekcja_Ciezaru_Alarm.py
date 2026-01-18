import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ===============================
# STAŁA KONFIGURACJA (ODGÓRNIE)
# ===============================
MODEL_PATH = "D:/EE/ProjektCiezary/python/train19/weights/best.pt"

# Dostępne indeksy do wyboru (0..3) i DWIE kamery wybierane w GUI
AVAILABLE_CAMERA_INDEXES = [0, 1, 2, 3]

ZOOM_FACTOR = 2.0               # stały zoom
CONF_TH = 0.5                   # stały próg conf
UNIQUE_CLASSES = True           # 1x na klasę (oprócz 25kg)
MULTI_ALLOWED_CLS_ID = 7        # 25kg (cls_id=7) może występować wiele razy
MODE_VALUES = {"M": 20, "F": 15}

# ===============================
# BACKENDY (per indeks) - jak nie ma klucza, użyje domyślnej listy
# ===============================
PREFERRED_BACKENDS = {
    0: [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("AUTO", 0)],
    1: [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("AUTO", 0)],
    2: [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("AUTO", 0)],
    3: [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("AUTO", 0)],
}

# ===============================
# MAPA KLAS I WARTOŚCI
# ===============================
class_values = {
    0: 0.5, 1: 1.5, 2: 10, 3: 15, 4: 1,
    5: 2.5, 6: 20, 7: 25, 8: 2, 9: 5, 10: 2.5,
}
class_names = {
    0: "0.5kg", 1: "1.5kg", 2: "10kg", 3: "15kg", 4: "1kg",
    5: "2.5kg", 6: "20kg", 7: "25kg", 8: "2kg", 9: "5kg", 10: "zacisk",
}

# ===============================
# MODEL
# ===============================
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"🔍 Model działa na: {device.upper()}")

# ===============================
# ZMIENNE WSPÓLNE (inicjalizacja; realne indeksy wybierane w GUI)
# ===============================
global_lock = threading.Lock()
stop_event = threading.Event()

# Aktualnie używane indeksy (ustawiane w start_processing)
active_camera_indexes = []

# Dane per kamera (słowniki tworzone/odświeżane przy starcie)
global_weights = {}
latest_frames = {}

# ===============================
# OBRAZ: ZOOM / OKNO SUMY
# ===============================
def apply_digital_zoom(frame, zoom_factor=1.0):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def render_total_image(total_with_mode: float, mode: str, mismatch: bool, w=520, h=260):
    # jeśli mismatch -> czerwone tło, inaczej czarne
    bg = (0, 0, 255) if mismatch else (0, 0, 0)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = bg

    cv2.putText(canvas, f"TRYB: {mode}", (int(w * 0.30), int(h * 0.38)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)
    cv2.putText(canvas, f"{total_with_mode:.1f} KG", (int(w * 0.17), int(h * 0.78)),
                cv2.FONT_HERSHEY_SIMPLEX, 2.1, (0, 255, 255), 4)
    return canvas

def placeholder_frame(text: str, w=640, h=640):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
    return img

# ===============================
# KAMERA
# ===============================
def open_camera(index: int):
    backends = PREFERRED_BACKENDS.get(index, [("AUTO", 0), ("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)])
    for name, api in backends:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, name
        cap.release()
    return None, None

# ===============================
# WĄTEK KAMERY + YOLO
# ===============================
def camera_worker(camera_index=0, label="Kamera"):
    cap, backend_name = open_camera(camera_index)
    if cap is None:
        print(f"❌ Nie udało się uruchomić {label} (index={camera_index}) żadnym backendem.")
        with global_lock:
            latest_frames[camera_index] = placeholder_frame(f"{label}: BRAK KAMERY")
            global_weights[camera_index] = 0.0
        return

    print(f"📷 {label} uruchomiona. Index={camera_index}, Backend={backend_name}")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            with global_lock:
                latest_frames[camera_index] = placeholder_frame(f"{label}: BRAK KLATKI")
                global_weights[camera_index] = 0.0
            time.sleep(0.05)
            continue

        frame = apply_digital_zoom(frame, zoom_factor=ZOOM_FACTOR)

        # crop 640x640 ze środka
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        crop_size = 640
        half = crop_size // 2

        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w)
        y2 = min(cy + half, h)

        cropped = frame[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_LINEAR)

        results = model(cropped, verbose=False, conf=CONF_TH)

        detections = []  # (cls_id, conf, x1, y1, x2, y2)
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            cls_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            for i in range(len(boxes)):
                x1b, y1b, x2b, y2b = map(int, boxes[i])
                detections.append((int(cls_ids[i]), float(confs[i]), x1b, y1b, x2b, y2b))

        # 1x na klasę (oprócz 25kg)
        if UNIQUE_CLASSES:
            best_per_class = {}
            multi_dets = []
            for cls_id, conf, x1b, y1b, x2b, y2b in detections:
                if cls_id == MULTI_ALLOWED_CLS_ID:
                    multi_dets.append((cls_id, conf, x1b, y1b, x2b, y2b))
                else:
                    prev = best_per_class.get(cls_id)
                    if (prev is None) or (conf > prev[0]):
                        best_per_class[cls_id] = (conf, x1b, y1b, x2b, y2b)

            final_dets = [(cid, data[0], data[1], data[2], data[3], data[4]) for cid, data in best_per_class.items()]
            final_dets.extend(multi_dets)
        else:
            final_dets = detections

        final_dets.sort(key=lambda x: (x[0], -x[1]))

        # Liczenie sumy + rysowanie
        total_weight = 0.0
        all_class_names = []

        for cls_id, conf, x1b, y1b, x2b, y2b in final_dets:
            total_weight += class_values.get(cls_id, 0.0)
            all_class_names.append(class_names.get(cls_id, f"klasa_{cls_id}"))

            cv2.rectangle(cropped, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
            text_y = max(y1b - 10, 10)
            cv2.putText(
                cropped, f"{class_names.get(cls_id, cls_id)} {conf:.2f}", (x1b, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

        header_height = 60
        cv2.rectangle(cropped, (0, 0), (cropped.shape[1], header_height), (50, 50, 50), -1)
        header_text = " | ".join(all_class_names) if all_class_names else "Brak obiektów"
        cv2.putText(cropped, header_text, (10, int(header_height * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, f"SUMA: {total_weight:.1f} kg", (10, int(header_height * 0.92)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(cropped, (320, 320), 5, (0, 0, 255), -1)

        with global_lock:
            global_weights[camera_index] = total_weight
            latest_frames[camera_index] = cropped.copy()

        time.sleep(0.001)

    cap.release()
    print(f"🛑 {label} zakończona.")

# ===============================
# GUI: SUMA + START/STOP + TRYB + PODGLĄD + WYBÓR INDEKSÓW KAMER (2 szt.)
# ===============================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detekcja Ciezaru")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.mode = tk.StringVar(value="M")
        self.preview = tk.BooleanVar(value=False)

        # wybór dwóch indeksów kamer (0..3)
        self.cam_a = tk.IntVar(value=0)
        self.cam_b = tk.IntVar(value=1)

        self.running = False
        self.threads = []
        self.tk_images = {}
        self.preview_windows_open = False

        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(root, text="Sterowanie", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        # Start/Stop
        row0 = ttk.Frame(controls)
        row0.grid(row=0, column=0, sticky="w")
        self.btn_start = ttk.Button(row0, text="Start", command=self.start_processing)
        self.btn_stop = ttk.Button(row0, text="Stop", command=self.stop_processing, state="disabled")
        self.btn_start.grid(row=0, column=0, padx=(0, 6))
        self.btn_stop.grid(row=0, column=1)

        # Wybór kamer
        row_cam = ttk.Frame(controls)
        row_cam.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(row_cam, text="Kamera A:").grid(row=0, column=0, padx=(0, 6))
        self.combo_a = ttk.Combobox(row_cam, width=5, state="readonly",
                                    values=[str(i) for i in AVAILABLE_CAMERA_INDEXES])
        self.combo_a.set(str(self.cam_a.get()))
        self.combo_a.grid(row=0, column=1, padx=(0, 12))

        ttk.Label(row_cam, text="Kamera B:").grid(row=0, column=2, padx=(0, 6))
        self.combo_b = ttk.Combobox(row_cam, width=5, state="readonly",
                                    values=[str(i) for i in AVAILABLE_CAMERA_INDEXES])
        self.combo_b.set(str(self.cam_b.get()))
        self.combo_b.grid(row=0, column=3)

        ttk.Label(controls, text="(Jeśli program działa: Stop → zmień → Start)").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        # Podgląd
        ttk.Checkbutton(
            controls,
            text="Podgląd (pokaż okna kamer z detekcją)",
            variable=self.preview
        ).grid(row=3, column=0, sticky="w", pady=(10, 0))

        # Tryb
        ttk.Label(controls, text="Tryb:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        row1 = ttk.Frame(controls)
        row1.grid(row=5, column=0, sticky="w")
        ttk.Radiobutton(row1, text="M (+20)", value="M", variable=self.mode).grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(row1, text="F (+15)", value="F", variable=self.mode).grid(row=0, column=1)

        # Suma
        total_box = ttk.LabelFrame(root, text="Suma (2 kamery + tryb)", padding=10)
        total_box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        total_box.columnconfigure(0, weight=1)

        self.total_label = ttk.Label(total_box)
        self.total_label.grid(row=0, column=0, sticky="ew")

        self.after(60, self.update_ui)

        # auto start (jeśli chcesz ręcznie, usuń)
        self.start_processing()

    def _get_selected_indexes(self):
        try:
            a = int(self.combo_a.get())
            b = int(self.combo_b.get())
        except ValueError:
            a, b = 0, 1
        return a, b

    def start_processing(self):
        global active_camera_indexes, global_weights, latest_frames

        if self.running:
            return

        cam_a, cam_b = self._get_selected_indexes()
        if cam_a == cam_b:
            # jeśli wybrano to samo, automatycznie ustaw B na inny
            for idx in AVAILABLE_CAMERA_INDEXES:
                if idx != cam_a:
                    cam_b = idx
                    self.combo_b.set(str(cam_b))
                    break

        active_camera_indexes = [cam_a, cam_b]

        # wyczyść i przygotuj słowniki pod wybrane indeksy
        with global_lock:
            global_weights = {idx: 0.0 for idx in active_camera_indexes}
            latest_frames = {idx: None for idx in active_camera_indexes}

        stop_event.clear()
        self.running = True

        self.threads = []
        for idx in active_camera_indexes:
            t = threading.Thread(target=camera_worker, args=(idx, f"Kamera {idx}"), daemon=True)
            t.start()
            self.threads.append(t)

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        # podczas działania blokuj zmianę wyboru (żeby nie mieszać wątków)
        self.combo_a.configure(state="disabled")
        self.combo_b.configure(state="disabled")

    def stop_processing(self):
        global active_camera_indexes

        if not self.running:
            return

        stop_event.set()
        for t in self.threads:
            t.join(timeout=1.0)
        self.running = False

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

        # odblokuj wybór kamer
        self.combo_a.configure(state="readonly")
        self.combo_b.configure(state="readonly")

        with global_lock:
            for idx in list(global_weights.keys()):
                global_weights[idx] = 0.0
            for idx in list(latest_frames.keys()):
                latest_frames[idx] = None

        self._close_preview_windows()

    def _close_preview_windows(self):
        # zamknij okna dla aktywnych kamer + ewentualnie "starych"
        for idx in AVAILABLE_CAMERA_INDEXES:
            try:
                cv2.destroyWindow(f"Kamera {idx}")
            except cv2.error:
                pass
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass
        self.preview_windows_open = False

    def _update_preview_windows(self):
        if not self.preview.get():
            if self.preview_windows_open:
                self._close_preview_windows()
            return

        # ZAWSZE pokazuj 2 okna: wybrane A i B
        cam_a, cam_b = self._get_selected_indexes()
        show_idxs = [cam_a, cam_b]

        with global_lock:
            frames = {idx: (latest_frames.get(idx).copy() if latest_frames.get(idx) is not None else None)
                      for idx in show_idxs}

        for idx in show_idxs:
            frame = frames.get(idx)
            if frame is None:
                frame = placeholder_frame(f"Kamera {idx}: BRAK KLATKI")
            cv2.imshow(f"Kamera {idx}", frame)

        cv2.waitKey(1)
        self.preview_windows_open = True

    def update_ui(self):
        # pobierz wagi z dwóch aktywnych kamer i policz mismatch
        cam_a, cam_b = self._get_selected_indexes()
        with global_lock:
            wa = float(global_weights.get(cam_a, 0.0))
            wb = float(global_weights.get(cam_b, 0.0))
            total_all = wa + wb

        # "identyczna" — używamy małej tolerancji dla floatów
        mismatch = abs(wa - wb) > 1e-6

        mode = self.mode.get()
        total_with_mode = total_all + MODE_VALUES.get(mode, 0)

        img = render_total_image(total_with_mode, mode, mismatch=mismatch)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.tk_images["total"] = imgtk
        self.total_label.configure(image=imgtk)

        self._update_preview_windows()
        self.after(60, self.update_ui)

    def on_close(self):
        self.stop_processing()
        self._close_preview_windows()
        self.destroy()

# ===============================
# START
# ===============================
if __name__ == "__main__":
    # wymagane: pip install pillow
    app = App()
    app.mainloop()
