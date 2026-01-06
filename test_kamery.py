import cv2


def open_two_cameras(cam1=0, cam2=1):
    # Na Windows CAP_DSHOW jest najbardziej stabilny
    cap1 = cv2.VideoCapture(cam1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(cam2, cv2.CAP_DSHOW)

    if not cap1.isOpened():
        raise RuntimeError(f"Nie mogę otworzyć kamery {cam1}")
    if not cap2.isOpened():
        raise RuntimeError(f"Nie mogę otworzyć kamery {cam2}")

    # Opcjonalnie ustaw rozdzielczość
    for cap in (cap1, cap2):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Podgląd z dwóch kamer uruchomiony")
    print("q lub ESC - wyjście")

    while True:
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()

        if not ok1 or not ok2:
            print("Błąd odczytu z jednej z kamer")
            break

        cv2.imshow(f"Kamera {cam1}", frame1)
        cv2.imshow(f"Kamera {cam2}", frame2)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q lub ESC
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ZMIEŃ indeksy jeśli trzeba (0,1,2...)
    open_two_cameras(cam1=0, cam2=1)
