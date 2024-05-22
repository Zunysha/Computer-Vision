import cv2
import numpy as np
from time import sleep
from tkinter import Tk, Label, Button, OptionMenu, StringVar, filedialog
from PIL import Image, ImageTk

largura_min = 80  # Largura minima do retangulo
altura_min = 80  # Altura minima do retangulo
offset = 6  # Erro permitido entre pixel
pos_linha = 550  # Posição da linha de contagem
delay = 60  # FPS do vídeo
detec = []
carros = 0

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

def start_vehicle_counting():
    cap = cv2.VideoCapture('video.mp4')
    subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

    global carros
    carros = 0

    while True:
        ret, frame1 = cap.read()
        tempo = float(1 / delay)
        sleep(tempo)
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = subtracao.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3)
        for (i, c) in enumerate(contorno):
            (x, y, w, h) = cv2.boundingRect(c)
            validar_contorno = (w >= largura_min) and (h >= altura_min)
            if not validar_contorno:
                continue

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            centro = pega_centro(x, y, w, h)
            detec.append(centro)
            cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

            for (x, y) in detec:
                if y < (pos_linha + offset) and y > (pos_linha - offset):
                    carros += 1
                    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
                    detec.remove((x, y))
                    print("car is detected : " + str(carros))

        cv2.putText(frame1, "VEHICLE COUNT : " + str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.imshow("Video Original", frame1)
        cv2.imshow("Detectar", dilatada)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

class ComputerVisionApp:
    def __init__(self, root):
        self.root = root
        root.title("Computer Vision GUI")
        root.geometry("800x600")
        # set the background color of GUI window
        root.config(background='#e6e6e6')

        # Adding the heading label to the root window
        self.heading = Label(root, text="Computer Vision GUI", font=("Arial Bold", 30, "bold"), fg="#333333", bg='#e6e6e6')
        self.heading.pack(pady=10)

        self.image_path = None

        # GUI Elements
        self.label = Label(root, text="Select an Image:", font=("Arial Bold", 20, "bold"), fg="#333333", bg='#e6e6e6')
        self.label.pack(pady=10)

        # Open button
        self.button_open = Button(root, text="Open Image", command=self.open_file, font=("Arial Bold", 20, "bold"),
                                  fg="#ffffff", bg="#4CAF50", activebackground="#45a049", pady=10)
        self.button_open.pack(pady=10)

        self.image_label = Label(root, bg='#e6e6e6')
        self.image_label.pack(pady=10)

        self.techniques = ["Original", "Gaussian Noise", "Averaging Filter", "Median Filter", "Gaussian Filter",
                           "Canny Edge Detection", "Sobel Edge Detection", "Prewitt Edge Detection",
                           "Roberts Edge Detection", "Laplacian Edge Detection", "Hough Line Transform",
                           "Harris Corner Detection"]

        self.selected_technique = StringVar(root)
        self.selected_technique.set(self.techniques[0])

        self.technique_menu = OptionMenu(root, self.selected_technique, *self.techniques)
        self.technique_menu.config(font=("Arial Bold", 15), fg="#333333", bg="#ffffff", activebackground="#dddddd")
        self.technique_menu.pack(pady=10)

        self.apply_button = Button(root, text="Apply Technique", command=self.apply_technique,
                                   font=("Arial Bold", 18), fg="#ffffff", bg="#4CAF50", activebackground="#45a049", pady=10)
        self.apply_button.pack(pady=10)

        self.reset_button = Button(root, text="Reset Result", command=self.reset_result,
                                   font=("Arial Bold", 18), fg="#ffffff", bg="#ff3333", activebackground="#d33d3d", pady=10)
        self.reset_button.pack(pady=10)

                # Heading for Vehicle Counting System
        self.vehicle_counting_heading = Label(root, text="Vehicle Counting System", font=("Arial Bold", 20, "bold"), fg="#333333", bg='#e6e6e6')
        self.vehicle_counting_heading.pack(pady=10)

        self.button_vehicle_counting = Button(root, text="Vehicle Counting System", command=start_vehicle_counting,
                                              font=("Arial Bold", 18), fg="#ffffff", bg="#007bff", activebackground="#0056b3", pady=10)
        self.button_vehicle_counting.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.load_and_display_image()

    def load_and_display_image(self):
        image = Image.open(self.image_path)
        image.thumbnail((400, 400))  # Resize the image for display
        photo = ImageTk.PhotoImage(image)

        # Update the label to display the loaded image
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def apply_technique(self):
        if self.image_path:
            technique = self.selected_technique.get()

            # Read the image
            img = cv2.imread(self.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply the selected technique
            if technique == "Gaussian Noise":
                # Add Gaussian Noise
                gauss = np.random.normal(0, 1, img_rgb.size)
                gauss = gauss.reshape(img_rgb.shape).astype('uint8')
                img_result = cv2.add(img_rgb, gauss)
            elif technique == "Averaging Filter":
                # Apply Averaging Filter
                img_result = cv2.blur(img_rgb, (5, 5))
            elif technique == "Median Filter":
                # Apply Median Filter
                img_result = cv2.medianBlur(img_rgb, 5)
            elif technique == "Gaussian Filter":
                # Apply Gaussian Filter
                img_result = cv2.GaussianBlur(img_rgb, (5, 5), 0)
            elif technique == "Canny Edge Detection":
                # Apply Canny Edge Detection
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                img_result = cv2.Canny(gray, 100, 200)
            elif technique == "Sobel Edge Detection":
                # Apply Sobel Edge Detection
                img_result = cv2.Sobel(img_rgb, cv2.CV_8U, 1, 0, ksize=5) + cv2.Sobel(img_rgb, cv2.CV_8U, 0, 1, ksize=5)
            elif technique == "Prewitt Edge Detection":
                # Apply Prewitt Edge Detection
                kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                img_result = cv2.filter2D(img_rgb, -1, kernelx) + cv2.filter2D(img_rgb, -1, kernely)
            elif technique == "Roberts Edge Detection":
                # Apply Roberts Edge Detection
                kernelx = np.array([[0, 1], [-1, 0]])
                kernely = np.array([[1, 0], [0, -1]])
                img_result = cv2.filter2D(img_rgb, -1, kernelx) + cv2.filter2D(img_rgb, -1, kernely)
            elif technique == "Laplacian Edge Detection":
                # Apply Laplacian Edge Detection
                img_gaussian = cv2.GaussianBlur(img_rgb, (3, 3), 0)
                img_result = cv2.Laplacian(img_gaussian, cv2.CV_8U)
            elif technique == "Hough Line Transform":
                # Apply Hough Line Transform
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 200)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
                img_result = img_rgb.copy()
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif technique == "Harris Corner Detection":
                # Apply Harris Corner Detection
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray, 2, 3, 0.04)
                dst = cv2.dilate(dst, None)
                img_result = img_rgb.copy()
                img_result[dst > 0.01 * dst.max()] = [0, 0, 255]
            else:
                # Original Image
                img_result = img_rgb

            # Display the result
            self.display_result(img_result)

    def display_result(self, img_result):
        # Display the result image
        result_image = Image.fromarray(img_result)
        result_image.thumbnail((400, 400))  # Resize the image for display
        result_photo = ImageTk.PhotoImage(result_image)

        # Update the label to display the result image
        self.image_label.config(image=result_photo)
        self.image_label.image = result_photo

    def reset_result(self):
        if self.image_path:
            # Reload and display the original image
            self.load_and_display_image()

# Create the Tkinter GUI
root = Tk()
app = ComputerVisionApp(root)
root.mainloop()
