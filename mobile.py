from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivymd.uix.boxlayout import BoxLayout
from kivy.uix.label import Label  # Importez Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np

class MyKivyApp(App):
    net = None
    classes = []
    capture = None
    image = None

    def build(self):
        # Réduisez la résolution de la vidéo
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, 640)  # Largeur
        self.capture.set(4, 480)  # Hauteur
        
        if not MyKivyApp.net:
            MyKivyApp.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        
        with open("coco.names", "r") as f:
            MyKivyApp.classes = f.read().strip().split("\n")
        
        self.image = Image()
        
        self.button = Button(text="Détecter")
        self.button.bind(on_press=self.detect_objects)  # Lier la détection au clic
        
        # Créez un titre en haut
        title = Label(text="Détection d'objets avec YOLO", size_hint=(1, 0.1), font_size='20sp')
        
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(title)
        layout.add_widget(self.image)
        layout.add_widget(self.button)
        
        return layout

    def detect_objects(self, instance):
        ret, frame = self.capture.read()
        if not ret:
            return

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        MyKivyApp.net.setInput(blob)
        outs = MyKivyApp.net.forward(MyKivyApp.net.getUnconnectedOutLayersNames())
        
        # Initialiser des listes pour les boîtes englobantes, les confiances et les classes
        boxes = []
        confidences = []
        class_ids = []

        # Analyser les détections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(MyKivyApp.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Couleur verte

                # Dessiner le rectangle autour de l'objet détecté
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Afficher le label et la confiance
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        texture = self.convert_frame_to_texture(frame)
        self.image.texture = texture

    def convert_frame_to_texture(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flipud(frame)
        buffer = frame.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        return texture

    def on_stop(self):
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    MyKivyApp().run()
