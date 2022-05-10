# Импорт нужных библиотек
import time
import os
from random import randint

import cv2
import face_recognition
import pickle
from threading import Thread
from PIL import Image, ImageDraw
import numpy as np
from flask import Flask, jsonify, request, redirect, render_template, url_for


needpercent = 60.0

app = Flask(__name__)



# Создание класса WebcamStream для реализации многопоточной обработки
class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        # задаем id 0 по умолчанию для камеры

        # открытие потокового захвата видео
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # чтение одного кадра из потока vcap для инициализации
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to readd')
            exit(0)

        # Для self.stopped установлено значение False, когда кадры считываются из потока self.vcap.
        self.stopped = True

        # ссылка на поток для чтения следующего доступного кадра из входного потока
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    # потоки daemon продолжают работать в фоновом режиме во время выполнения программы

    # метод запуска потока для захвата следующего доступного кадра во входном потоке
    def start(self):
        self.stopped = False

        self.t.face_enc_recogned = []
        self.t.face_enc_names = []
        self.t.maxaccur = 0.0
        self.t.namesfind = []
        self.t.peoplefind = []
        self.t.start()

    # метод загрузки данных с pickles файлов
    def dataload(self):
        self.t.data = []
        self.t.images = os.listdir(f"pickles")
        # print(images)
        for (i, image) in enumerate(self.t.images):
            name = image.strip().split('_')[0]
            # print(name)
            self.t.data.append(pickle.loads(open(f"pickles/{name}_encodings.pickle", "rb").read()))

    # метод чтения следующего кадра
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to readf')
                self.stopped = True
                break
        self.vcap.release()



    def show_prediction_labels_on_image(self, pil_image, predictions, accur):

        #pil_image = Image.fromarray(frameName)
        draw = ImageDraw.Draw(pil_image)
        iterat = 0
        for name, (top, right, bottom, left) in predictions:
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            if(name == 'unknown'):
                isUnknown = True
            else:
                isUnknown = False

            name = name.encode("UTF-8")
            accur[iterat] = 100*(1.0-accur[iterat])
            stringaccuracy = toFixed(accur[iterat], 2) + "%"
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 20), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 17), name, fill=(255, 255, 255, 255))
            if not isUnknown:
                draw.text((left + 6, bottom - text_height - 3), stringaccuracy, fill=(255, 255, 255, 255))
            iterat+=1
        del draw

        #pil_image.show()

    def predict(self, frameName, knn_clf=None, model_path=None, distance_threshold=0.6):

        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either though knn_clf or model_path")

        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        X_face_locations = face_recognition.face_locations(frameName)

        if len(X_face_locations) == 0:
            return []

        faces_encodings = face_recognition.face_encodings(frameName, known_face_locations=X_face_locations)

        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


    # Метод распознавания лица на фотографии
    def recognize(self, frame):
        # self.t.locations = face_recognition.face_locations(frame, number_of_times_to_upsample=1, model="hog")
        # self.t.encodings = face_recognition.face_encodings(frame, self.t.locations)

        distance_threshold = 1-needpercent/100
        with open('face_knn.clf', 'rb') as f:
            clf = pickle.load(f)
        X_face_locations = face_recognition.face_locations(frame)
        if len(X_face_locations) != 0:
            faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)



            for face_encoding, face_location in zip(faces_encodings, X_face_locations):
                if (len(self.t.peoplefind) == 0):
                    self.t.peoplefind.append(face_encoding)
                    print("Added 0")

                for j in range(len(self.t.peoplefind)):
                    result1 = face_recognition.compare_faces(face_encoding, np.asarray(self.t.peoplefind))
                    accur1 = face_recognition.face_distance(face_encoding, np.asarray(self.t.peoplefind))
                    accuracy = 1.0 - float(min(accur1))
                    accuracy *= 100.0
                    # print(np.asarray(self.t.peoplefind))
                    # print(np.asarray(result1))
                    print(result1, accur1)
                    # if(len(result1)>0):
                    if accuracy < 40:
                        self.t.peoplefind.append(face_encoding)
                        print("added 1")



            # Use the KNN model to find the best matches for the test face
            closest_distances = clf.kneighbors(faces_encodings, n_neighbors=1)
            #print(closest_distances)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
            accuracy = [closest_distances[0][i][0] for i in range(len(X_face_locations))]
            predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(clf.predict(faces_encodings), X_face_locations, are_matches)]
            # Predict classes and remove classifications that aren't within the threshold
            clf = [pred if rec else "unknown" for pred, rec in
                   zip(clf.predict(faces_encodings), are_matches)]
            n_unknowns = 0
            for i in clf:
                if i == 'unknown':
                    n_unknowns += 1
            clf = [i for i in clf if i != 'unknown']
            clf_str = ", ".join(clf)
            for i in clf:
                if i not in self.t.namesfind:
                    self.t.namesfind.append(i)
            if n_unknowns != 0:
                if clf_str != "":
                    if n_unknowns == 1:
                        clf_str += f' and {n_unknowns} unknown face'
                    else:
                        clf_str += f' and {n_unknowns} unknown faces'
                else:
                    if n_unknowns == 1:
                        clf_str += f'{n_unknowns} unknown face'
                    else:
                        clf_str += f'{n_unknowns} unknown faces'
            #print(clf_str)
            pilImg = Image.fromarray(frame)
            self.show_prediction_labels_on_image(pilImg, predictions, accuracy)
            frame = np.asarray(pilImg)
        return frame
        #cv2.imshow('frame', frame)

        # sumrec = 0
        # sumfaces = 0
        # sumunrec = 0
        # #print(self.t.encodings)
        '''for face_encoding, face_location in zip(self.t.encodings, self.t.locations):
            Accuracies = []
            Names = []
            sumfaces += 1
            #ifAnotherPerson = []
            face_loc = face_location

            for i in range(len(self.t.images)):
                result = face_recognition.compare_faces(self.t.data[i]["encodings"], face_encoding)
                #print(face_encoding,i)
                #print(type(self.t.data[i]["encodings"]), type(face_encoding))
                #print(result, i)
                accur = face_recognition.face_distance(self.t.data[i]["encodings"], face_encoding)
                match = None
                #print(match)
                #print(result," result")
                #print(accuracy*100/100,self.t.data[i]["name"])
                left_top = (face_location[3], face_location[0])
                right_bottom = (face_location[1], face_location[2])
                color = [0, 255, 0]
                cv2.rectangle(frame, left_top, right_bottom, color, 4)

                if True in result: #and k!=1:
                    accuracy = 1.0-float(min(accur))
                    accuracy *= 100.0
                    match = self.t.data[i]["name"]
                        #sumrec += 1
                    self.t.face_enc_recogned.append(face_encoding)
                    self.t.face_enc_names.append(match)
                    #print(len(self.t.face_enc_recogned[0]))
                    #print(accuracy,match)

                    Accuracies.append(accuracy)
                    Names.append(match)

                if(len(self.t.peoplefind) == 0):
                    self.t.peoplefind.append(face_encoding)
                    print("Added 0")

                for j in range(len(self.t.peoplefind)):
                    result1 = face_recognition.compare_faces(face_encoding, np.asarray(self.t.peoplefind))
                    accur1 = face_recognition.face_distance(face_encoding, np.asarray(self.t.peoplefind))
                    accuracy = 1.0-float(min(accur1))
                    accuracy *= 100.0
                    #print(np.asarray(self.t.peoplefind))
                    #print(np.asarray(result1))
                    print(result1,accur1)
                    #if(len(result1)>0):
                    if accuracy<40:
                        self.t.peoplefind.append(face_encoding)
                        print("added 1")
            MaxAccuracy = 0.0
            MaxID = -1
            for i in range(len(Accuracies)):
                if MaxAccuracy<Accuracies[i]:
                    MaxAccuracy = Accuracies[i]
                    MaxID = i
            stringaccuracy = toFixed(MaxAccuracy, 2) + "%"
            if MaxAccuracy>needpercent:
                sumunrec += 1
                #print(stringaccuracy,MaxID,Names,Accuracies)
                cv2.putText(frame, Names[MaxID], (face_location[3] + 18, face_location[2] + 24), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 4)
                cv2.putText(frame, stringaccuracy, (face_location[1]-80, face_location[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 4)
                if Names[MaxID] not in self.t.namesfind:
                    self.t.namesfind.append(Names[MaxID])
            else:
                #for face_encoding, face_location in zip(self.t.encodings, self.t.locations):
                if len(face_loc)==4:
                    cv2.putText(frame, "Another Person", (face_location[3] + 18, face_location[2] + 24), cv2.FONT_HERSHEY_SIMPLEX,
                            1,(255, 255, 255), 4)
            #print()
            #print()
        '''

    # метод возврата последнего прочитанного кадра
    def read(self):
        return self.frame

    def getNamesOnVideo(self):
        peoples = ""
        print(
            f"На видео были обнаружены {len(self.t.peoplefind)} человек и распознаны {len(self.t.namesfind)} человек: ")
        peoples += f"На видео были обнаружены {len(self.t.peoplefind)} студентов, из них были распознаны {len(self.t.namesfind)}"
        # peoples += f"На видео были обнаружены {len(self.t.namesfind)} студентов"
        for i in range(len(self.t.namesfind)):
            print(self.t.namesfind[i])
            peoples += ";"+self.t.namesfind[i]
        return peoples

    # метод, вызываемый для прекращения чтения фреймов
    def stop(self):
        self.stopped = True


# инициализация и запуск многопоточного входного потока захвата веб-камеры
def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"

@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template("index.html")
@app.route('/start', methods=['GET', 'POST'])
def back():
    if request.method == 'POST':
        return render_template("index.html")
    return render_template("index.html")


@app.route('/getusers',methods=['POST'])
def getusers():
    f = open('users.txt', 'r')
    text = f.readlines()
    f.close()
    print(text[0])
    return str(text[0])

@app.route('/sentusers',methods=['POST'])
def sentusers():
    value = request.form['text']
    f = open('users.txt', 'a')
    print(value)
    f.write(value)
    f.close()
    return "true"



@app.route('/test',methods=['POST'])
def testing():

    webcam_stream = WebcamStream(stream_id=0)  # stream_id = 0 is for primary camera
    webcam_stream.start()
    # обработка кадров во входном потоке
    num_frames_processed = 0
    start = time.time()
    webcam_stream.dataload()
    while True:
        if webcam_stream.stopped is True:
            break
        else:
            frame = webcam_stream.read()
            frame = webcam_stream.recognize(frame)
            # добавление задержки для имитации времени обработки кадра
            delay = 0.03
            # значение задержки в секундах. поэтому задержка = 1 эквивалентна 1 секунде
            time.sleep(delay)
            num_frames_processed += 1
        cv2.imshow('frame', frame)
        cv2.imwrite(f"static/last.png", frame)
        key = cv2.waitKey(1)
        print(time.time() - start)
        if key == ord('q') or time.time() - start >= 60:
            break
    end = time.time()
    webcam_stream.stop()
    # остановка трансляцию с веб-камеры

    # printing time elapsed and fps
    # вывод затраченного времени и fps
    elapsed = end - start
    fps = num_frames_processed / elapsed
    print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

    # закрытие всех окон
    cv2.destroyAllWindows()
    stringToApp = webcam_stream.getNamesOnVideo()
    return str(stringToApp)

@app.route('/back', methods=['GET', 'POST'])
def facesss(videoid = 0):
    webcam_stream = WebcamStream(stream_id=videoid)  # stream_id = 0 is for primary camera
    webcam_stream.start()
    # обработка кадров во входном потоке
    num_frames_processed = 0
    start = time.time()
    webcam_stream.dataload()
    str = "1"

    if request.method == 'POST':
        while True:
            if webcam_stream.stopped is True:
                break
            else:
                frame = webcam_stream.read()
                frame = webcam_stream.recognize(frame)
                # добавление задержки для имитации времени обработки кадра
                delay = 0.03
                # значение задержки в секундах. поэтому задержка = 1 эквивалентна 1 секунде
                time.sleep(delay)
                num_frames_processed += 1
            cv2.imshow('frame', frame)
            cv2.imwrite(f"static/last.png", frame)
            key = cv2.waitKey(1)
            print(time.time() - start)
            str = webcam_stream.getNamesOnVideo()
            if key == ord('q')  or time.time() - start >= 20:
                break
        end = time.time()
        webcam_stream.stop()
        # остановка трансляцию с веб-камеры

        # printing time elapsed and fps
        # вывод затраченного времени и fps
        elapsed = end - start
        fps = num_frames_processed / elapsed
        print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

        # закрытие всех окон
        cv2.destroyAllWindows()
        return render_template('ht.html',prediction = webcam_stream.getNamesOnVideo())
    return render_template('index.html')




def main():
    # facesss(0)
    # f = open('users.txt', 'a')
    # for i in range(10000):
    #     f.write(str(randint(0, 1000)) + " " + "f ")
    # f.close()
    pass


if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0',  port=5001, debug=False)