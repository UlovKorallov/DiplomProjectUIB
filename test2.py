# Импорт нужных библиотек
import time
import os
import cv2
import face_recognition
import pickle
import numpy as np
from threading import Thread

needpercent = 65.0


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
            print('[Exiting] No more frames to read')
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
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()
# Метод распознавания лица на фотографии
    def recognize(self,frame):
        self.t.locations = face_recognition.face_locations(frame, number_of_times_to_upsample=1, model="hog")
        self.t.encodings = face_recognition.face_encodings(frame, self.t.locations)
        sumrec = 0
        sumfaces = 0
        sumunrec = 0
        #print(self.t.encodings)
        for face_encoding, face_location in zip(self.t.encodings, self.t.locations):
            Accuracies = []
            Names = []
            sumfaces += 1
            #ifAnotherPerson = []
            face_loc = face_location
            '''if len(self.t.face_enc_recogned)>0:
                print(len(self.t.face_enc_names),"fff")
                for i in range(len(self.t.face_enc_recogned)):

                    result = face_recognition.compare_faces([self.t.face_enc_recogned[i]], face_encoding)
                    accur = face_recognition.face_distance([self.t.face_enc_recogned[i]], face_encoding)
                    match = None
                    if True in result:
                        accuracy = 1.0-float(min(accur))
                        accuracy *= 100.0
                        stringstring = toFixed(accuracy,2)+"%"
                        match = self.t.face_enc_names[i]
                        sumrec += 1
                    left_top = (face_location[3], face_location[0])
                    right_bottom = (face_location[1], face_location[2])
                    color = [0, 255, 0]
                    cv2.rectangle(frame, left_top, right_bottom, color, 4)
                    left_bottom = (face_location[3], face_location[2])
                    right_bottom = (face_location[1], face_location[2] + 20)
                    # cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)
                    if (match != None):
                        print(match,1)
                    cv2.putText(frame, match, (face_location[3] + 18, face_location[2] + 24), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 4)
                    if accuracy>40.0:
                        cv2.putText(frame, stringstring, (face_location[1]-80, face_location[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 4)
'''
            for i in range(len(self.t.images)):
                '''k=0
                for j in range(len(self.t.face_enc_recogned)):
                    result1 = face_recognition.compare_faces([self.t.face_enc_recogned[j]], face_encoding)
                    #if True in result1:
                     #   k=1'''
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

        '''if sumrec > 0 or sumfaces > 0:
            print(sumrec, sumfaces)'''

    # метод возврата последнего прочитанного кадра
    def read(self):
        return self.frame

    def getNamesOnVideo(self):
        print(f"На видео были обнаружены {len(self.t.peoplefind)} человек и распознаны {len(self.t.namesfind)} человек: ")
        for i in range(len(self.t.namesfind)):
            print(self.t.namesfind[i])

    # метод, вызываемый для прекращения чтения фреймов
    def stop(self):
        self.stopped = True

# инициализация и запуск многопоточного входного потока захвата веб-камеры
def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"
def facesss(videoid):
    webcam_stream = WebcamStream(stream_id=videoid)  # stream_id = 0 is for primary camera
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
            src = frame

            # процент изменения размера изображения
            scale_percent = 100

            # рассчитать 50 процентов исходных размеров
            width = int(src.shape[1] * scale_percent / 100)
            height = int(src.shape[0] * scale_percent / 100)

            dsize = (width, height)

            # Изменение размера изображения
            output = cv2.resize(src, dsize)
            webcam_stream.recognize(output)
            # добавление задержки для имитации времени обработки кадра
            delay = 0.03
            # значение задержки в секундах. поэтому задержка = 1 эквивалентна 1 секунде
            time.sleep(delay)
            num_frames_processed += 1
        cv2.imshow('frame', output)
        cv2.imwrite(f"udalit_extra.jpg", output)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    webcam_stream.getNamesOnVideo()
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




def main():
    #facesss("video.mp4")
    facesss(0)

    pass
if __name__ == '__main__':
    main()