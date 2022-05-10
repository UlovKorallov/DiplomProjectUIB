# Импорт нужных библиотек
import os
import pickle
import sys
import face_recognition
import cv2

# Создание функции train_model_by_img с параметром name. Функция отвечает за обучение модели на скриншотах.
def train_model_by_img(name):

    if not os.path.exists(f"datasets\{name}_dataset"):
        print(f"[ERROR] there is no directory 'datasets\{name}_dataset'")
        sys.exit()

    known_encodings = []
    images = os.listdir(f"datasets\{name}_dataset")

    print(images)

    for(i, image) in enumerate(images):
        print(f"[+] processing img {i + 1}/{len(images)}")
        # print(image)
        face_img = face_recognition.load_image_file(f"datasets\{name}_dataset/{image}")
        #print(face_img)
        if(len(face_recognition.face_encodings(face_img))>0):
            face_enc = face_recognition.face_encodings(face_img)[0]
        else:
            print("No face")
            continue

        #print(face_enc)

        if len(known_encodings) == 0:
            known_encodings.append(face_enc)
        else:
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])
                #print(len(known_encodings))

                if True in result:
                    known_encodings.append(face_enc)
                    print("+")
                    break
                else:
                    print("-")
                    break

    # print(known_encodings)
    # print(f"Length {len(known_encodings)}")

    data = {
        "name": name,
        "encodings": known_encodings
    }

    if not os.path.exists("pickles"):
        os.mkdir("pickles")

    with open(f"pickles/{name}_encodings.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    return f"[INFO] File {name}_encodings.pickle successfully created"

# Функция создания скриншотов с потокового видео
def take_screenshot_from_video(name,cam):
    cap = cv2.VideoCapture(cam)
    count = 0

    if not os.path.exists("datasets"):
        os.mkdir(f"datasets")
    if not os.path.exists(f"datasets\{name}_dataset"):
        os.mkdir(f"datasets\{name}_dataset")
    count_frames = 0
    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(fps)

        if ret:
            cv2.imshow("frame", frame)
            k = cv2.waitKey(20)
            if count_frames % fps == 0:
                cv2.imwrite(f"datasets\{name}_dataset/{count}_{name}.jpg", frame)
                print(f"Take a screenshot {count}")
                count += 1

            if k == ord(" "):
                cv2.imwrite(f"datasets\{name}_dataset/{count}_{name}_extra.jpg", frame)
                print(f"Take an extra screenshot {count}")
                count += 1
            elif k == ord("q"):
                print("Q pressed, closing the app")
                break

        else:
            print("[Error] Can't get the frame...")
            break
        count_frames += 1

    cap.release()
    cv2.destroyAllWindows()
    train_model_by_img(name)

# Функция main, через которую идет запуск тренировочной модели
def main():
    take_screenshot_from_video("KIM",0)
    #train_model_by_img("Kamilla",0)


if __name__ == '__main__':
    main()