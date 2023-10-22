import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of images to collect for each class
dataset_size = 100

# Ask the user how many classes they want to collect
number_of_classes = int(input("Enter the number of new classes to collect: "))

# Find the starting class based on existing directories
existing_dirs = [int(dir_name) for dir_name in os.listdir(DATA_DIR) if dir_name.isdigit()]
if existing_dirs:
    start_class = max(existing_dirs) + 1
else:
    start_class = 0

cap = cv2.VideoCapture(1)
for j in range(start_class, start_class + number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
