import cv2
import numpy as np
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for layer in model.layers:
    layer.trainable = False


x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)


model = Model(inputs=model.input, outputs=predictions)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


cap = cv2.VideoCapture(0)


while True:
    # get a frame
    ret, frame = cap.read()


    cv2.imshow("Frame", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame = cv2.resize(frame, (224, 224))


    frame = np.array(frame)


    frame = frame / 255.0


    prediction = model.predict(frame[np.newaxis, ...])


    if prediction > 0.5:
        print("Theft detected")
    else:
        print("No theft detected")
cap.release()
cv2.destroyAllWindows()
