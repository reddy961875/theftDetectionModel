import cv2
import matplotlib
import numpy
import torch
import torchvision.transforms as transforms

# Load the trained model
model = torch.load("theft_detection_model.pt")
model.eval()
input_example = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(input_example)

print(output)

labels = {0: 'Not theft', 1: 'Theft'}

# Define the preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Open the video capture
cap = cv2.VideoCapture(0)

# Loop over the frames of the video
while True:
    # Grab the current frame
    ret, frame = cap.read()

    # Break the loop if the frame could not be grabbed
    if not ret:
        break

    frame = preprocess(frame)
    frame = frame.unsqueeze(0)

    # Make a prediction on the frame using the model
    with torch.no_grad():
        prediction = model(frame)
        prediction = torch.nn.functional.softmax(prediction, dim=1)

    # Get the class label with the highest prediction score
    label = labels[prediction.argmax().item()]

    # Draw the class label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Theft Detection", frame)

    # Check if the user pressed the "q" key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Release the video capture
    cap.release( )

    cv2.destroyAllWindows( )
