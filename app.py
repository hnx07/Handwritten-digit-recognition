import cv2
import numpy as np
import model

isDrawing = False
img = np.zeros((330, 280), dtype=np.uint8)
W1, b1, W2, b2, W3, b3 = None, None, None, None, None, None

def loadModelWeights(weightPath):
    data = np.load(weightPath)
    return data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3']

def setupUi():
    global img
    img[0:280, 0:280] = 0
    cv2.line(img, (0, 280), (280, 280), (255), 2)
    cv2.line(img, (140, 280), (140, 330), (255), 2)
    cv2.putText(img, "Erase", (45, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255), 2)
    cv2.putText(img, "Predict", (155, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255), 2)

def predictDigit():
    global img, W1, b1, W2, b2, W3, b3
    roi = img[0:280, 0:280]
    imgResized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    X = imgResized.reshape(784, 1) / 255.0
    _, _, _, _, _, A3 = model.propagateForward(X, W1, b1, W2, b2, W3, b3)
    prediction = np.argmax(A3, axis=0)[0]
    print("=========================")
    print(f"Prediction Answer: {prediction}")
    print("=========================\n")

def handleMouseEvents(event, x, y, flags, param):
    global isDrawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > 280: 
            if x < 140:
                setupUi() 
                print("Đã xóa bảng vẽ.")
            else:
                predictDigit() 
        else: 
            isDrawing = True
            cv2.circle(img, (x, y), 8, (255), -1) 
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDrawing and y <= 280:
            cv2.circle(img, (x, y), 12, (255), -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        isDrawing = False

def executeApp():
    global W1, b1, W2, b2, W3, b3
    W1, b1, W2, b2, W3, b3 = loadModelWeights("weights/best_weight.npz")
    setupUi()
    cv2.namedWindow("Digit Recognition")
    cv2.setMouseCallback("Digit Recognition", handleMouseEvents)
    print("Press 'q' or 'ESC' to leave");
    while True:
        cv2.imshow("Digit Recognition", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): 
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    executeApp()