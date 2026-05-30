import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    gesture = "No Hand Detected"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            lm = hand_landmarks.landmark
            if (
                lm[8].y < lm[6].y and
                lm[12].y < lm[10].y and
                lm[16].y < lm[14].y and
                lm[20].y < lm[18].y
            ):
                gesture = "OPEN PALM"
            elif (
                lm[8].y > lm[6].y and
                lm[12].y > lm[10].y and
                lm[16].y > lm[14].y and
                lm[20].y > lm[18].y
            ):
                gesture = "FIST"
            elif (
                lm[4].y < lm[3].y and
                lm[8].y > lm[6].y
            ):
                gesture = "THUMBS UP"
    cv2.putText(
        img,
        gesture,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
