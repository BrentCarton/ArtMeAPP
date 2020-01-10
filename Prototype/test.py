import cv2
import sys
import dlib
import numpy as np
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(1)
i = int(0)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def faceSwap(label):
    print(label)
    frame = cv2.imread(label)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame_gray)
    painting = cv2.imread('painting2.jpg')
    painting_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    height, width, channels = painting.shape
    painting_new_face = np.zeros((height, width, channels), np.uint8)

    # Video
    faces = detector(frame_gray)
    for face in faces:
        landmarks = predictor(frame_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))



        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(frame, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(frame, frame, mask=mask)

        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])


            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

    # Painting
    faces2 = detector(painting_gray)
    for face in faces2:
        landmarks = predictor(painting_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))


        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

    lines_space_mask = np.zeros_like(frame_gray)
    lines_space_new_face = np.zeros_like(painting)

    # Triangulation of both inputs
    for triangle_index in indexes_triangles:
        # Triangulation of the video
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = frame[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)


        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(frame, frame, mask=lines_space_mask)

        # Triangulation of painting
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        painting_new_face_rect_area = painting_new_face[y: y + h, x: x + w]
        painting_new_face_rect_area_gray = cv2.cvtColor(painting_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(painting_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        painting_new_face_rect_area = cv2.add(painting_new_face_rect_area, warped_triangle)
        painting_new_face[y: y + h, x: x + w] = painting_new_face_rect_area

    # Face swapped (putting 1st face into 2nd face)
    painting_face_mask = np.zeros_like(painting_gray)
    painting_head_mask = cv2.fillConvexPoly(painting_face_mask, convexhull2, 255)
    painting_face_mask = cv2.bitwise_not(painting_head_mask)

    painting_head_noface = cv2.bitwise_and(painting, painting, mask=painting_face_mask)
    result = cv2.add(painting_head_noface, painting_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, painting, painting_head_mask, center_face2, cv2.NORMAL_CLONE)

    cv2.imshow("Faceswap", seamlessclone)

def faceDetection(i):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x,y,h,w) in faces:
        cv2.rectangle(frame, (x, y), (x+w+30, y+h+30), (238,130,238), 2) 
    if len(faces) > 0:
        print(i)
        if i%10 == 0:
            roi_color = frame[y:y + h, x:x + w] 
            print("[INFO] Object found. Saving locally.") 
            label = str('faces.jpg')
            cv2.imwrite(label, roi_color)
            try:
                faceSwap(label)
            except:
                print("An exception occurred")
    cv2.imshow('Video', frame)

while True:
    faceDetection(i)
    i = i+1
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()