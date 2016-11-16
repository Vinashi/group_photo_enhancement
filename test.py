import sys
import numpy as np
import os
import dlib
import glob
import cv2
from matplotlib import pyplot as plt
from skimage import io

predictor_path = 'shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat'
faces_folder_path = 'faces'

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()
face_matrix = []
filename = 0
faceNumber = 1
faceScore = 2
faceTurn = 3
faceSmile = 4
eyeScore = 5
faceRect = 6
facePoints = 7
image_scores = []

def eye_score((x1,y1), (x2,y2), (x3,y3), (x4,y4)):
    a = np.array((x1,y1))
    b = np.array((x2,y2))
    c = np.array((x3,y3))
    d = np.array((x4,y4))
    p = (np.linalg.norm(c-d))/(2*np.linalg.norm(a-b))
    if p < 0.07:
        return [p, "closed"]
    else:
        return [p, "open"]

def smile_score((x1,y1), (x2,y2), (x3,y3), (x4,y4)):
    a = np.array((x1,y1))
    b = np.array((x2,y2))
    c = np.array((x3,y3))
    d = np.array((x4,y4))
    return (np.linalg.norm(c-d))/(np.linalg.norm(a-b))
    

def turn_score((x1,y1), (x2,y2), (x3,y3)):
    a = np.array((x1,y1))
    b = np.array((x2,y2))
    c = np.array((x3,y3))
    p = (np.linalg.norm(a-b))/(np.linalg.norm(a-b)+np.linalg.norm(b-c))
    if p < 0.2 or p > 0.8:
        return [p, "turned"]
    else:
        return [p, "straight"]

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    

    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
    
    count= 0    
    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            count = count + 1 
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri
        

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    

#if __name__ == '__main__' :
def swap(ref, target, ref_pts, target_pts) :   
    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3 :
        print >>sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
        sys.exit(1)

    # Read images
    filename1 = ref
    filename2 = target
    
    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2);
    img1Warped = np.copy(img2);    
    
    # Read array of corresponding points
    points1 = ref_pts
    points2 = target_pts    
    
    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
          
    for i in xrange(0, len(hullIndex)):
        hull1.append(points1[hullIndex[i]])
        hull2.append(points2[hullIndex[i]])
    
    
    # Find delanauy traingulation for convex hull points
    sizeImg2 = img2.shape    
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
     
    dt = calculateDelaunayTriangles(rect, hull2)
    
    if len(dt) == 0:
        quit()
    
    # Apply affine transformation to Delaunay triangles
    for i in xrange(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in xrange(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])
        
        warpTriangle(img1, img1Warped, t1, t2)
    
            
    # Calculate Mask
    hull8U = []
    for i in xrange(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    
    mask = np.zeros(img2.shape, dtype = img2.dtype)  
    
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    
    r = cv2.boundingRect(np.float32([hull2]))    
    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
        
    
    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    
    #plt.imshow(output)
    #cv2.waitKey(0)
    
    #cv2.destroyAllWindows()
    return output

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    total = 0
    #win.clear_overlay()
    #win.set_image(img)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = face_detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k+1, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        faceData = []
        faceData.append(f)
        faceData.append(k+1)
        pt_array = []
        shape = predictor(img, d)
        for i in range(68):
            pt = (shape.part(i).x, shape.part(i).y)
            pt_array.append(pt)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                          shape.part(1)))
        # Draw the face landmarks on the screen.
        #win.add_overlay(shape)
        turn = turn_score((shape.part(3).x,shape.part(3).y), 
                          (shape.part(30).x,shape.part(30).y), 
                          (shape.part(13).x,shape.part(13).y))
        
        smile = smile_score((shape.part(51).x,shape.part(51).y), 
                            (shape.part(57).x,shape.part(57).y), 
                            (shape.part(62).x,shape.part(62).y), 
                            (shape.part(66).x,shape.part(66).y))
                                           
        left_eye_score = eye_score((shape.part(42).x,shape.part(42).y), 
                                   (shape.part(45).x,shape.part(45).y), 
                                   (shape.part(43).x,shape.part(43).y), 
                                   (shape.part(47).x,shape.part(47).y))
        
        right_eye_score = eye_score((shape.part(36).x,shape.part(36).y), 
                                   (shape.part(39).x,shape.part(39).y), 
                                   (shape.part(38).x,shape.part(38).y), 
                                   (shape.part(40).x,shape.part(40).y))
        
        eyes = left_eye_score[0]+right_eye_score[0]
        face_score = (turn[0] + smile + eyes)/3
        total += face_score
        faceData.append(face_score)
        faceData.append(turn)
        faceData.append(smile)
        faceData.append([eyes, "closed" if left_eye_score[1]=='closed' or \
                         right_eye_score[1]=='closed' else 'open'])
        faceData.append((d.left(),d.top(),d.right()-d.left(),d.bottom()-d.top()))
        faceData.append(pt_array)
        face_matrix.append(faceData)
    if len(dets)!=0:
        image_scores.append([f, total/len(dets)])
    else:
        image_scores.append(0)
    #face_matrix.append(faceData)
    #win.add_overlay(dets)
    dlib.hit_enter_to_continue()
    
ref_image = io.imread(max(image_scores)[0])
#plt.imshow(ref_image)
k = 0
finalImg = ref_image
refFile = []
for i in face_matrix:
    if i[filename] == max(image_scores)[0]:
        k = k+1
        refFile.append(i)
refFacePoints = []
for i in range(0,k):
    refFacePoints.append(refFile[i][facePoints])
    for j in face_matrix:
        if j[filename] != refFile[0][filename] and j[faceNumber] == refFile[i][faceNumber] and j[faceScore] > refFile[i][faceScore]:
            target_swap = refFile[i][facePoints] 
            swap_points = j[facePoints]
            swapFile = j[filename]
            finalImg = swap(refFile[0][filename], swapFile, target_swap, swap_points)
            plt.imshow(finalImg)

#image = swap(face_matrix[3][filename], face_matrix[4][filename], face_matrix[3][facePoints], face_matrix[4][facePoints])
#image2 = swap(face_matrix[0][filename], face_matrix[1][filename], face_matrix[0][facePoints], face_matrix[1][facePoints])
#if finalImg!=0:
cv2.imwrite("out.jpg", finalImg) 
#target = open("test1.jpg.txt", 'w')
#for i, j in enumerate(face_matrix[1][facePoints]):
#    target.write("{} {}".format(j[0],j[1]))
#    target.write("\n")
#target.close()




