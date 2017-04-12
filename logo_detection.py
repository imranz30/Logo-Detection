import cv2
import numpy

oculyze_cv_task =cv2.imread('oculyze_cv_task.jpg')
oculyze_logo =cv2.imread('oculyze_logo.png')

ngrey = cv2.cvtColor(oculyze_logo, cv2.COLOR_BGR2GRAY)
hgrey = cv2.cvtColor(oculyze_cv_task, cv2.COLOR_BGR2GRAY)

# build feature detector and descriptor extractor
hessian_threshold = 2000
detector = cv2.SURF(hessian_threshold)
hkeypoints,hdescriptors = detector.detectAndCompute(hgrey,None)
nkeypoints,ndescriptors = detector.detectAndCompute(ngrey,None)

# extract vectors of size 64 from raw descriptors numpy arrays
rowsize = len(hdescriptors) / len(hkeypoints)
if rowsize > 1:
    hrows = numpy.array(hdescriptors, dtype = numpy.float32).reshape((-1, rowsize))
    nrows = numpy.array(ndescriptors, dtype = numpy.float32).reshape((-1, rowsize))
    #print hrows.shape, nrows.shape
else:
    hrows = numpy.array(hdescriptors, dtype = numpy.float32)
    nrows = numpy.array(ndescriptors, dtype = numpy.float32)
    rowsize = len(hrows[0])

# kNN training - learn mapping from hrow to hkeypoints index
samples = hrows
responses = numpy.arange(len(hkeypoints), dtype = numpy.float32)
#print len(samples), len(responses)
knn = cv2.KNearest()
knn.train(samples,responses)

# creating a combined image to draw lines between matches to identify logos
(hA, wA) = oculyze_cv_task.shape[:2]
(hB, wB) = oculyze_logo.shape[:2]
vis = numpy.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = oculyze_cv_task
vis[0:hB, wA:] = oculyze_logo

# retrieve index and value through enumeration
for i, descriptor in enumerate(nrows):
    descriptor = numpy.array(descriptor, dtype = numpy.float32).reshape((1, rowsize))
    #print i, descriptor.shape, samples[0].shape
    retval, results, neigh_resp, dists = knn.find_nearest(descriptor, 1)
    res, dist =  int(results[0][0]), dists[0][0]
    #print res, dist

    if dist < 0.3:
        # draw matched keypoints in red color
        color = (0, 0, 255)

    # draw matched key points on oculyze_cv_task image
    x1,y1 = hkeypoints[res].pt
    center1 = (int(x1),int(y1))
    cv2.circle(oculyze_cv_task,center1,2,color,-1)

    # draw matched key points on oculyze_logo image
    x2,y2 = nkeypoints[i].pt
    center = (int(x2),int(y2))
    c = (int(x2)+wA, int(y2))
    cv2.circle(oculyze_logo,center,2,color,-1)

    # draw lines on the combined image to identify logos
    cv2.line(vis,center1,c,(0,255,0),1)

cv2.imwrite("result.png", vis)
cv2.imshow('oculyze_cv_task',oculyze_cv_task)
cv2.imshow('oculyze_logo',oculyze_logo)
cv2.imshow('detected_logos', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
