import cv2
import mediapipe as mp 
import time as tm 
import matplotlib.pyplot as plt
import numpy as np
import joblib as jb
import pandas as pd
import numpy as np
import tensorflow as tf
class PoseDetector:
    '''
    It is a pose detector class use is to detect 33 points in human body
    If upperBody is true then 25 points
    '''
    _poses={0:"Downdog Pose",1:"Tree Pose",2:"No Pose",3:"Goddess Pose",4:"Mountain Pose",5:"Warrior2 Pose"}
    def __init__(self,mode=False,smooth_lm=True,detectionConf=0.5,trackingConf=0.5):
        
        # parameters of model
        self.mode=mode
        self.upperBody=False
        self.smooth=smooth_lm
        self.detectionConf=detectionConf
        self.trackingConf=trackingConf

        #model
        self.mpPose= mp.solutions.pose
        self.model= self.mpPose.Pose(self.mode,self.upperBody,smooth_landmarks=self.smooth,
        min_detection_confidence=self.detectionConf,
        min_tracking_confidence=self.trackingConf)
        
        # Drawing Tools ===>Landmarks
        self.Draw=mp.solutions.drawing_utils

        # load the prediction model
        #self.model_pose_class=jb.load("PoseClassModel.pkl")  # ===> Old == 3 poses
        self.model_pose_class=tf.keras.models.load_model("MyAnnPose1.h5")
    
    def findPose(self,img,draw=True):
        img=img.copy()
        # Step 1 :  Cvt to RGB and take the result
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #print("Image Given",img_rgb.shape)
        self.results=self.model.process(img_rgb)
        #print(self.results.pose_landmarks.landmark)
        #Step2 : Check that is there any points
        if self.results.pose_landmarks:
            #Step3 : It Checkes user need to draw the points for not
            if draw:
                landmarks=self.results.pose_landmarks
                
                # Setting Color and thickness of Connections and Landmarks
                lm_draw_spec=self.Draw.DrawingSpec(thickness=2, circle_radius=2,color=(123,22,76))
                conn_draw_spec=self.Draw.DrawingSpec(thickness=3,color=(250,44,250),circle_radius=5)
                
                ## Drawing Landmarks 
                self.Draw.draw_landmarks(img,landmarks,self.mpPose.POSE_CONNECTIONS,landmark_drawing_spec=lm_draw_spec,connection_drawing_spec=conn_draw_spec)
                img=cv2.resize(img,(640,480))
                return img
            else:
                img=cv2.resize(img,(640,480))
                return img
            
            
        else:
            print("No Human is There")
            return img
    
    def getPosition(self,img,draw=True):
        lmdict={}
        points=self.mpPose.PoseLandmark
        names=list(map(lambda x : x.name,list(points)))
        #print("NAMES: ",names)
        if self.results.pose_landmarks:
            landmark=self.results.pose_landmarks.landmark
            for id,landmark in enumerate(landmark):
                # ID the index of lm
                ht,wth,ch= img.shape
                px,py,pz=int(landmark.x*wth),int(landmark.y*ht),int(landmark.z*ch)
                #px,py,pz=landmark.x,landmark.y,landmark.z
                # Put in the dict
                # Prediction below line is change differes from dataset creation
                lmdict[names[id]+"_X"],lmdict[names[id]+"_Y"],lmdict[names[id]+"_Z"]=(landmark.x,landmark.y,landmark.z)
                

                if draw:
                    aqua=(255,255,0)
                    cv2.circle(img,(px,py),2,aqua,cv2.FILLED)
            return lmdict
        return None
    def PredictPose(self,dct):
        if dct:
            #
            #print(dct)
            X=pd.DataFrame(dct,index=[0]).values.reshape(1,-1)  # For one value only we need to pass inde
            y_pd= np.argmax(self.model_pose_class.predict(X),axis=1)[0]
            #print(y_pd)
            name= PoseDetector._poses[y_pd]
            #probability=self.model_pose_class.predict_proba(X)
            return name
        else:
            return PoseDetector._poses[2]



#myd=PoseDetector()





#plt.imshow(img)
#x,y=x*img.shape[1],y*img.shape[0]
#plt.scatter(x[:3],y[:3])
#plt.show()
'''

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.imshow("Pose Estimated",imgPose)
cv2.waitKey(0) 
'''







if __name__=="__main__":
    x = np.array([[ 0.49792907,  0.28732818, -0.2154572 ,  0.51059067,  0.2581543 ,
        -0.18288292,  0.5182963 ,  0.25947154, -0.18289085,  0.52476424,
         0.26096183, -0.18291025,  0.48634654,  0.25266859, -0.19118372,
         0.47797781,  0.2497822 , -0.19137213,  0.47009575,  0.24659458,
        -0.19140539,  0.5289923 ,  0.27357891,  0.00985628,  0.45186186,
         0.24891096, -0.02724816,  0.50674605,  0.3155441 , -0.14122768,
         0.48114225,  0.31068677, -0.15237489,  0.56223047,  0.4010469 ,
         0.07711304,  0.38639545,  0.39872676, -0.01962954,  0.58391553,
         0.61547172, -0.09704317,  0.38779652,  0.62055224, -0.16539319,
         0.54660642,  0.53407538, -0.4024967 ,  0.47762543,  0.5232802 ,
        -0.39172047,  0.5497402 ,  0.51523829, -0.44303089,  0.50588012,
         0.50834489, -0.43812221,  0.5459221 ,  0.48447883, -0.43472549,
         0.5057354 ,  0.48445877, -0.42605159,  0.54062521,  0.49612498,
        -0.40211704,  0.4995974 ,  0.48831165, -0.38672087,  0.55686486,
         0.74039876,  0.03210373,  0.45418808,  0.76304394, -0.03189586,
         0.70638025,  0.62129694, -0.32984698,  0.40747425,  0.71691781,
        -0.50597328,  0.52549964,  0.83563137, -0.29710716,  0.58216184,
         0.8092972 , -0.34199658,  0.50213057,  0.83558124, -0.28405127,
         0.58379245,  0.76732749, -0.31154224,  0.45516497,  0.87212342,
        -0.37959963,  0.66647327,  0.80318916, -0.37612185]])
    # Load model
    mod = PoseDetector()
    model=mod.model_pose_class
    print(np.argmax(model.predict(x),axis=1)) # Columnwise aggreate max value

#    img=cv2.imread("Yoga.png")
#   mydetct = PoseDetector()
    #jb.dump(mydetct,"MyDetector.pkl")
#    imgPose=mydetct.findPose(img)
#    points=mydetct.getPosition(img,draw=False)
#    cv2.imshow("Mu Detect",imgPose)
#    cv2.waitKey(0)
    #fig=plt.figure()
    #ax= plt.axes(projection="3d")
    #point=np.array( list(points.values()))
    #x,y,z= point[:,0],point[:,1],point[:,2]
    #cap=cv2.VideoCapture("pose_segmentation.mp4")
    
    '''
    cap=cv2.VideoCapture("pose4.mp4")
    prevT=0
    
    myDetector=PoseDetector()
    while True:
        succ,img=cap.read()
        if succ:
            #Img format
            img = cv2.flip(img,1) #1 -->Horizontally
            img=cv2.resize(img,(640,480))


            # Detected Image with points
            img=myDetector.findPose(img,draw=False)
            # Getting pixels
            positions=myDetector.getPosition(img,draw=False)
            print(positions[14])
            cv2.circle(img,positions[14],5,(0,0,255),cv2.FILLED)

            #FrameRate
            currT = tm.time()
            fps= 1//(currT-prevT)
            fps_st="FPS : {}".format(fps)
            cv2.putText(img,fps_st,(10,50),cv2.FONT_HERSHEY_TRIPLEX,fontScale=1,color=(0,0,255),thickness=2)
            cv2.imshow("My Detections",img)
            k=cv2.waitKey(1)
            if k & 0xff ==ord('k') or k & 0xff == ord('K'):
                break
            prevT=currT
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    '''

        


        






