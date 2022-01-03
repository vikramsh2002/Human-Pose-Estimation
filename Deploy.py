import streamlit as st 
from PIL import Image
import numpy as np
from PoseDetection import PoseDetector as mpd
import cv2
import os
import matplotlib.pyplot as plt
st.title("VRL Human Pose Estimator")
st.image("MyLogoLight.png")
st.cache()
def getmodel():
    return mpd(detectionConf=0.7)
def getPose_Color_Align(name_pose):
    align="right"
    color="aqua"
    if name_pose=="No Pose":
            align="left"
            color="red"
    else:
        align="center"
        if name_pose=="Tree Pose":
            color="green"
        elif name_pose=="Downdog Pose":
            color="brown"
        elif name_pose=="Mountain Pose":
            color="blue"
        elif name_pose=="Goddess Pose":
            color="yellow"
        elif name_pose=="Warrior2 Pose":
            color="purple"
    return align,color
def RunVideo(name=0):
    cap=cv2.VideoCapture(name)
    emp=st.empty()
    while True:
        stat,frame=cap.read()
        if not stat:
            break  # No Frame left
        # Find the Pose and marked frame
        frame_mark = model.findPose(frame)
        # Get the Positions
        pos=model.getPosition(frame)
        # Get Pose Name
        name_pose=model.PredictPose(pos)  # Take the positions and Predict the Pose
        # Put the pose name
        cv2.putText(frame_mark,name_pose,(50,50),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)
        
        align,color=getPose_Color_Align(name_pose)
        emp.markdown("<h1 style='color:{};text-align:{}'>{}</h1></center>".format(color,align,name_pose),unsafe_allow_html=True)
                
        #Show the video
        frame_mark=cv2.resize(frame_mark,(640,480))
        cv2.imshow("Pose Detection",frame_mark)
                
        
        k=cv2.waitKey(1)
        if k&0xff in [ord("q"),ord("Q"),ord("k"),ord("K")]:
            break
    cap.release()
    cv2.destroyAllWindows()
    if name:
        os.remove(name) # No Extra Space we use as Work over
    emp.markdown("")



type_file=st.sidebar.selectbox("Choose the Type of File to Estimate",["Image","Video","Webcam"])
model=getmodel()
if type_file=="Image":
    s=st.file_uploader(label="Dump The Image",type=["jpg","png","jfif"])
    plot= st.checkbox(label="Show 3D Plot")
    
    b=st.button("Pose Estimate")
    if b:
        
        if s:
            img=np.array(Image.open(s))
            
            #print(img.shape[2])          
            if img.shape[2]>3:
                img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
                
            #print(img.shape)
            
            img_Detect=model.findPose(img)
            dct=model.getPosition(img)
            name_pose=model.PredictPose(dct)
            align,color=getPose_Color_Align(name_pose)


            col1,col2=st.columns(2)
            st.markdown("<center><h1 style='color:{}; test-align:{};'>{}</h1> </center>".format(color,align,name_pose) ,unsafe_allow_html=True )
            col1.image(img_Detect)
            #st.image([img,img_Detect])
            if plot:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                col2.pyplot(model.Draw.plot_landmarks(model.results.pose_world_landmarks, model.mpPose.POSE_CONNECTIONS))
        else:
            print("No Link Image")
elif type_file=="Video":
    vid=st.file_uploader("Dump The Video Here..",type=["mp4","avi","wmv"])
    if vid:
        b=st.button("Pose Estimate")
        
        if b:
            name=vid.name
            with open(name,"wb") as f:
                f.write(vid.read())
            RunVideo(name)
else:
    b=st.button("Pose Estimate On Webcam")
    if b:
        RunVideo()

            
    


