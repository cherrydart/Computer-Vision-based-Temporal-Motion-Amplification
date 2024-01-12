import cv2
from cv2 import *
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack


class motion_amplify():

    def __init__(self, input_loc, output_loc):
        self.input_loc = input_loc
        self.output_loc = output_loc

    #Build Gaussian Pyramid
    def build_gaussian_pyramid(self,src,level=3):
        print("Building Gaussian Pyramid")
        s=src.copy()
        pyramid=[s]
        for i in range(level):
            s=cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid

    #Build Laplacian Pyramid
    def build_laplacian_pyramid(self,src,levels=3):
        print("Building Laplacian Pyramid")
        gaussianPyramid = self.build_gaussian_pyramid(src, levels)
        pyramid=[]
        for i in range(levels,0,-1):
            GE=cv2.pyrUp(gaussianPyramid[i])
            L=cv2.subtract(gaussianPyramid[i-1],GE)
            pyramid.append(L)
        return pyramid

    #load video from file
    def load_video(self, video_filename):
        print("Loading the video")
        cap=cv2.VideoCapture(video_filename)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_tensor=np.zeros((frame_count,height,width,3),dtype='float')
        x=0
        while cap.isOpened():
            ret,frame=cap.read()
            if ret is True:
                video_tensor[x]=frame
                x+=1
            else:
                break
        return video_tensor

    #save video to files
    def save_video(self, video_tensor):
        print("Saving the video")
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        [height,width]=video_tensor[0].shape[0:2]
        writer = cv2.VideoWriter(self.output_loc, fourcc, 30, (width, height), 1)
        for i in range(0,video_tensor.shape[0]):
            writer.write(cv2.convertScaleAbs(video_tensor[i]))
        writer.release()

    #build laplacian pyramid for video
    def laplacian_video(self,video_tensor,levels=3):
        print("Building video from Laplacian Pyramid")
        tensor_list=[]
        for i in range(0,video_tensor.shape[0]):
            frame=video_tensor[i]
            pyr=self.build_laplacian_pyramid(frame,levels=levels)
            if i==0:
                for k in range(levels):
                    tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
            for n in range(levels):
                tensor_list[n][i] = pyr[n]
        return tensor_list

    #butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, order=5):
        print("Applying a filter")
        omega = 0.5 * self.fps
        low = lowcut / omega
        high = highcut / omega
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.lfilter(b, a, data, axis=0)
        return y

    #reconstract video from laplacian pyramid
    def reconstract_from_tensorlist(self, filter_tensor_list,levels=3):
        print("Reconstructing the video")
        final=np.zeros(filter_tensor_list[-1].shape)
        for i in range(filter_tensor_list[0].shape[0]):
            up = filter_tensor_list[0][i]
            for n in range(levels-1):
                up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]
            final[i]=up
        return final

    #manify motion
    def magnify_motion(self, low, high, channel,amplification):
        print("Magnifying the video")
        t=self.load_video(self.input_loc)
        lap_video_list=self.laplacian_video(t,channel)
        filter_tensor_list=[]
        for i in range(channel):
            filter_tensor = self.butter_bandpass_filter(lap_video_list[i],low,high)
            filter_tensor *= amplification
            filter_tensor_list.append(filter_tensor)
        recon = self.reconstract_from_tensorlist(filter_tensor_list)
        final = t+recon
        self.save_video(final)

if __name__== "__main__":
    # magnify_color("baby.mp4",0.4,3)
    output_path_name = "D:\\Magnified_hand.avi"
    input_path_name = "D:\\Python\\IMG_4143.mp4"
    motion_amplification = motion_amplify(input_path_name, output_path_name)
    motion_amplification.magnify_motion(1, 2, 3, 20)
    