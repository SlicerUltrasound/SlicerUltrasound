import os
import io
import json
import re
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import pydicom
from pydicom.encaps import generate_pixel_data_frame, decode_data_sequence
from tqdm import tqdm
import logging

def read_frames_from_dicom(dicom_file_path):
    """
    Reads frames from a dicom file and returns a numpy array in the format of [Frames, Channels, Height, Width]. PyTorch convention is NCHW for image batches.
    :param dicom_file_path: path to the dicom file
    :return: numpy array [N,C,H,W]
    """
    ds = pydicom.dcmread(dicom_file_path)
    width = ds.Columns
    height = ds.Rows
    channels = ds.SamplesPerPixel

    try:
        num_frames = ds.NumberOfFrames
    except:
        num_frames = 1
        print(f"Warning: No NumberOfFrames found in {dicom_file_path}, trying to read with num_frames=1")
    
    output = np.zeros((num_frames, channels, height, width), dtype=np.uint8)
    
    try:
        pixel_data_frames = generate_pixel_data_frame(ds.PixelData)
        
        for i in range(num_frames):
            frame_item = next(pixel_data_frames)
            image = Image.open(io.BytesIO(frame_item))  # jpeg uncompressed
            frame = np.array(image)
            # If frame is grayscale, add a channel dimension
            if len(frame.shape) == 2:
                frame = np.expand_dims(frame, axis=2)
            frame = np.transpose(frame, (2, 0, 1))  # Convert from rows, cols, channels to channels, rows, cols
            output[i, :, :, :] = frame
    except Exception as e:
        try:
            # Fallback to decode_data_sequence approach
            frame_data = decode_data_sequence(ds.PixelData)
            
            for i, frame_item in enumerate(frame_data):
                if i >= num_frames:
                    break
                image = Image.open(io.BytesIO(frame_item))  # jpeg uncompressed
                frame = np.array(image)
                # If frame is grayscale, add a channel dimension
                if len(frame.shape) == 2:
                    frame = np.expand_dims(frame, axis=2)
                frame = np.transpose(frame, (2, 0, 1))  # Convert from rows, cols, channels to channels, rows, cols
                output[i, :, :, :] = frame

        except Exception as e:
            try:
                # Fallback to ds.pixel_array approach
                pixel_data_frames = ds.pixel_array # this seems to be more robust? but slower
                # ensure that the shape is (num_frames, height, width, channels). 
                # it is sometimes (height, width, channels)
                if len(pixel_data_frames.shape) == 3 and num_frames == 1:
                    pixel_data_frames = np.expand_dims(pixel_data_frames, axis=0)
                
                for i in range(num_frames):
                    frame = pixel_data_frames[i, :, :]
                    if len(frame.shape) == 2:
                        frame = np.expand_dims(frame, axis=2)
                    frame = np.transpose(frame, (2, 0, 1))  # Convert from rows, cols, channels to channels, rows, cols
                    output[i, :, :, :] = frame

            except Exception as e:
                raise e
    return output
