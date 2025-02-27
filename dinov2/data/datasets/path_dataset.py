import PIL.Image as Image
import sys
import os
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data as data
import math
import time
import h5py
from multiprocessing import Pool, cpu_count, Manager, Value
from tqdm import tqdm

class training_tile_dataset(data.Dataset):
    def __init__(self, data_len=None, rank=None, transform=None, tile_size=224, schedule_path=None):
        self.schedule_path = schedule_path
        self.transform = transform
        self.rank = rank
        self.data_len = data_len
        self.slide_data = pd.read_csv('path/to/metadata.csv', dtype=str) # File with slide information: slide, slide_path
        schedules = pd.read_csv(f'path/to/schedule_list.csv') # File that summarizes the schedules: epoch, schedule path
        self.schedules = dict(zip(schedules.epoch, schedules.schedule_path))
        self.current_schedule = None # slide, x, y
        self.slides = None #{slide_id: h5 file}
    
    def _load_schedule(self, epoch):
        self.current_schedule = pd.read_csv(os.path.join(f'{self.schedule_path}', self.schedules[epoch]), dtype=str)
        data_len = len(self.current_schedule)
        return data_len
    
    def _load_slides(self):
        unique_slides = self.current_schedule.slide_id.unique()
        current_slides = self.slide_data[self.slide_data.slide_id.isin(unique_slides)].reset_index(drop=True)
        start_time = time.time()

        self.slides = {}

        for _, row in current_slides.iterrows():
            slide_id, slide_path = row.slide_id, row.slide_path
            try:
                h5_file = h5py.File(slide_path, 'r')
                self.slides[slide_id] = h5_file
            except Exception as e:
                print(f"Error opening slide {slide_id}: {e}")

        if self.rank == 0:
            elapsed_time = time.time() - start_time
            print(f'\nOpened handles for {len(self.slides)} slides in {elapsed_time:.2f} seconds')
    
    def set_data(self, epoch):
        data_len = self._load_schedule(epoch)
        self._load_slides()
        self.data_len = data_len
        print(f"Load {self.data_len} tiles for epoch {epoch}...")
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):  
        # Get the row corresponding to the current index
        row = self.current_schedule.iloc[index]
        
        # Retrieve the HDF5 file for the slide and image bytes
        try:
            h5_file = self.slides[row.slide_id]
            img_bytes = h5_file[row.ind][:]
        except KeyError as e:
            raise KeyError(f"Error accessing slide ID {row.slide_id} or key {row.ind}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error accessing data for slide ID {row.slide_id}, key {row.ind}: {e}")
        
        # Reconstruct the PIL Image
        try:
            img = Image.frombytes("RGB", (224, 224), img_bytes.tobytes()) # hardcoded size --> TODO: change to self.tile_size
        except Exception as e:
            raise RuntimeError(f"Error reconstructing image for slide ID {row.slide_id}, key {row.ind}: {e}")
        
        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)
        
        return img, None
