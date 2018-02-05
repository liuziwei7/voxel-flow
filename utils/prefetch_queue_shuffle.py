from __future__ import print_function

import glob
import numpy as np
import os 
import Queue
import random
import scipy
from scipy import misc
import threading 

class DummpyData(object):
  def __init__(self, data):
    self.data = data
  def __cmp__(self, other):
    return 0

def prefetch_job(load_fn, prefetch_queue, data_list, shuffle, prefetch_size):
  """
  """
  data_count = 0
  total_count = len(data_list)
  idx = 0
  while True:
    if shuffle:
      if data_count == 0:
        random.shuffle(data_list)
      data = load_fn(data_list[data_count]) #Load your data here.
      if type(data) is list:
        for data_point in data: 
          idx = random.randint(0, prefetch_size)
          dummy_data = DummpyData(data_point)
          prefetch_queue.put((idx, dummy_data), block=True)
      else:
        idx = random.randint(0, prefetch_size)
        dummy_data = DummpyData(data)
        prefetch_queue.put((idx, dummy_data), block=True)
    else:
      data = load_fn(data_list[data_count]) #Load your data here.
      dummy_data = DummpyData(data)
      prefetch_queue.put((idx, dummy_data), block=True)
      idx = (idx + 1) % prefetch_size

    data_count = (data_count + 1) % total_count

class PrefetchQueue(object):
  def __init__(self, load_fn, data_list, batch_size=32, prefetch_size=None, shuffle=True, num_workers=4):
    self.data_list = data_list
    self.shuffle = shuffle
    self.prefetch_size = prefetch_size
    self.load_fn = load_fn
    self.batch_size = batch_size
    if prefetch_size is None:
      self.prefetch_size = 4 * batch_size

    # Start prefetching thread
    # self.prefetch_queue = Queue.Queue(maxsize=prefetch_size)
    self.prefetch_queue = Queue.PriorityQueue(maxsize=prefetch_size)
    for k in range(num_workers):
      t = threading.Thread(target=prefetch_job,
        args=(self.load_fn, self.prefetch_queue, self.data_list,
              self.shuffle, self.prefetch_size))
      t.daemon = True
      t.start()

  def get_batch(self):
    data_list = []
    for k in range(0, self.batch_size):
      # if self.prefetch_queue.empty():
      #   print('Prefetch Queue is empty, waiting for data to be read.')
      _, data_dummy = self.prefetch_queue.get(block=True)
      data = data_dummy.data
      data_list.append(np.expand_dims(data,0))
    return np.concatenate(data_list, axis=0)


if __name__ == '__main__':
  # Simple Eval Script For Usage.
  def load_fn_example(data_file_path):
    return scipy.misc.imread(data_file_path)

  import time
  data_path_pattern = '/home/VoxelFlow/dataset/ucf101/*.jpg' 
  data_list = glob.glob(data_path_pattern)  # dataset.read_data_list_file()
  load_fn = load_fn_example  # dataset.process_func()
  num_workers=2
  batch_size = 32
  
  # Prefetch IO.
  p_queue = PrefetchQueue(load_fn, data_list, batch_size, num_workers=num_workers)
  time.sleep(5)
  print('Start') 
  import datetime
  a = datetime.datetime.now()
  for k in range(0,50):
    time.sleep(0.1)
    X = p_queue.get_batch()
  b = datetime.datetime.now()
  delta = b - a
  print(delta)
  print("%d miliseconds" % int(delta.total_seconds()))

  # Naive FILE IO
  import glob
  data_list = glob.glob(data_path_pattern) 
  a = datetime.datetime.now()
  for k in range(0,50):
    time.sleep(0.1)
    data_sub_list = data_list[k*batch_size:(k+1)*batch_size]
    im_list = [np.expand_dims(scipy.misc.imread(file_name),0) for file_name in data_sub_list]
    X = np.concatenate(im_list,axis=0)  

  b = datetime.datetime.now()
  delta = b - a
  print(delta)
  print("%d miliseconds" % int(delta.total_seconds()))
