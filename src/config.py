import os
from tensorflow.keras import backend

'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

'''
# channel first -> B C H W
# channel last  -> B H W C 
'''
backend.set_image_data_format('channels_last')

GPUs=["GPU:0", "GPU:1"]

DATA_LABEL_PATH="../data/label.txt"
