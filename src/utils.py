import config

def load_label(data_label_file: str) -> List:
    fp = open(data_label_file, 'r')
    label = fp.read().split('\n')[:-1]
    fp.close()
    return label