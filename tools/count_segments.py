'''
    count how many chunks are in each dataset
'''
import torch
import torch.utils.data as data
import json
import os
sr = 8000
seglen = 4 * sr
minlen = 2 * sr
root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/dataset"
def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def count_chunks(json_folders):
    for json_folder in json_folders:
        mixfiles, wavlens = list(zip(*load_json(os.path.join(root, json_folder, 'mix.json')))) # list of 20000 filenames, and 20000 lengths
        num_chunks = 0
        for wavlen in wavlens:
            num_chunks += ((wavlen - seglen) // minlen + 1)
        print(json_folder, num_chunks)

if __name__ == "__main__":
    tr_json = ["2spkr_json/tr/",
            "3spkr_json/tr/",
            "4spkr_json/tr/",
            "5spkr_json/tr/"]
    val_json = ["2spkr_json/cv/",
                "3spkr_json/cv/",
                "4spkr_json/cv/",
                "5spkr_json/cv/"]
    test_json = ["2spkr_json/tt",
                "3spkr_json/tt",
                "4spkr_json/tt",
                "5spkr_json/tt"]
    count_chunks(tr_json)
    count_chunks(val_json)
    count_chunks(test_json)

