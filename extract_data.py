import json
import gzip
import random
import math
from itertools import islice
from collections import defaultdict
from m3inference import get_lang

random.seed(9)

split_ratio = {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2}
split = [0.6, 0.2, 0.2]
split_order = ['train', 'val', 'test']

src = "/export/c10/zach/data/demographics/descriptions/exact_gender_noun.json.gz"

all_users = defaultdict(dict)
print("loading data from source file {}".format(src))
# count = 0
with gzip.open(src, 'r') as inf:
    for line in inf:
        # count += 1
        # if count == 1000:
        #     break
        data = json.loads(line.decode('utf8'))
        if 'user' not in data.keys():
            continue
        tmp = {}
        user = data['user']       
        user_id = user['id_str']
        tmp['id'] = user_id
        tmp['name'] = user['name']
        tmp['screen_name'] = user['screen_name']
        if 'description' in user.keys():
            tmp['description'] = user['description']
        else:
            tmp['description'] = ''
        tmp['lang'] = get_lang(tmp['description'])
        tmp['gender'] = user['label'][0]
        # ignore users with multiple labels
        if len(user['label']) != 1:
            continue
        all_users[user_id] = tmp

print("spliting data")
id_list = list(all_users.keys())
random.shuffle(id_list)

# def writeSplitData(set_name):
#     if set_name == "train":
#         start = 0
#         end = math.floor(len(id_list)*split_ratio['train']) 
#     elif set_name == 'val':
#         start = math.floor(len(id_list)*split_ratio['train']) + 1
#         end = start + math.floor(len(id_list)*t)
Inputt = iter(id_list)
split_length = [math.floor(len(id_list)*ele) for ele in split]
output = [list(islice(Inputt, ele)) for ele in split_length]
for idx, part in enumerate(output):
    with open('./test/{}_data.jsonl'.format(split_order[idx]), 'w') as outf:
        for _id in part:
            user = all_users[_id]
            outf.write("{}\n".format(json.dumps(user)))

