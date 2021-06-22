from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import json


parser = ArgumentParser()
parser.add_argument('--input', type=str, nargs='+')
parser.add_argument('--output', type=str)
parser.add_argument('--item_input', type=str)
parser.add_argument('--item_output', type=str)
parser.add_argument('--history_output', type=str)
args = parser.parse_args()


def get_title_isbnidx(its: list):
    title = ''
    for idx, it in enumerate(its):
        it = it.replace('-', '')
        if not it.split(' ')[0].isdigit():
            title += it
        else:
            break

    return title, idx

# 008: 學位論文
def is_legal(cdc_code: str):
    cdc_code = cdc_code.strip()
    if len(cdc_code) < 3:
        return None
    cdc = cdc_code.split(' ')[0]
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
    if all(map(lambda c: c in chars, cdc)) and '008' not in cdc:
        return cdc_code
    else:
        return None


item_dct = {}  # title: {other prop.}
with open(args.item_input) as f:
    for line in f.readlines():
        d = line.strip().split(',')
        title, isbn_idx = get_title_isbnidx(d[4:])
        cdc = is_legal(d[2])
        if cdc:
            item_dct[str(d[0])] = {'title': title.strip(' /'),
                              'CDC_code':cdc,
                              'create_date': d[3],
                              'publisher': d[-1]}


def preprocess(file: str):
    data = []
    with open(file) as f:
        for line in f.readlines():
            d = list(map(lambda x: x.strip(' /'), line.strip().split(',')))
            dct = {'book_index1': str(d[0]),
                   'book_index2': str(d[1]),
                   'CDC_code': is_legal(d[2]),
                   'ISBN': d[-6],
                   'identifier': d[-5],
                   'borrow_date': d[-4],
                   'expexted_return_date': d[-3],
                   'return_date': d[-2],
                   'student_id': d[-1],
                   'title': ''.join(d[3:-6])
                   }

            if d[0] in item_dct:
                dct = {**dct, **item_dct[str(d[0])]}
                data.append(dct)
    return pd.DataFrame(data)


df = [preprocess(file) for file in tqdm(args.input)]
df = pd.concat(df)
people = {}
df.to_csv(args.output, index=False)
for p_id, b_id, date in zip(tqdm(df['student_id']), df['book_index1'], df['borrow_date']):
    if p_id in people:
        people[p_id].append((b_id, date))
    else:
        people[p_id] = [(b_id, date)]

with open(args.history_output, 'w') as f:
    json.dump(people, f)

item_dct = [{'book_index1': k, **v} for k, v in item_dct.items()]
print(f'numbers of books: {len(item_dct)}')
print(f'numbers of records: {len(df)}')
dct_df = pd.DataFrame(item_dct)

print('end')
dct_df.to_csv(args.item_output, index=False)
