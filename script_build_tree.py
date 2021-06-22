from argparse import ArgumentParser
from anytree import RenderTree
from cdc_tree import CDCTree
from tqdm import tqdm
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--item', type=str)
parser.add_argument('--record', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()
tree = CDCTree()

err_cnt = 0
item_df = pd.read_csv(args.item)
for book_idx, cdc, title, create_date, pub in zip(tqdm(item_df['book_index1']),
                                item_df['CDC_code'],
                                item_df['title'],
                                item_df['create_date'],
                                item_df['publisher'],
                                ):
    try:
        cdc = cdc.strip().split(' ')[0].split('.')[0]
    except AttributeError:
        continue
    try:
        tree.add_book(cdc, book_idx, title, create_date, pub)
    except ValueError:
        err_cnt += 1
print(err_cnt)
df = pd.read_csv(args.record)
for (index, row), _ in zip(df.iterrows(), tqdm(range(len(df)))):
    try:
        tree.update(row['CDC_code'].split(' ')[0].split('.')[0], row['book_index1'], row['title'])
    except Exception:
        pass

tree.save(args.output)
tree = CDCTree.from_built(args.output)
print(RenderTree(tree.root))
