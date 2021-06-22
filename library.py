import json, csv
import torch
from typing import List


class Books:
    def __init__(self, item_csv: str):
        self.book_id = []
        self.category = []
        self.cdc2cat = {}
        self.create_date = []
        self.book_id2index = {}
        self.prepare(item_csv)
        
        # 以 onehot 方式代表每本書
        # 此方法為測試用
        # 因為沒辦法完全表示類別與類別間關係
        # 可以結合 cdc_tree 裡面的 tree sturucture 取得更好 feature
        # 或是加入書的其他屬性(EX: 出版日期)
        self.book_feature_size = len(self.cdc2cat) 
        self.cat2one_hot = torch.eye(self.book_feature_size)
    
    def prepare(self, item_csv):
        with open(item_csv)as f:
            rows = csv.reader(f)
            for r in rows: # book_index1,title,CDC_code,create_date,publisher
                self.book_id2index[r[0]] = len(self.book_id)
                self.book_id.append(r[0])
                cdc = r[2].split()[0].split('.')[0][:2]
                if cdc not in self.cdc2cat:
                    self.cdc2cat[cdc] = len(self.cdc2cat)
                self.category.append(self.cdc2cat[cdc])
                self.create_date.append(r[3])
    
    #  取得多本書的 init feature (以類別CDC)
    def get_books_feature(self, book_idxs: List):
        idxs = [self.book_id2index[bid] for bid in book_idxs]
        categorys = [self.category[idx] for idx in idxs]
        feats = self.cat2one_hot[categorys]
        return feats
    
    def __len__(self):
        return len(self.book_id)


class Library:
    def __init__(self, 
                 record_json: str):
        with open(record_json) as f:
            self.stu_id2book = json.load(f)

    def get_records_by_stuid(self, stu_id: str):
        try:
            return self.stu_id2book[stu_id]
        except KeyError:
            raise KeyError("cannot find stu_id")
        # return self.records[self.records['student_id'] == stu_id]
    
    def get_book_id_by_stuid(self, stu_id: str):
        """get someone's lent books

        Args:
            stu_id (str): student id
        """
        return [r[0] for r in self.get_records_by_stuid(stu_id)]


if __name__ == '__main__':
    lib = Library('data/history_2012.json', 'data/item_all_new_2012.csv')
    # print(lib.find_record_by_stuid('400110002'))
    print(lib.books.get_books_feature(['004990157', '004990158']).shape)