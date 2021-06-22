import csv
import random
from argparse import ArgumentParser
from tqdm import tqdm

from model import LinUCB

from utils import CandidateGenerator


parser = ArgumentParser()
parser.add_argument('--train_record_csv', default='data/prep_2006_2011.csv')
parser.add_argument('--train_record_json',
                    default='data/history_2006_2011.json')
parser.add_argument('--train_course_json', default='data/course_train_with_dim.json')
parser.add_argument('--books_csv', default='data/item_all_new.csv')

args = parser.parse_args()
gen = CandidateGenerator(args.train_course_json,
                         args.train_record_json, args.books_csv) # 產生 candidate

train_records = list(csv.reader(open(args.train_record_csv)))[1:] # 每筆 record


def train(model, train_records, N=30):
    preds = [] # predicted result
    pbar = tqdm(total=len(train_records)) # progress bar
    for i, record in enumerate(train_records): # for each record
        pbar.update(1) # progress bar + 1
        ground_book, stu = record[0], record[8] # 借閱的書、借閱的學生學號
        stu_feature = gen.rse.get_student_feature(stu) # 取得學生 feature

        try:
            dct = gen.generate(stu, ground_book, N) # 從選課相似度高的其他學生借過的學生中選 N 本書做為 candidate
            
            # book id (dataset), 候選書本 feature(one hot), book index (在 Model 中)
            cand_books, cand_books_features, cand_books_index = dct['book_id'], dct['book_features'], dct['cand_books_index']
            a = model.train_one_records(stu_feature, # [STU_FEAT]
                                        cand_books_features, # [N, BOOK_FEAT]
                                        cand_books_index, # [N]
                                        cand_books.index(ground_book)) # int, 標準答案，取得是在這一批資料中第幾個
     
        except KeyError: # 此學生沒借過書
            continue
        
        # 計算 training MRR
        grounds = gen.lib.get_book_id_by_stuid(stu) # 取得此學生其他借過的書
        preds.append(gen.books.book_id[cand_books_index[a]] in grounds) # 模型取得的最好選擇有沒有在此學生借過的紀錄中
        if i % 1000 == 1:
            mrr = sum(preds) / float(len(preds))
            pbar.set_description(f"MRR: {mrr:.5f}")
    pbar.close()
    return preds


model = LinUCB(len(gen.books), gen.books.book_feature_size,
               gen.rse.stu_cc_features.shape[-1], alpha=0.8)
preds = train(model, train_records, N=50)
model.save('model.pt')
print(f'train_avg_reward: {sum(preds) / len(preds)}')
  