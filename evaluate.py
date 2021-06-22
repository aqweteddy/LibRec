import csv
from argparse import ArgumentParser

from tqdm import tqdm

from metrics import roc_auc, mrr
from model import LinUCB
from utils import CandidateGenerator

parser = ArgumentParser()
parser.add_argument('--ckpt', default='model_old.pt')
parser.add_argument('--test_record_csv', default='data/prep_2012.csv')
parser.add_argument('--test_record_json',
                    default='data/history_2012.json')
parser.add_argument('--test_course_json',
                    default='data/course_test_with_dim.json')
# parser.add_argument('--course_category_csv',
#                     default='data/course_category.csv')
parser.add_argument('--books_csv', default='data/item_all_new.csv')
args = parser.parse_args()

model = LinUCB.load(args.ckpt)
gen = CandidateGenerator(args.test_course_json,
                         args.test_record_json, args.books_csv)
records = list(csv.reader(open(args.test_record_csv)))[1:]


def eval(model, records, N=30):
    tot_acc = []

    pbar = tqdm(total=len(records))
    for i, record in enumerate(records[:500]):
        pbar.update(1)
        ground_book, stu = record[0], record[8]
        stu_feature = gen.rse.get_student_feature(stu)

        try:
            dct = gen.generate(stu, n=N)
            cand_books, cand_books_features, cand_books_index = dct[
                'book_id'], dct['book_features'], dct['cand_books_index']
        except KeyError:
            continue
        preds = model.predict(stu_feature,
                            cand_books_index,
                            cand_books_features)

        ground_books = gen.lib.get_book_id_by_stuid(stu)
        grounds = [int(c in ground_books) for c in cand_books]
        try:
            tot_acc.append(mrr(preds, grounds))
        except ZeroDivisionError:
            print(cand_books, preds, grounds)
    print(f'MRR: {sum(tot_acc) / len(tot_acc)}')
    pbar.close()
    return preds


eval(model, records, N=30)
