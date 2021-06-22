from course import RelatedStudentExtractor
from library import Library, Books
import random


class CandidateGenerator:
    def __init__(self, course_json,
                 record_json, books_csv):
        self.rse = RelatedStudentExtractor(course_json) # 取得相關學生用
        self.lib = Library(record_json) # 取得學生借過的書
        self.books = Books(books_csv) # 書的屬性 (EX: 類別CDC、book_id、)
        self.stu2cands = {}

    # 產生候選書
    def generate(self, stu_id,  ground_book=None, n=30):
        if stu_id in self.stu2cands:
            cand_books = self.stu2cands[stu_id]
        else:
            try:
                related_stus = self.rse.get_related_students_id(stu_id) # 取得類似學生 STU_ID
            except KeyError: # 此學生沒借過書
                raise KeyError
            cand_books = set()
            for s in related_stus:
                try:
                    for b in self.lib.get_book_id_by_stuid(s):  
                        if b != ground_book:
                            cand_books.add(b)
                except KeyError:
                    pass
            self.stu2cands[stu_id] = cand_books
            
        cand_books = list(cand_books)
        if ground_book is not None:
            cand_books = random.sample(cand_books, k=min(n - 1, len(cand_books)))
            cand_books.append(ground_book)
            random.shuffle(cand_books)
        else:
            cand_books = random.sample(cand_books, k=min(n, len(cand_books)))
            
        result = {'book_id': cand_books,
                  'book_features': self.books.get_books_feature(cand_books),
                  'cand_books_index': [self.books.book_id2index[c] for c in cand_books],
                  }
        return result
