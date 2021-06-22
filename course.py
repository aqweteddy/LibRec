import json
import torch


class RelatedStudentExtractor:
    def __init__(self, course_json: str):
        """取得 學生 特徵
        (1) 由選過的課程 [1, 406] one-hot
        (2) 選過的課程類別 [1, 14] one-hot

        Args:
            course_json (str): course_train_with_dim.json
        """
        with open(course_json) as f:
            self.stu2course = json.load(f)

        self.stu2course = {k['studentID']: k for k in self.stu2course}
        # self.course2dim(course_id2dim), dim2index (dim: 通識向度)
        self.course2stu, self.stu2property, self.course2dim, self.dim2index = self.__preproess(
            self.stu2course)

        self.course_names = list(self.course2stu.keys())
        self.stu_ids = list(self.stu2course.keys())

        # 由學校課程編號轉為要放入模型的編號(從0開始)，例如: 7407015 -> 0
        self.course_name2index = {name: idx for idx,
                                  name in enumerate(self.course_names)}
        # 學號轉成要放入模型的編號(從0開始) 例如: 403410022 -> 235
        self.stu_id2index = {name: idx for idx,
                             name in enumerate(self.stu_ids)}

        # 學生特徵矩陣 [15726, 406] [人數, 課程數]
        self.stu_c_features = self.generate_stu_course_features()  # course
        # 學生特徵矩陣 [15726, 14] [人數, 課程圍度(dim)數]
        self.stu_cc_features = self.generate_stu_course_category_features()  # course category
        # 計算每個學生與每個學生間的相似度 (課程圍度)
        self.similarity = self.cosine_similarity
        self.stu_cc_similarity = self.similarity(
            self.stu_cc_features, self.stu_cc_features.T)
        # 計算每個學生與每個學生間的相似度 (課程數)
        self.stu_c_similarity = self.similarity(
            self.stu_c_features, self.stu_c_features.T)
        # 以 concat 方式結合兩種 feature
        cat_features = torch.cat(
            (self.stu_c_features, self.stu_cc_features), dim=-1)
        self.stu_cat_similarity = self.similarity(cat_features, cat_features.T)

    # 產生學生特徵矩陣
    def generate_stu_course_features(self):
        users_courses = torch.zeros(
            (len(self.stu_ids), len(self.course_names)))
        print(f'similarity_matrix_shape: {users_courses.shape}')
        for course, val in self.course2stu.items():
            course_idx = self.course_name2index[course]
            for stuid in val['students']:
                users_courses[self.stu_id2index[stuid]][course_idx] = 1.
        return users_courses

    # 產生學生特徵矩陣
    def generate_stu_course_category_features(self):
        users_courses = torch.zeros((len(self.stu_ids), len(self.dim2index)))
        print(f'similarity_matrix_shape: {users_courses.shape}')
        for course, val in self.course2stu.items():
            course_cat = self.course2dim.get(course, '0')
            course_cat_id = self.dim2index.get(course_cat, len(course_cat))
            for stuid in val['students']:
                users_courses[self.stu_id2index[stuid]][course_cat_id] = 1.
        return users_courses

    # 查詢某學號學生的 feature
    def get_student_feature(self, stu_id: str, method='course_category'):
        idx = self.stu_id2index.get(stu_id, 0)
        if method == 'course_category':
            return self.stu_cc_features[idx]

    def cosine_similarity(self, a: torch.tensor, b: torch.tensor, eps: float = 1e-8):
        a_n, b_n = a.norm(dim=-1)[:, None], b.norm(dim=-1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm)
        return sim_mt

    # 拆解資料
    def __preproess(self, stu2course: dict):
        # {course_id: {'name': xxx, 'student': [student_id, ...]}}
        course2stu = {}
        stu_property = {}  # {student_id: {grduate_year, department, academy}}
        course2dim = {}
        dim2index = {}
        for stu_id, stu in stu2course.items():
            for c in stu['course']:
                # if c['ID'] not in self.course_id2category:
                #     continue
                course2dim[c['ID']] = c['dim']
                if c['ID'] in course2stu:
                    course2stu[c['ID']]['students'].append(stu['studentID'])
                else:
                    course2stu[c['ID']] = {'name': c['name'],
                                           'students': [stu['studentID']]}

            stu_property[stu['studentID']] = stu['property']
        dim2index = {d: i for i, d in enumerate(
            sorted(set(course2dim.values())))}
        return course2stu, stu_property, course2dim, dim2index

    # 找選課相似的學生的學號
    def get_related_students_id(self, stu_id: str, topn: int = 10, score=False, sim_data='course', p=0.5):
        """[summary]

        Args:
            stu_id (str): student id
            topn (int, optional): topN similarity. Defaults to 10.
            score (bool, optional): return score or not. Defaults to False.
            sim_data (str, optional): similarity data, `category` `course` `weight` or `concat`. Defaults to 'course'.

        Raises:
            KeyError: [description]

        Returns:
            [type]: [description]
        """
        try:
            index = self.stu_id2index[stu_id]
        except KeyError:
            raise KeyError("can't find student id in course data.")

        if sim_data == 'course':
            sim = self.stu_c_similarity[index]
        elif sim_data == 'category':
            sim = self.stu_cc_similarity[index]
        elif sim_data == 'concat':
            sim = self.stu_cat_similarity[index]
        elif sim_data == 'weight':
            sim_c = self.stu_cc_similarity[index]
            sim_cc = self.stu_c_similarity[index]
            sim = (1 - p) * sim_c + p * sim_cc
        else:
            raise ValueError('sim_data argument is error')
        val, related_index = torch.sort(sim, dim=-1, descending=True)
        result = [self.stu_ids[idx]
                  for idx in related_index[:topn + 1] if idx != index]

        if score:
            return result, val[val != 1.]
        else:
            return result

    # 取得某學生選的課
    def get_student_courses(self, stu_id: str):
        try:
            courses = [c for c in self.stu2course[stu_id]['course']]
        except KeyError:
            raise KeyError("can't find student id in course data.")
        return courses


if __name__ == '__main__':
    crl = RelatedStudentExtractor(
        'data/course_train.json', 'data/course_category.csv')
    related_stu_ids, val = crl.get_related_students_id(
        '400110002', score=True, sim_data='category')

    courses_list = crl.get_student_courses('400110002')
    tmp = ' '.join([c['name'] for c in courses_list])
    print(f"{400110002}: {tmp}")

    for stu_id, v in zip(related_stu_ids, val):
        courses_list = crl.get_student_courses(stu_id)
        tmp = ' '.join([c['name'] for c in courses_list])
        print(f"{stu_id} {v}: {tmp}")
