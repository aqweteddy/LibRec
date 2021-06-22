from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import json
import os


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

parser = ArgumentParser()
parser.add_argument('--input', type=str, help='dir')
parser.add_argument('--output_json', type=str, help='filename')

args = parser.parse_args()
df = pd.concat([pd.read_csv(os.path.join(args.input, y), encoding='big5') 
                for y in find_csv_filenames(args.input)])
    
df = df.drop('畢業學期',axis=1)
df.columns = ['leaveSchool', 'studentID', 'department', 'graduateYear', 
              'academy', 'courseID', 'courseName', 'CourseCategory']
result = []
df = df[df['CourseCategory'] == '通識']

for student_id, d in tqdm(df.groupby('studentID')):
    stu_property = {'leaveSchool': d.get('leaveSchool').iloc[0], 
                    'department': d.get('department').iloc[0], 
                    'graduateYear': int(d.get('graduateYear').iloc[0]), 
                    'academy': d.get('academy').iloc[0], 
                   }
    course_list = [{'category': c, 'ID': str(idx), 'name': name} 
                       for c, idx, name in zip(d['CourseCategory'], d['courseID'], d['courseName'])]
    dct = {'studentID': str(student_id), 'property': stu_property, 'course': course_list}
    result.append(dct)

with open(args.output_json, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
