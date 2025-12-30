import pandas as pd
from sklearn.preprocessing import LabelEncoder

CSV_PATH = r'C:\Users\poulo\Downloads\archive (2)\hospital data analysis.csv'

def print_encodings(csv_path):
    data = pd.read_csv(csv_path)
    le_gender = LabelEncoder()
    le_condition = LabelEncoder()
    le_procedure = LabelEncoder()
    gender_mapping = dict(zip(le_gender.fit(data['Gender']).classes_, le_gender.transform(le_gender.classes_)))
    condition_mapping = dict(zip(le_condition.fit(data['Condition']).classes_, le_condition.transform(le_condition.classes_)))
    procedure_mapping = dict(zip(le_procedure.fit(data['Procedure']).classes_, le_procedure.transform(le_procedure.classes_)))
    print('Gender encoding:', gender_mapping)
    print('Condition encoding:', condition_mapping)
    print('Procedure encoding:', procedure_mapping)

if __name__ == "__main__":
    print_encodings(CSV_PATH)
