import requests

csv_content = """Hours_Studied,Attendance,Parental_Involvement,Access_to_Resources,Extracurricular_Activities,Sleep_Hours,Previous_Scores,Motivation_Level,Internet_Access,Tutoring_Sessions,Family_Income,Teacher_Quality,School_Type,Peer_Influence,Physical_Activity,Learning_Disabilities,Parental_Education_Level,Distance_from_Home,Gender,external_id,student_name
5,80,High,Medium,Yes,7,80,High,Yes,1,High,High,Public,Positive,3,No,College,Near,Male,u1,User 1
"""
files = {'file': ('test.csv', csv_content, 'text/csv')}
response = requests.post("http://localhost:8000/api/v1/upload", files=files)
print("Upload status:", response.status_code)
print("Response:", response.json())
