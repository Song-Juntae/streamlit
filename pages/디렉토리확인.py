import os
import psutil

path = os.getcwd()

path

file_list = os.listdir(path)

file_list

file_list = os.listdir('/')

file_list

# 메모리 = psutil.virtual_memory()
# 메모리
# 사용가능한 = 메모리.used / 1024**3
# 사용가능한

# # 파일 삭제 코드
# 제거할파일리스트 = []
# for i in 제거할파일리스트:
#     os.remove(i)