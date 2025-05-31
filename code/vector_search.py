import faiss
import numpy as np

# 128차원 랜덤 벡터 생성
d = 128
nb = 10000
np.random.seed(1234)
data = np.random.random((nb, d)).astype('float32')

# FAISS 인덱스 생성 및 데이터 추가
index = faiss.IndexFlatL2(d)  # L2 거리 기반 인덱스
index.add(data)

# 검색 수행
query = np.random.random((1, d)).astype('float32')
D, I = index.search(query, k=5)  # 가장 가까운 5개 벡터 검색
print(I)  # 검색된 벡터의 인덱스 출력

