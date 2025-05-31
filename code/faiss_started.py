import numpy as np
import faiss    

# 1. 벡터 생성
# 모든 벡터들의 집합 그리고 검색의 대상인 query 벡터들의 집합을 만든다.
# 이 때 벡터의 형태는 기본 리스트나 torch.Tensor이면 안되고, 무조건 np.array여야한다.
d = 64					# dimension of vector
num_total = 10000		# number of total vectors
num_query = 5			# number of query vectors

np.random.seed(1234)             # make reproducible

total_vectors = np.random.random((num_total, d)).astype('float32')
query_vectors = np.random.random((num_query, d)).astype('float32')

# 2. Index (객체) 구축
Index = faiss.IndexFlatL2(d)    # Index를 구축하고, 모든 벡터들을 Index에 집어넣는다. 
print(Index.is_trained)		    # True
Index.add(total_vectors)	    # add 연산
print(Index.ntotal)		        # 10000

# Index에는 여러 유형이 있는데, 가장 간단한 IndexFlatL2 Index를 사용하였다.
# IndexFlatL2는 학습을 하지 않아도 되기 때문에, 학습을 하지 않았어도 is_trained 값이 True로 나온다.

# 3. 검색 수행
# IndexFlatL2에서는 k-nearest-neighbor 검색 방법을 사용하므로 몇개의 유사한 벡터를 가져올 것인지에 대한 값인 k를 정해준다.
# 각 query_vector에 대해 Index로부터 유사한 벡터 k개를 검색한다.
# 이 때 반환 값은 L2 거리 값과 검색된 벡터의 Index에서의 정수 인덱스 값이다. (주의: Index는 faiss Index 객체를 의미하고, indexes는 우리가 흔히 말하는 정수 인덱스를 의미한다.)
k = 3
distances, indexes = Index.search(query_vectors, k)

print(distances)	# num_query x k
# [[5.0900974 5.670616  5.7184644]
#  [6.8587017 6.956442  7.08839  ]
#  [5.2772083 5.4527187 5.7965107]
#  [5.809805  6.0942507 6.1004057]
#  [5.14467   5.5545816 5.785306 ]]
print(indexes)
# [[1204 3271 2568]
#  [8063 2700  919]
#  [3919 8653 4130]
#  [4429  230  317]
#  [9103  199 6044]]

# 검증을 위해 query 벡터들을 넣는 자리에 total_vectors의 앞단 5개 벡터를 넣어보았다.
distances, indexes = Index.search(total_vectors[:5], k)

print(distances)
# [[0.        5.985731  6.0058537]
#  [0.        5.56559   5.769157 ]
#  [0.        5.665924  5.6770835]
#  [0.        5.748039  6.273041 ]
#  [0.        5.45057   5.683684 ]]
print(indexes)
# [[   0 5325 7124]
#  [   1 3549  555]
#  [   2  304 5103]
#  [   3 5425 8762]
#  [   4 8984 8897]]

# total_vectors 전체가 Index에 들어가 있기 때문에, 검색 결과를 보면 각 Top1 인덱스는 자기 자신의 인덱스다.
# 자기 자신과의 거리는 0이므로, 각 Top1 거리 값도 0이다.

