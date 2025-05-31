# D2506_FAISS
> FAISS(Facebook AI Similarity Search) <br/>
- Facebook AI Research에서 개발한 고차원 벡터 검색 및 유사성 검색을 위한 라이브러리 <br/>
- 주로 대량의 벡터 데이터를 빠르게 검색하는 데 사용되며, 머신러닝 및 딥러닝에서 임베딩 벡터를 활용하는 다양한 응용 분야에서 널리 사용됩니다.

---
### FAISS 주요 특징
- `고속 검색` : 수백만 개 이상의 벡터 데이터에서도 빠른 검색이 가능하도록 최적화되어 있습니다.
- `다양한 인덱싱 방법` : Flat 인덱스, Product Quantization, Hierarchical 인덱스 등 다양한 방식으로 벡터를 저장하고 검색할 수 있습니다.
- `GPU 지원` : 대량의 데이터 검색을 더욱 빠르게 수행할 수 있도록 GPU 가속을 지원합니다.
- `유사성 검색` : 주어진 쿼리 벡터와 가장 유사한 벡터를 찾아내는 기능을 제공합니다.

---
### FAISS 사용법
- FAISS를 사용하려면 먼저 Python 환경에서 faiss 패키지를 설치해야 합니다:

```shell
pip install faiss-cpu  # CPU 버전
pip install faiss-gpu  # GPU 버전
```
<br/>

이후 벡터를 저장하고 검색하는 과정은 다음과 같습니다:

```python 
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

```

