# (TI-)NBC
NBC [[1]](#references) and TI-NBC [[2]](#references) implementations for Data Mining course @ WUT.
 
## Installation
### Install
```bash
make install
```
### Run (cmd)
```bash
nbc --path "comma_separated_dataset.csv"
```
### Run (python)
```python
from nbc import clustering
import numpy as np

vectors = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [10.0, 10.0],
    [11.0, 11.0]
])
k = 1
clusters = clustering.nbc(vectors=vectors, k=k)
```
Output - dictionary (vector id, cluster id - where -1 stands for a noise):
```bash
{0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
```

### Docker
Building docker image
```bash
docker build -t nbc:latest .
```
Example run script (assuming input and output should be in "data" directory)
```bash
bash run_docker.sh
```
### Unit tests
```bash
make test
```

## NBC algorithm 

## TI-NBC algorithm

## References
[1] [Zhou S., Zhao Y., Guan J., Huang J. (2005) A Neighborhood-Based Clustering Algorithm. In: Ho T.B., Cheung D., Liu H. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2005. Lecture Notes in Computer Science, vol 3518. Springer, Berlin, Heidelberg](https://link.springer.com/chapter/10.1007/11430919_43)

[2] [Kryszkiewicz M., Lasek P. (2010) A Neighborhood-Based Clustering by Means of the Triangle Inequality. In: Fyfe C., Tino P., Charles D., Garcia-Osorio C., Yin H. (eds) Intelligent Data Engineering and Automated Learning – IDEAL 2010. IDEAL 2010. Lecture Notes in Computer Science, vol 6283. Springer, Berlin, Heidelberg](https://link.springer.com/chapter/10.1007/978-3-642-15381-5_35)