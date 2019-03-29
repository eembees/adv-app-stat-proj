# notes while analyzing output.json
## Removing data points 

calling `df.groupby('fac').size()` returned
```
fac
HUM        12842
JUR         1403
SAMF        4411
SCIENCE     7656
SUND        3246
TEO          699
dtype: int64
```
and after removing pass/fail courses and courses with no statistics (which are a bit difficult to work with), using `df.dropna(subset=['g_10'], inplace=True)`
```
fac
HUM        5274
JUR         981
SAMF       2379
SCIENCE    4961
SUND       1871
TEO         296
```
so I change TEO to HUM and JUR to SAMF, because the course catalog is significantly smaller than what we're looking for, and we can't predict anything useful from 200 datapoints.
In the end, I get this: 
```
fac
HUM        5570
SAMF       3360
SCIENCE    4961
SUND       1871
dtype: int64
```