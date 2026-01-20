NumPy Day 2: Matrices, Aggregations, Logical Ops (Theory + Practice)

## 1. Matrix (2D Array)
### 1.1 Concepts
A matrix in NumPy is a 2D ndarray with shape `(rows, cols)` stored contiguously in memory (row-major by default). This layout favors row-wise access for speed. Matrices underpin linear algebra operations, tabular datasets, and image representations (height × width × channels). Structured variants such as identity (`np.eye`), diagonal (`np.diag`), and triangular (`np.triu`, `np.tril`) matrices are common.

### 1.2 Construction and Inspection
```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])        # from lists
Z = np.zeros((2, 3))                         # zeros
O = np.ones((3, 3), dtype=float)             # ones
I = np.eye(4)                                # identity
seq = np.arange(1, 13).reshape(3, 4)         # sequential
rand = np.random.randint(0, 10, (3, 3))      # random ints

rows, cols = A.shape
print(A.shape, A.ndim, A.size, A.dtype, A.nbytes)
```

### 1.3 Transpose
Transposition swaps rows and columns using `A.T` or `np.transpose(A)`. It is typically a view, so it is inexpensive. One-dimensional vectors remain one-dimensional after `.T`; use `A[:, None]` or `A.reshape(-1, 1)` to obtain an explicit column vector when needed.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
AT = A.T
# Real-world: switch Products×Months to Months×Products
sales = np.array([[100, 150, 200], [80, 90, 110], [120, 130, 140]])
sales_by_month = sales.T
```

### 1.4 Slicing in Depth
Slicing syntax follows `A[row_start:row_end:step, col_start:col_end:step]`. Negative indices count from the end, and omitted bounds imply the start or end. Slices produce views (no copy) unless advanced indexing intervenes.

```python
M = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

row1 = M[1, :]
col2 = M[:, 2]
block = M[0:2, 1:3]
step = M[::2, ::2]
rev_rows = M[::-1, :]
rev_cols = M[:, ::-1]
```

Practical: image crops and channel extraction
```python
img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
crop = img[100:300, 200:500, :]
red = img[:, :, 0]
flip_h = img[:, ::-1, :]
```

### 1.5 Reshape
Reshape changes the view of data without altering values, provided the total element count is preserved. The dimension `-1` lets NumPy infer the appropriate size. Reshape returns a view when possible.

```python
v = np.arange(12)
resh2x6 = v.reshape(2, 6)
resh3x4 = v.reshape(3, 4)
resh_auto = v.reshape(3, -1)   # -> (3, 4)

# Flattening options
flat_copy = v.reshape(-1).copy()
flat_view = v.ravel()
```

Practical: ML input preparation
```python
images = np.random.rand(1000, 28, 28)
X = images.reshape(1000, -1)   # (1000, 784) for models
```

## 2. Aggregation Functions
Aggregations compress arrays into summary values along specified axes, enabling quick statistical insights.

### 2.1 Core Statistics
```python
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

np.sum(data)
np.sum(data, axis=0)
np.mean(data, axis=1)
np.median(data)
np.std(data), np.var(data)
np.min(data), np.max(data)
np.argmin(data), np.argmax(data)
np.ptp(data)
```

### 2.2 Cumulative and Order Statistics
```python
arr = np.array([1, 2, 3, 4])
np.cumsum(arr)
np.cumprod(arr)
np.percentile(arr, 75)
np.quantile(arr, 0.25)
```

### 2.3 Practical Applications
Sales analysis benefits from totals per product (axis 1), best month detection (axis 0), and growth rates. Quality control monitors drift via mean and variance, while finance frequently combines rolling windows (with `np.lib.stride_tricks.sliding_window_view`) and aggregations for indicators.

```python
sales = np.array([[1000, 1200, 1100, 1300],
                  [800, 900, 850, 920],
                  [1500, 1600, 1550, 1700]])
total_per_product = sales.sum(axis=1)
avg_per_month = sales.mean(axis=0)
best_month = avg_per_month.argmax()
```

## 3. Logical Operations
Logical operations run element-wise and produce boolean arrays that drive filtering, masking, and conditional logic.

### 3.1 Comparisons
```python
x = np.array([1, 2, 3, 4, 5])
x > 3
x <= 2
x == 3
x != 3
```

### 3.2 Boolean Operators
```python
a = np.array([True, True, False])
b = np.array([True, False, True])

np.logical_and(a, b)   # same as a & b
np.logical_or(a, b)    # same as a | b
np.logical_not(a)      # same as ~a
np.logical_xor(a, b)   # same as a ^ b
```

### 3.3 Combined Conditions and Masking
```python
data = np.array([10, 25, 30, 15, 40, 5])
mask = (data > 10) & (data < 35)
filtered = data[mask]

complex_mask = ((data > 20) & (data < 35)) | (data < 8)
data[complex_mask]
```

Practical anomaly detection
```python
temps = np.array([22, 35, 28, 38, 25, 30, 40, 20])
too_hot = temps > 35
alert_days = temps[too_hot]
```

## 4. np.where(), np.all(), np.any() (Deep Dive)
### 4.1 np.where
`np.where` either returns indices satisfying a condition or branches element-wise between two choices.

```python
arr = np.array([10, 20, 30, 40, 50])
idx = np.where(arr > 25)
values = arr[idx]

labels = np.where(arr > 25, "High", "Low")
masked = np.where(arr > 25, arr, 0)

M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
r, c = np.where(M > 5)
selected = M[r, c]
```

### 4.2 np.all
`np.all` verifies that every element—or every element along a chosen axis—meets a condition.
```python
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.all(M > 0)
np.all(M > 5, axis=1)

data = np.array([1, 2, np.nan, 4])
np.all(~np.isnan(data))
```

### 4.3 np.any
`np.any` confirms whether at least one element or at least one element along an axis—meets a condition.
```python
np.any(M > 8)
np.any(M < 0)
np.any(M == 1, axis=0)
```

### 4.4 Differences and When to Use (Tabular)
| Function | Purpose | What it returns | Typical use | Example call |
|----------|---------|-----------------|-------------|--------------|
| np.where | Locate indices or branch values based on a condition | Indices tuple or element-wise selection result | Select positions or replace values conditionally | `np.where(arr > 0, arr, 0)` |
| np.all   | Require every element (or every element along an axis) to satisfy a condition | Single boolean or boolean array per axis | Validating that an entire row/column passes a rule | `np.all(matrix > 0, axis=1)` |
| np.any   | Check whether at least one element (or one per axis) satisfies a condition | Single boolean or boolean array per axis | Early alert when a threshold is crossed anywhere | `np.any(scores == 100)` |

Practical comparison
```python
scores = np.array([85, 72, 90, 68, 95, 88])
high_idx = np.where(scores > 80)[0]
all_pass = np.all(scores >= 60)
any_perfect = np.any(scores == 100)

adjusted = np.where(scores < 70, scores + 5, scores)
```

## 5. flatten()
`flatten()` produces a one-dimensional copy. `ravel()` and `reshape(-1)` return views when possible. Ordering can be row-major (`'C'`) or column-major (`'F'`).

```python
M = np.array([[1, 2], [3, 4]])
f = M.flatten()
r = M.ravel()
u = M.reshape(-1)

f[0] = 99
r[1] = 77
```

Use cases include creating ML feature vectors from images, computing global statistics without structural constraints, and preparing data for APIs that expect one-dimensional input.

## 6. np.rot90()
`np.rot90` rotates an array by 90-degree increments within a specified plane. Signature: `np.rot90(a, k=1, axes=(0, 1))` where positive `k` rotates counter-clockwise and negative `k` rotates clockwise.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

np.rot90(M, k=1)
np.rot90(M, k=2)
np.rot90(M, k=-1)
```

Practical scenarios include image orientation fixes, board or puzzle rotations, and preparing data for alternative visualization layouts.

Alternatives without `np.rot90`
```python
ccw_90 = np.flipud(M.T)
cw_90 = np.fliplr(M.T)
rot_180 = M[::-1, ::-1]
```

## 7. Fancy Indexing Masking
### 7.1 Boolean Masking
```python
arr = np.array([10, 20, 30, 40, 50])
mask = arr > 25
arr[mask]
arr[mask] = -1
```

2D masking
```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

mask = M > 5
vals = M[mask]
M[mask] = 0
```

### 7.2 Integer Array Indexing
```python
arr = np.array([10, 20, 30, 40, 50])
idx = np.array([0, 2, 4])
arr[idx]

rows = np.array([0, 1, 2])
cols = np.array([0, 1, 2])
diag = M[rows, cols]
```

### 7.3 Combined Masks
```python
data = np.array([15, 8, 22, 35, 12, 40, 5])
filtered = data[(data > 10) & (data < 30)]
```

### 7.4 Practical Scenarios
Data cleaning replaces outliers or invalid markers, stock analysis isolates days with large price moves, and image processing performs color thresholding with channel-wise masks.
```python
temps = np.array([22, 25, -999, 28, 30, -999, 24, 26])
valid = temps > -100
clean = temps[valid]
temps[~valid] = clean.mean()

prices = np.array([100, 102, 98, 105, 103, 99, 107, 110])
changes = np.diff(prices)
big_moves = np.abs(changes) > 3
significant_days = prices[1:][big_moves]
```

## Quick Reference
Transpose uses `A.T`. Slicing follows `A[r0:r1:step, c0:c1:step]`. Reshape with `A.reshape(new_shape)` using `-1` for inference. Core aggregations include `sum`, `mean`, `min` or `max`, `std` or `var`, `percentile`, and `cumsum`. Logical work relies on comparisons plus `logical_and`, `logical_or`, `logical_not`, and `logical_xor`, with mask combination via `&`, `|`, and `~`. Conditional helpers: `np.where`, `np.all`, `np.any`. Flattening: `flatten()` creates a copy; `ravel()` or `reshape(-1)` provide views when possible. Rotation: `np.rot90(a, k, axes)`; alternatives use transpose plus flips. Masking: boolean or integer index arrays select and mutate targeted elements.
