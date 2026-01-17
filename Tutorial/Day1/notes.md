# NumPy Essentials

## 1) What is NumPy
- **Definition:** Numerical computing library providing `ndarray` (N-dimensional array), vectorized operations (*ufuncs*), broadcasting, linear algebra, random sampling, FFT, and more.
- **Design:** Built for speed and memory efficiency via contiguous storage and C/Fortran backends.

## 2) NumPy Importance
- **Speed:** Vectorized operations avoid Python loops; heavy work runs in optimized native code.
- **Memory efficiency:** Homogeneous `dtype` and contiguous layout reduce overhead.
- **Ecosystem:** Integrates with SciPy, Pandas, scikit-learn, Matplotlib.
- **Expressiveness:** Broadcasting and ufuncs simplify complex numeric tasks.

## 3) NumPy Properties
- **Homogeneous data:** Single `dtype` per array (mixed inputs get coerced).
- **Fixed size:** Shape doesn’t change in-place; most ops return new arrays.
- **Core attributes:** `arr.ndim` (dimensions), `arr.shape` (per-axis lengths), `arr.size` (total elements), `arr.dtype` (data type).

## 4) What is Python List
- **Definition:** Built-in ordered collection that can hold mixed types (int, str, float, etc.).
- **Mutability:** Resizable and supports insertions/deletions and structural changes.

## 5) Importance of Python List
- **Flexibility:** Ideal for mixed data and dynamic structures.
- **Convenience:** Easy literals, slicing, and comprehensions.
- **General-purpose:** Great for non-numeric or heterogeneous workflows.

## 6) Difference Between List and NumPy Array
| Aspect | Python List | NumPy Array |
|---|---|---|
| Heterogeneity | Mixed types allowed | Single `dtype` |
| `+` behavior | Concatenates lists | Element-wise addition (with broadcasting) |
| Element-wise ops | Use loops/comprehensions | Vectorized ufuncs (`+`, `*`, etc.) |
| Performance | Slower for numeric compute | Faster; contiguous memory and native backends |
| Memory layout | References to Python objects | Contiguous typed buffer |
| Resizing | Easy in-place (`append`, `pop`) | Fixed size; “resize” creates new array |
| Slicing | Produces a new list (copy) | Returns a view when possible (no copy) |
| Broadcasting | Not supported | Supported across compatible shapes |

## 7) Creation of NumPy Array
```python
import numpy as np

np.array([1, 2, 3])                   # from list
np.array([1, 2, 3], dtype=np.float32)  # specify dtype
np.zeros((2, 3))                       # zeros
np.ones(5)                             # ones
np.arange(0, 10, 2)                    # sequence with step
np.linspace(0, 1, 5)                   # 5 points between 0 and 1
```

## 8) %timeit — What is the use of this
- **Purpose:** Jupyter/IPython magic that measures average execution time of a statement.
```python
# In a notebook cell
import numpy as np
lst = list(range(1_000_000))
arr = np.array(lst)

%timeit [x * 2 for x in lst]      # Python loop/comprehension
%timeit arr * 2                    # Vectorized NumPy operation
```

## 9) What is range()
- **Definition:** Python built-in that returns a *lazy* sequence of integers.
```python
list(range(5))          # [0, 1, 2, 3, 4]
list(range(2, 8, 2))    # [2, 4, 6]
```

## 10) What is arange()
- **Definition:** NumPy sequence generator that returns an `ndarray`.
- **Note:** Supports integer and float steps; *be cautious with floating-point rounding*.
```python
np.arange(5)            # array([0, 1, 2, 3, 4])
np.arange(2, 8, 2)      # array([2, 4, 6])
np.arange(0.0, 1.0, 0.2)# array([0. , 0.2, 0.4, 0.6, 0.8])
```

## 11) Difference Between range and arange
| Aspect | `range` | `np.arange` |
|---|---|---|
| Type returned | Python range object (lazy) | NumPy `ndarray` |
| Element types | Integers only | Integers or floats (depends on inputs/dtype) |
| Typical use | Iteration/counting in Python loops | Numeric arrays for computation |
| Precision | Exact steps for ints | Float steps can accumulate rounding; use `np.linspace` for evenly spaced floats |
| Memory | Lazy, minimal memory | Materialized array in memory |

## 12) Operations on NumPy Array vs List
```python
import numpy as np

arr = np.array([1, 2, 3])
lst = [1, 2, 3]

arr + 2       # array([3, 4, 5])  # vectorized element-wise
[x + 2 for x in lst]  # [3, 4, 5]  # explicit loop/comprehension

arr + np.array([10, 20, 30])  # array([11, 22, 33])  # broadcasting
lst + [10, 20, 30]            # [1, 2, 3, 10, 20, 30] # concatenation, not element-wise
```

## 13) What is Dimension, Shape, Size
- **Dimension (`ndim`):** Number of axes (1D, 2D, 3D, …).
- **Shape (`shape`):** Tuple of lengths along each axis.
- **Size (`size`):** Total number of elements.

## 14) How Dimension, Shape and Size is used
- **Indexing/slicing:** Depend on dimensions and shape.
- **Reshaping:** Change view/layout while preserving `size` when possible.
- **Validation:** Broadcasting and matrix multiplication require compatible shapes.
```python
arr = np.arange(6)          # shape (6,), ndim=1, size=6
arr2 = arr.reshape(2, 3)    # shape (2, 3), ndim=2, size=6
```

## 15) Difference Between Shape, Size and Dimension
| Attribute | Meaning | Example on `arr = np.arange(6).reshape(2, 3)` |
|---|---|---|
| `ndim` | Number of axes | `2` |
| `shape` | Tuple of lengths per axis | `(2, 3)` |
| `size` | Total elements (product of shape) | `6` |

## 16) How to find Shape, Size and Dimension
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.ndim   # 2
arr.shape  # (2, 3)
arr.size   # 6
```

## 17) Type Conversion in NumPy
- **Create with dtype:** `np.array([1, 2, 3], dtype=np.int32)`.
- **Promotion:** Mixed inputs promote to a common dtype.
- **Cast later:** Use `astype()` (returns a copy with new dtype).
```python
np.array([1, 2.5, True]).dtype  # float64 (promotion)
np.array([1.7, 2.2]).astype(np.int32)  # array([1, 2])
```

## 18) What is dtype and how it is used; use cases
- **`dtype`:** Attribute describing element type (e.g., `int32`, `float64`, `bool`).
- **Use cases:** Control memory (e.g., `float32`), interoperability, numeric precision, serialization.
```python
arr = np.array([1, 2, 3], dtype=np.int16)
arr.dtype  # dtype('int16')
```

## 19) What is astype and how it is used; use cases
- **Note:** NumPy does not have `stype`; it’s `astype()`.
- **`astype(dtype)`:** Converts and returns a new array with the target dtype.
- **Use cases:** Cast floats→ints, bytes↔strings, booleans↔ints, downcast to save memory.
```python
arr = np.array([0.9, 1.9, 2.1])
arr.astype(np.int32)  # array([0, 1, 2])
```

## 20) Applications of astype and dtype
- **Precision control:** `float32` vs `float64`.
- **Index safety:** Ensure integer indices (e.g., `int64` on 64-bit systems).
- **ML/IO prep:** `float32` tensors, `uint8` images, CSV strings.
- **Compatibility:** Match external libraries’ expected dtypes.

## 21) Difference Between dtype and astype
| Aspect | `dtype` | `astype` |
|---|---|---|
| What it is | Attribute describing current element type | Method to convert to a different dtype |
| Mutates data? | No, read-only descriptor | Returns a new array (does not change in-place) |
| Example | `arr.dtype` → `dtype('float64')` | `arr.astype(np.float32)` → new array with `float32` |
| Use cases | Inspect type, plan memory/precision | Cast for IO/ML models, precision control, compatibility |

## 22) What is np.round — Basic understanding
- **Alias:** `np.around`.
- **Behavior:** Rounds to nearest value with optional decimals; uses *banker’s rounding* (ties to even).
```python
np.round([1.25, 1.35], 1)  # array([1.2, 1.4])
np.round(1234, -2)         # 1200
```

## 23) What is transpose — Basic understanding
- **Purpose:** Reorders axes; for 2D, swaps rows and columns; for ND, permutes axes.
```python
M = np.array([[1, 2], [3, 4]])
M.T                 # array([[1, 3], [2, 4]])

X = np.arange(24).reshape(2, 3, 4)
np.transpose(X, (1, 0, 2)).shape  # (3, 2, 4)
```