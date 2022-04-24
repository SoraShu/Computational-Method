# 高斯(Gauss)列主元消去法

## 问题分析

### 实验要求

编写代码实现高斯(Gauss)列主元消去法，用计算机计算线性方程组的解。

### 算法背景

高斯(Gauss)列主元消去法源自高斯消元法。高斯消元法是一种直接解法，只包含舍入误差。若线性方程组的为 n 元方程组，利用高斯消元法求解线性方程组的计算次数约为 $\frac{1}{3}n^3$ （舍去低次项）。高斯列主元消去法在高斯消元法基础上改进，目的是减少舍入误差。

### 算法特点

高斯消元法只包含有限次的四则运算，高斯(Gauss)列主元消去法在其基础上增加了行交换的操作。相比于直接求逆矩阵，高斯(Gauss)列主元消去法计算效率更高。

### 待求解问题

给定 $n$ 阶线性方程组 $\boldsymbol{Ax}=$$\boldsymbol{b}$ ，求 $\boldsymbol{x}$。

## 数学原理

高斯（Gauss）列主元消去法：对给定的 $n$ 阶线性方程组 $\boldsymbol{Ax}=$$\boldsymbol{b}$ ，首先进行列主元消元过程，然后进行回代过程，最后得到解或确定该线性方程组是奇异的。如果系数矩阵的元素按绝对值在数量级方面相差很大，那么，在进行列主元消元过程前，先把系数矩阵的元素进行行平衡：系数矩阵的每行元素和相应的右端向量元素同除以该行元素绝对值最大的元素。这就是所谓的平衡技术。然后再进行列主元消元过程。

## 程序设计流程

```python
def Gauss_method(
    A: ndarray,
    b: ndarray
) -> list[Optional[tuple[ndarray, ndarray]]]:
    n, o1 = A.shape
    o2, o3 = b.shape
    if(n != o1 or n != o2 or o3 != 1):
        raise ValueError  # 维度不符合要求，抛出异常
    m = np.hstack([A, b])
    for k in range(n-1):  # k is col
        p = m[k:, k].argmax()+k
        if m[p, k] == 0:
            return None
        if p != k:  # swap row p and k
            m[[p, k], :] = m[[k, p], :]
        for i in range(k+1, n):
            # row i -=([i,k]/[k,k])*row k
            t = m[i, k]/m[k, k]
            m[i, :] -= t*m[k, :]
    if m[n-1, n-1] == 0:
        return None
    x = np.zeros(n, dtype=float)
    x[n-1] = m[n-1, n]/m[n-1, n-1]  # b=m[:,n]
    for k in reversed(range(n-1)):
        temp = m[k, n]
        for j in range(k+1, n):
            temp -= m[k, j]*x[j]
        x[k] = temp/m[k, k]
    return (x.reshape([n, 1]), m)
```

## 实验结果、结论与讨论

$\boldsymbol{Ax}=$$\boldsymbol{b}$ ，求 $\boldsymbol{x}$。

### 问题 1

#### (1)

$$
\boldsymbol{A}=\begin{bmatrix}
0.4096 & 0.1234 & 0.3678 & 0.2943 \\
0.2246 & 0.3872 & 0.4015 & 0.1129 \\
0.3645 & 0.1920 & 0.3781 & 0.0643 \\
0.1784 & 0.4002 & 0.2786 & 0.3927 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
1.1951 \\
1.1262 \\
0.9989 \\
1.2499 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
1.00000000 \\
1.00000000 \\
1.00000000 \\
1.00000000 \\
\end{bmatrix}
$$

#### (2)

$$
\boldsymbol{A}=\begin{bmatrix}
136.01 &  90.86 &   0 &   0 \\
 90.86 &  98.81 & -67.59 &   0 \\
  0 & -67.59 & 132.01 &  46.26 \\
  0 &   0 &  46.26 & 177.17 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
226.87 \\
122.08 \\
110.68 \\
223.43 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
1.00000000 \\
1.00000000 \\
1.00000000 \\
1.00000000 \\
\end{bmatrix}
$$

#### (3)

$$
\boldsymbol{A}=\begin{bmatrix}
1   & 1/2 & 1/3 & 1/4 \\
1/2 & 1/3 & 1/4 & 1/5 \\
1/3 & 1/4 & 1/5 & 1/6 \\
1/4 & 1/5 & 1/6 & 1/7 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
25/12 \\
77/60 \\
57/60 \\
319/420 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
1.00000000 \\
1.00000000 \\
1.00000000 \\
1.00000000 \\
\end{bmatrix}
$$

#### (4)

$$
\boldsymbol{A}=\begin{bmatrix}
10 & 7 &  8 &  7 \\
 7 & 5 &  6 &  5 \\
 8 & 6 & 10 &  9 \\
 7 & 5 &  9 & 10 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
32 \\
23 \\
33 \\
31 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
1.00000000 \\
1.00000000 \\
1.00000000 \\
1.00000000 \\
\end{bmatrix}
$$

### 问题 2

#### (1)

$$
\boldsymbol{A}=\begin{bmatrix}
197    & 305  & -206    & -804   \\
 46.8  &  71.3 &  -47.4  &   52   \\
 88.6  &  76.4 &  -10.8  &  802   \\
  1.45 &   5.9 &    6.13 &   36.5 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
136   \\
 11.7 \\
 25.1 \\
  6.6 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
0.95367911 \\
0.32095685 \\
1.07870808 \\
-0.09010851 \\
\end{bmatrix}
$$

#### (2)

$$
\boldsymbol{A}=\begin{bmatrix}
0.5398 &  0.7161 & -0.5554 & -0.2982 \\
0.5257 &  0.6924 &  0.3565 & -0.6255 \\
0.6465 & -0.8187 & -0.1872 &  0.1291 \\
0.5814 &  0.9400 & -0.7779 & -0.4042 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
 0.2058 \\
-0.0503 \\
 0.1070 \\
 0.1859 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
0.51617730 \\
0.41521947 \\
0.10996610 \\
1.03653922 \\
\end{bmatrix}
$$

#### (3)

$$
\boldsymbol{A}=\begin{bmatrix}
10 &  1 & 2 \\
 1 & 10 & 2 \\
 1 &  1 & 5 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
13 \\
13 \\
 7 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
1.00000000 \\
1.00000000 \\
1.00000000 \\
\end{bmatrix}
$$

$$
\boldsymbol{A}=\begin{bmatrix}
 4 & -2 & -4 \\
-2 & 17 & 10 \\
-4 & 10 &  9 \\
\end{bmatrix}
,
\boldsymbol{b}=\begin{bmatrix}
-2 \\
25 \\
15 \\
\end{bmatrix}
$$

$$
\boldsymbol{x}=\begin{bmatrix}
1.00000000 \\
1.00000000 \\
1.00000000 \\
\end{bmatrix}
$$

### 结论

其实此问题亦能化为已知 $\boldsymbol{A},\boldsymbol{b}$ ，求 $\boldsymbol{A}^{-1}\boldsymbol{b}$ 。若先求 $\boldsymbol{A}^{-1}$ 后与 $\boldsymbol{b}$ 相乘，相比于高斯消元法计算次数要多。
