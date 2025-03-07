# Homework 4 Bit Battle Leaderboard

## KMeans Performance Results

*Last updated: 2025-03-07 09:49:38 EST*

| Rank | Runtime (seconds) |
|------|------------------|
| 1 | 8.5030 |
| 2 | 104.5437 |
| 3 | 114.7586 |
| 4 | 142.8440 |
| 5 | 181.6587 |
| 6 | 219.7886 |

# KMeans Algorithm

The k-Means clustering problem on n points is NP-Hard for any dimension d ≥ 2, however, for the 1D case
there exists exact polynomial time algorithms. The best known algorithm has a runtime of $n2^{
O(\sqrt{\lg \lg n \lg k})}$.
For the details of all kinds of methods to solve the 1d kmeans problem, please refer to  [Grønlund et al. (2017)](https://arxiv.org/abs/1701.07204).

Our homework problem changes the objective function of vanilla kmeans so we write the solution of our own problem below.

## Dynamic Programming Formulation

We formulate the problem as a DP where the subproblem DP[i][m] represents the minimum k-means cost for clustering the first i points into m clusters. The transition considers where the last cluster (the m-th cluster) starts. If the m-th cluster covers points from index j to i (inclusive) – meaning the first m−1 clusters cover points 1 to j−1 – then the recurrence is:

$$
DP[i][m] = \min_{m \leq j \leq i} (DP[j-1][m-1] + d(x_j,\ldots,x_i))
$$

where $d(x_j,\ldots,x_i)$ is the weighted cost (sum of squared distances) of assigning points $x_j \ldots x_i$ to a single cluster. The base cases are $DP[0][0]=0$ (zero points in zero clusters costs 0) and $DP[i][0]=\infty$ for $i>0$ (cannot cluster >0 points into 0 clusters). Likewise, $DP[0][m]=\infty$ for $m>0$. Ultimately, $DP[n][k]$ (with n = number of points) gives the minimum cost for k clusters.

### Contiguous clusters
Because the input points are sorted, the optimal clustering will group contiguous points. This is why we restrict that the last cluster spans a contiguous interval $j\ldots i$. Contiguous clusters simplify cost computation and reduce the search space drastically (no need to consider non-contiguous groupings).

### Efficient Cost Computation for a Cluster

The cost $d(x_j,\ldots,x_i)$ is defined as the weighted sum of squared distances of points $x_j$ through $x_i$ from their mean. We can compute this in O(1) time with precomputed prefix sums:

Let $P[t]=\sum_{r=1}^t x_r$ (prefix sum up to index t) and $Q[t]=\sum_{r=1}^t x_r^2$ (prefix sum of squares).

For any segment $j\ldots i$, the sum is $S=P[i]-P[j-1]$ and the sum of squares is $Q[i]-Q[j-1]$. The number of points in the segment is $n_{seg}=i-j+1$.

The mean of points $x_j\ldots x_i$ is $\mu=S/n_{seg}$. The sum of squared errors (SSE) for this segment is:

$$
 \sum_{r=j}^i (x_r-\mu)^2 = (Q[i]-Q[j-1])-\frac{S^2}{n_{seg}}
$$

This formula computes the within-cluster SSE in constant time from the prefix sums. We can prepare the prefix arrays in O(n) time. Using such O(1) cost lookup, a direct DP implementation would take O(n²·k) time in the worst case, because for each state $DP[i][m]$ we might scan a range of j. We need to reduce this using a smarter optimization.

### Divide-and-Conquer Optimization (Monotonic DP)

The key to optimizing the DP is to exploit the monotonicity of the optimal partition point as i grows. Intuitively, if $opt(m,i)$ is the index j that gives the minimum for $DP[i][m]$ (i.e. the optimal start of the last cluster for i points and m clusters), it turns out that $opt(m,i)$ will non-decrease as i increases under this cost function. In other words, the best split point for a larger index i is at least as far right as the best split for a smaller index.


This property holds here because the cost function (SSE) satisfies the quadrangle inequality (a form of convexity on contiguous segments).

How this helps: If $opt(m,i_0)=j_0$, then when computing $DP[i][m]$ for any $i>i_0$, we only need to consider split positions $j\geq j_0$. We can prune transitions that go left of the previous optimum.

This monotonicity enables using a divide-and-conquer DP approach (also known as Knuth optimization in DP) to compute each row of the DP table in linear time. Instead of trying every j for every i naively, we do a recursive divide-and-conquer:

1. To compute $DP[*][m]$ (the m-th cluster row for all i from 1..n), solve for a midpoint i = mid (e.g. n/2), find the optimal j for this mid by checking only a limited range $[j_{min},j_{max}]$ (based on known bounds from nearby indices).

2. Once $opt(m,mid)$ is found, use it to narrow down the search space for the left half (i < mid) and right half (i > mid). Recursively compute those halves, carrying the constrained j-range for each.

By always narrowing the j-range using the monotonicity, we ensure each potential split index is considered only a constant number of times across the entire row computation. This yields roughly O(n) work for each of the k clusters (rather than O(n²)), resulting in about O(n·k) time overall (with at most a very small additional log factor in the recursion). 

More reading materials for DnC DP can be found [here](https://cp-algorithms.com/dynamic_programming/divide-and-conquer-dp.html) and [here](https://cp-algorithms.com/dynamic_programming/knuth-optimization.html)

### Algorithm Outline

Using the above ideas, we can implement the function kmeans_1d(points, cluster_num) as follows:

1. **Preprocessing**: Sort the input list points (if not already sorted) in non-decreasing order. Compute prefix sums P and prefix squared-sums Q for the array. These allow O(1) cost calculations for any segment's SSE cost.

2. **Initialize DP tables**: Create a DP table DP[m][i] of size (k+1) × (n+1) to store minimum costs (or use two rows that get reused for memory efficiency). Also prepare a table Opt[m][i] to store the argmin (optimal split index) that gave DP[m][i]. We have DP[0][0] = 0 and DP[0][i] = ∞ for i>0. For convenience, use 1-indexing for points (i.e., points 1..n) in the DP.

3. **Fill DP for 1 cluster**: For m = 1, the optimal clustering is just one cluster containing all points up to i. So:
   DP[1][i] = d(x_1,...,x_i) for each i = 1..n, using the prefix formula for cost. The Opt[1][i] for all i is obviously 1 (the cluster starts at the first point).

4. **Dynamic programming for m = 2 to k clusters**: For each m from 2 to k, compute DP[m][i] for i = m..n. Use the divide-and-conquer recursion to efficiently find the optimal split for each i:
   - Define a recursive function compute(m, i_left, i_right, opt_left, opt_right) that computes DP[m][i] for i in the range [i_left, i_right], assuming the optimal split j for those i will lie in [opt_left, opt_right].
   - If i_left > i_right, return (no points to process). Otherwise, select a mid index: mid = floor((i_left + i_right)/2).
   - Search for the best split index best_j in the range [opt_left, min(opt_right, mid-1)] that minimizes DP[m-1][j-1] + d(x_j,...,x_mid). (Here j is the starting index of cluster m, so the previous cluster ends at j-1.) Compute the cost in this range and find the best_j that yields minimum cost.
   - Set DP[m][mid] to that minimum cost and record Opt[m][mid] = best_j.
   - Recurse on the left half: compute(m, i_left, mid-1, opt_left, best_j) and on the right half: compute(m, mid+1, i_right, best_j, opt_right). This recursion exploits the monotonicity by narrowing the search bounds for j.
   - Initially, call compute(m, m, n, opt_left = m, opt_right = n) for each m. (The smallest j we can have is m because we need at least one point per cluster.)

5. **Reconstruct Clusters**: After filling the DP table for m = k and i = n (which gives the minimum cost), backtrack using the Opt table to get cluster boundaries. Start from i = n and m = k: let j = Opt[k][n]. This means the k-th cluster runs from point j to n. Then set i = j-1 and m = k-1, and repeat: for that state, j = Opt[m][i] gives the start of the m-th cluster. Continue until m = 1. This yields the cluster boundary indices in reverse order. Finally, construct the list of clusters by slicing the original points list according to these boundaries.
