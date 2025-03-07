def kmeans_1d_dp(points, cluster_num):
    # points.sort()  # ensure sorted order
    n = len(points)
    # Prefix sums for efficient cost calculation
    P = [0] * (n+1)
    Q = [0] * (n+1)
    for i in range(1, n+1):
        P[i] = P[i-1] + points[i-1]
        Q[i] = Q[i-1] + points[i-1]**2

    # Initialize DP and Opt tables
    INF = float('inf')
    DP = [[INF] * (n+1) for _ in range(cluster_num+1)]
    Opt = [[0] * (n+1) for _ in range(cluster_num+1)]
    DP[0][0] = 0
    for i in range(1, n+1):
        DP[0][i] = INF
    # Base case: 1 cluster for first i points
    for i in range(1, n+1):
        # cost of one cluster covering points[0..i-1]
        sum_ = P[i] - P[0]
        sq_sum = Q[i] - Q[0]
        DP[1][i] = sq_sum - (sum_*sum_)/i
        Opt[1][i] = 1  # one cluster always starts at 1
    
    # Helper: cost of segment [j..i] (1-indexed inclusive)
    def cluster_cost(j, i):
        sum_ = P[i] - P[j-1]
        sq_sum = Q[i] - Q[j-1]
        length = i - j + 1
        return sq_sum - (sum_*sum_)/length

    # Compute DP rows 2..k using divide-and-conquer
    def compute_dp(m, i_left, i_right, opt_left, opt_right):
        if i_left > i_right:
            return
        mid = (i_left + i_right) // 2
        best_split = None
        best_cost = INF
        # Search for optimal split index j in [opt_left, opt_right]
        for j in range(opt_left, min(opt_right, mid-1) + 1):
            cost = DP[m-1][j] + cluster_cost(j+1, mid)
            if cost < best_cost:
                best_cost = cost
                best_split = j
        DP[m][mid] = best_cost
        Opt[m][mid] = best_split + 1  # cluster m starts at index best_split+1
        # Recurse on left and right halves with updated bounds
        compute_dp(m, i_left, mid-1, opt_left, best_split)
        compute_dp(m, mid+1, i_right, best_split, opt_right)

    # Fill DP for m = 2..cluster_num
    for m in range(2, cluster_num+1):
        # each cluster must have at least 1 point, so lowest i for m clusters is i=m
        compute_dp(m, i_left=m, i_right=n, opt_left=m-1, opt_right=n-1)

    # Reconstruct clustering from Opt table
    clusters = []
    m = cluster_num
    i = n
    # Backtrack from Opt[k][n] down to Opt[1][...]
    while m > 0:
        start_index = Opt[m][i]   # start index of cluster m (1-indexed)
        clusters.append(points[start_index-1 : i])  # segment [start_index..i] -> slice in 0-index
        i = start_index - 1      # move to end of previous cluster
        m -= 1
    clusters.reverse()  # reverse to get clusters in original order
    return clusters