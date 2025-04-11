# 📄 DAA Cheat Sheet – Full Syllabus Coverage

---

## 🟢 1. Algorithm Analysis Basics

- **Big-O Notation**: Upper bound  
- **Big-Ω Notation**: Lower bound  
- **Big-Θ Notation**: Tight bound  

### Master Theorem (for T(n) = aT(n/b) + f(n)):

| Case | Condition                          | Time Complexity            |
|------|------------------------------------|----------------------------|
| 1    | f(n) = O(n^log_b a - ε)            | Θ(n^log_b a)               |
| 2    | f(n) = Θ(n^log_b a)                | Θ(n^log_b a * log n)       |
| 3    | f(n) = Ω(n^log_b a + ε) + regularity | Θ(f(n))                    |

---

## 🟠 2. Greedy Algorithms

### ✅ Steps:
- Sort (by value/weight, finish time, etc.)
- Greedy choice
- Check optimal substructure

### ✅ Key Problems:

- **Fractional Knapsack**:  
  Sort by value/weight → pick max first  
  Time: O(n log n)

- **Job Scheduling**:  
  Sort by profit & deadline → place greedily

- **MST (Prim’s / Kruskal’s)**:  
  - *Prim’s*: Greedy expansion from one vertex  
  - *Kruskal’s*: Sort edges → pick min edge avoiding cycles (Union-Find)

- **Dijkstra’s Algorithm**: For shortest path (no negatives)  
  Time: O((V + E) log V) with priority queue

---

## 🔵 3. Dynamic Programming (DP)

### ✅ Steps:
- Identify subproblems
- Define recursive relation
- Use bottom-up tabulation or memoization

### ✅ Key Problems:

- **LCS (X, Y)**:  
  `dp[i][j] = dp[i−1][j−1]+1` if `X[i]=Y[j]` else `max(dp[i−1][j], dp[i][j−1])`  
  Time: O(mn)

- **Matrix Chain Multiplication**:  
  `dp[i][j] = min(dp[i][k] + dp[k+1][j] + dims[i−1]*dims[k]*dims[j])`  
  Time: O(n^3)

- **0/1 Knapsack**:  
  `dp[i][w] = max(dp[i−1][w], val[i] + dp[i−1][w−wt[i]])`

- **Floyd-Warshall**: All-pairs shortest path  
  `dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])`  
  Time: O(n^3)

- **Bellman-Ford**: Shortest path with negatives  
  Time: O(VE)

---

## 🟡 4. Divide & Conquer

### ✅ Key Problems:

- **Merge Sort** – Time: O(n log n)

- **Quick Sort** – Avg: O(n log n), Worst: O(n^2)

- **Binary Search** – Time: O(log n)

- **Matrix Multiplication (Strassen's)** –  
  T(n) = 7T(n/2) + O(n^2.81)

---

## 🟣 5. String Matching Algorithms

| Algorithm    | Idea                          | Time                        |
|--------------|-------------------------------|-----------------------------|
| Naive        | Slide pattern over text       | O(mn)                       |
| KMP          | Build prefix table to avoid re-check | O(m + n)              |
| Rabin-Karp   | Hash comparison               | O(mn) worst, O(n + m) avg   |
| Boyer-Moore  | Skip based on mismatch        | Best for large patterns     |

---

## 🔴 6. Graph Algorithms

| Algorithm       | Use                          | Time Complexity        |
|-----------------|-------------------------------|------------------------|
| BFS / DFS       | Traversal                     | O(V + E)               |
| Prim’s          | MST                           | O(E log V)             |
| Kruskal’s       | MST                           | O(E log E)             |
| Dijkstra’s      | Shortest path (no negative)   | O((V + E) log V)       |
| Bellman-Ford    | Handles negative edges        | O(VE)                  |
| Floyd-Warshall  | All-pairs shortest            | O(V^3)                 |

---

## ⚫ 7. Complexity & NP Problems

- **P**: Solvable in polynomial time
- **NP**: Verifiable in polynomial time
- **NP-Complete**: In NP, all NP problems reduce to it
- **NP-Hard**: As hard as NP-Complete, may not be in NP
