### Greedy Algorithms

#### Greedy Choice and Optimal Substructure Property

- **Greedy Choice Property**: A globally optimal solution can be arrived at by selecting a local optimal choice. This means that at each step, the algorithm makes the choice that seems best at that moment.
- **Optimal Substructure Property**: An optimal solution to the problem contains optimal solutions to the subproblems. This means that the problem can be broken down into smaller subproblems, and the optimal solution to the overall problem can be constructed from the optimal solutions to these subproblems.

#### Problem Solving Using Greedy Approach

1. **Making Change**

   - **Problem**: Given a set of coin denominations, find the minimum number of coins to make a given amount.
   - **Algorithm**:
     1. Sort the denominations in descending order.
     2. Initialize the count of coins to 0.
     3. For each denomination, while the amount is greater than or equal to the denomination, subtract the denomination from the amount and increment the count.
   - **Example**: For denominations [1, 5, 10, 25] and amount 30, the coins used would be one 25 and one 5.
   - **Detailed Steps**:
     1. Start with amount = 30.
     2. Use the largest denomination, 25. Subtract 25 from 30, leaving 5. Count = 1.
     3. Use the next largest denomination, 10. Since 5 < 10, skip to the next denomination.
     4. Use the next denomination, 5. Subtract 5 from 5, leaving 0. Count = 2.
     5. The minimum number of coins is 2.

2. **Fractional Knapsack**

   - **Problem**: Given a set of items with weights and values, determine the maximum value that can be obtained with a given capacity.
   - **Algorithm**:
     1. Calculate the value-to-weight ratio for each item.
     2. Sort items by their value-to-weight ratio in descending order.
     3. Add items to the knapsack until the capacity is full, taking fractions of items if necessary.
   - **Example**: For items [(value=60, weight=10), (value=100, weight=20), (value=120, weight=30)] and capacity 50, the maximum value is 240.
   - **Detailed Steps**:
     1. Calculate value-to-weight ratios: [(60/10=6), (100/20=5), (120/30=4)].
     2. Sort items by ratio: [(60, 10), (100, 20), (120, 30)].
     3. Add the first item (60, 10) to the knapsack. Capacity left = 40.
     4. Add the second item (100, 20) to the knapsack. Capacity left = 20.
     5. Add the third item (120, 30) to the knapsack. Since the capacity is 20, take 2/3 of the item. Value added = 120 \* (20/30) = 80.
     6. The total value is 60 + 100 + 80 = 240.

3. **Job Scheduling**

   - **Problem**: Given a set of jobs with deadlines and profits, find the maximum profit that can be obtained by scheduling jobs.
   - **Algorithm**:
     1. Sort jobs by their profit in descending order.
     2. Initialize a result array to keep track of scheduled jobs.
     3. For each job, find the latest available time slot before its deadline and schedule it.
   - **Example**: For jobs [(profit=20, deadline=2), (profit=15, deadline=2), (profit=10, deadline=1), (profit=5, deadline=3), (profit=1, deadline=3)], the maximum profit is 35.
   - **Detailed Steps**:
     1. Sort jobs by profit: [(20, 2), (15, 2), (10, 1), (5, 3), (1, 3)].
     2. Initialize result array of size max_deadline (3 in this case).
     3. Schedule job (20, 2) in the latest available slot before deadline 2. Result = [0, 20, 0].
     4. Schedule job (15, 2) in the latest available slot before deadline 2. Result = [15, 20, 0].
     5. Job (10, 1) cannot be scheduled as the slot before deadline 1 is occupied.
     6. Schedule job (5, 3) in the latest available slot before deadline 3. Result = [15, 20, 5].
     7. Job (1, 3) cannot be scheduled as all slots before deadline 3 are occupied.
     8. The maximum profit is 15 + 20 + 5 = 40.

4. **Huffman Coding**

   - **Problem**: Construct an optimal prefix code for a given set of characters and their frequencies.
   - **Algorithm**:
     1. Create a priority queue and insert all characters with their frequencies.
     2. While there is more than one node in the queue, remove the two nodes with the lowest frequency, create a new node with these two nodes as children, and insert the new node back into the queue.
     3. The remaining node is the root of the Huffman tree.
   - **Example**: For characters [(A, 5), (B, 9), (C, 12), (D, 13), (E, 16), (F, 45)], the Huffman tree will give the optimal prefix codes.
   - **Detailed Steps**:
     1. Insert all characters into the priority queue: [(A, 5), (B, 9), (C, 12), (D, 13), (E, 16), (F, 45)].
     2. Remove the two smallest nodes (A, 5) and (B, 9), create a new node with frequency 14, and insert it back.
     3. Remove the two smallest nodes (C, 12) and (new node, 14), create a new node with frequency 26, and insert it back.
     4. Continue this process until one node remains, which is the root of the Huffman tree.
     5. Assign codes to characters based on the tree structure.

5. **Optimal Merge Pattern**

   - **Problem**: Given a set of files, find the optimal way to merge them to minimize the total cost.
   - **Algorithm**:
     1. Use a min-heap to store the sizes of the files.
     2. While there is more than one file in the heap, remove the two smallest files, merge them, and insert the merged file back into the heap.
     3. The total cost is the sum of the costs of all merges.
   - **Example**: For files [4, 8, 6, 12], the optimal merge pattern has a total cost of 58.
   - **Detailed Steps**:
     1. Insert all files into the min-heap: [4, 8, 6, 12].
     2. Remove the two smallest files (4 and 6), merge them to get a file of size 10, and insert it back.
     3. Remove the two smallest files (8 and 10), merge them to get a file of size 18, and insert it back.
     4. Remove the two smallest files (12 and 18), merge them to get a file of size 30.
     5. The total cost is 10 + 18 + 30 = 58.

6. **Graph Coloring Algorithm**

   - **Problem**: Assign colors to the vertices of a graph such that no two adjacent vertices share the same color.
   - **Algorithm**:
     1. Sort the vertices by their degree in descending order.
     2. Assign the first available color to each vertex, ensuring no two adjacent vertices share the same color.
   - **Example**: For a graph with vertices [A, B, C, D] and edges [(A, B), (B, C), (C, D)], a possible coloring is A=1, B=2, C=1, D=2.
   - **Detailed Steps**:
     1. Sort vertices by degree: [B, C, A, D].
     2. Assign color 1 to vertex B.
     3. Assign color 2 to vertex C (since it is adjacent to B).
     4. Assign color 1 to vertex A (since it is not adjacent to B or C).
     5. Assign color 2 to vertex D (since it is not adjacent to A or C).

7. **Minimum Spanning Trees**

   - **Prim's Algorithm**:
     - **Problem**: Find the minimum spanning tree of a graph.
     - **Algorithm**:
       1. Start with an arbitrary vertex.
       2. Initialize a priority queue with all edges incident to the starting vertex.
       3. While the priority queue is not empty, remove the edge with the smallest weight, add it to the MST, and add all edges incident to the newly added vertex to the priority queue.
     - **Example**: For a graph with vertices [A, B, C, D] and edges [(A, B, 1), (A, C, 3), (B, C, 1), (C, D, 4)], the MST has a total weight of 6.
     - **Detailed Steps**:
       1. Start with vertex A.
       2. Initialize the priority queue with edges [(A, B, 1), (A, C, 3)].
       3. Remove edge (A, B, 1), add it to the MST, and add edge (B, C, 1) to the priority queue.
       4. Remove edge (B, C, 1), add it to the MST, and add edge (C, D, 4) to the priority queue.
       5. Remove edge (A, C, 3) (discard it as it forms a cycle).
       6. Remove edge (C, D, 4), add it to the MST.
       7. The MST has edges [(A, B, 1), (B, C, 1), (C, D, 4)] with a total weight of 6.
   - **Kruskal's Algorithm**:
     - **Problem**: Find the minimum spanning tree of a graph.
     - **Algorithm**:
       1. Sort all edges by their weight in ascending order.
       2. Initialize a disjoint-set data structure to keep track of connected components.
       3. For each edge, if the edge connects two different components, add it to the MST and merge the components.
     - **Example**: For the same graph as above, the MST has a total weight of 6.
     - **Detailed Steps**:
       1. Sort edges by weight: [(A, B, 1), (B, C, 1), (A, C, 3), (C, D, 4)].
       2. Initialize disjoint-set with each vertex in its own component.
       3. Add edge (A, B, 1) to the MST and merge components A and B.
       4. Add edge (B, C, 1) to the MST and merge components B and C.
       5. Skip edge (A, C, 3) as it forms a cycle.
       6. Add edge (C, D, 4) to the MST and merge components C and D.
       7. The MST has edges [(A, B, 1), (B, C, 1), (C, D, 4)] with a total weight of 6.

8. **Shortest Path Algorithms**
   - **Dijkstra's Algorithm**:
     - **Problem**: Find the shortest path from a source vertex to all other vertices in a graph with non-negative weights.
     - **Algorithm**:
       1. Initialize the distance to the source vertex as 0 and to all other vertices as infinity.
       2. Use a priority queue to store vertices with their current shortest distance.
       3. While the priority queue is not empty, remove the vertex with the smallest distance, update the distances to its neighbors, and add the neighbors to the priority queue.
     - **Example**: For a graph with vertices [A, B, C, D] and edges [(A, B, 1), (A, C, 4), (B, C, 2), (C, D, 3)], the shortest path from A to D is A->B->C->D with a total distance of 6.
     - **Detailed Steps**:
       1. Initialize distances: [A=0, B=∞, C=∞, D=∞].
       2. Initialize the priority queue with vertex A.
       3. Remove vertex A, update distances to B (1) and C (4), and add them to the priority queue.
       4. Remove vertex B, update distance to C (3), and add it to the priority queue.
       5. Remove vertex C, update distance to D (6), and add it to the priority queue.
       6. Remove vertex D.
       7. The shortest path from A to D is A->B->C->D with a total distance of 6.

### Searching Algorithms and String Matching Algorithms

#### Branch and Bound

1. **The Assignment Problem**

   - **Problem**: Assign n tasks to n agents such that the total cost is minimized.
   - **Algorithm**:
     1. Use the Hungarian algorithm to find the optimal assignment.
     2. The algorithm uses a combination of row and column operations to find the minimum cost assignment.
   - **Example**: For a cost matrix [[4, 1, 3], [2, 0, 5], [3, 2, 2]], the optimal assignment has a total cost of 5.
   - **Detailed Steps**:
     1. Subtract the smallest element in each row from all elements in the row.
     2. Subtract the smallest element in each column from all elements in the column.
     3. Cover all zeros in the matrix with a minimum number of horizontal and vertical lines.
     4. If the number of lines is n, an optimal assignment is found. Otherwise, adjust the matrix and repeat.
     5. The optimal assignment is found by tracing the zeros in the adjusted matrix.

2. **The Knapsack Problem**
   - **Problem**: Given a set of items with weights and values, determine the maximum value that can be obtained with a given capacity.
   - **Algorithm**:
     1. Use a branch and bound approach to explore possible solutions.
     2. Prune branches that cannot lead to an optimal solution.
   - **Example**: For items [(value=60, weight=10), (value=100, weight=20), (value=120, weight=30)] and capacity 50, the maximum value is 220.
   - **Detailed Steps**:
     1. Start with an empty knapsack and calculate the upper bound for each item.
     2. Explore the branch with the highest upper bound first.
     3. Prune branches that have a lower upper bound than the current best solution.
     4. The optimal solution is found by exploring all possible branches.

#### String Matching Algorithms

1. **The Naïve String-Matching Algorithm**

   - **Problem**: Find all occurrences of a pattern in a text.
   - **Algorithm**:
     1. Slide the pattern over the text one character at a time.
     2. For each position, check if the pattern matches the substring of the text.
   - **Example**: For text "ABAAABCD" and pattern "AABA", the pattern occurs at index 1.
   - **Detailed Steps**:
     1. Start with the pattern at the beginning of the text.
     2. Compare the pattern with the substring of the text.
     3. If a mismatch is found, slide the pattern one character to the right.
     4. Repeat until the end of the text is reached.
     5. The pattern is found at index 1.

2. **The Rabin-Karp Algorithm**

   - **Problem**: Find all occurrences of a pattern in a text using hashing.
   - **Algorithm**:
     1. Compute the hash value of the pattern and the first substring of the text of the same length.
     2. Slide the substring over the text one character at a time, updating the hash value.
     3. If the hash values match, check if the substring matches the pattern.
   - **Example**: For text "ABAAABCD" and pattern "AABA", the pattern occurs at index 1.
   - **Detailed Steps**:
     1. Compute the hash value of the pattern "AABA".
     2. Compute the hash value of the first substring of the text "ABAA".
     3. Slide the substring one character to the right and update the hash value.
     4. If the hash values match, check if the substring matches the pattern.
     5. The pattern is found at index 1.

3. **KMP Algorithm for Pattern Searching**
   - **Problem**: Find all occurrences of a pattern in a text using a prefix table.
   - **Algorithm**:
     1. Compute the prefix table for the pattern.
     2. Use the prefix table to skip characters in the text that have already been matched.
   - **Example**: For text "ABAAABCD" and pattern "AABA", the pattern occurs at index 1.
   - **Detailed Steps**:
     1. Compute the prefix table for the pattern "AABA": [0, 1, 0, 1].
     2. Start with the pattern at the beginning of the text.
     3. Compare the pattern with the substring of the text.
     4. If a mismatch is found, use the prefix table to skip characters in the text.
     5. Repeat until the end of the text is reached.
     6. The pattern is found at index 1.

### Dynamic Programming

#### The Principle of Optimality

- **Principle of Optimality**: An optimal solution to a problem contains optimal solutions to the subproblems. This means that the optimal solution to the overall problem can be constructed from the optimal solutions to the subproblems.

#### Dynamic Programming Using Memorization and Bottom-Up Approach

- **Memorization**: Store the results of subproblems to avoid redundant calculations. This is often implemented using a top-down approach with recursion and a memoization table.
- **Bottom-Up Approach**: Solve smaller subproblems first and build up the solution to the original problem. This is often implemented using an iterative approach with a table.

#### Problem Solving Using Dynamic Programming

1. **Calculating the Binomial Coefficient**

   - **Problem**: Compute the binomial coefficient C(n, k).
   - **Algorithm**:
     1. Use the formula C(n, k) = C(n-1, k-1) + C(n-1, k).
     2. Store the results of subproblems in a table.
   - **Example**: C(5, 2) = 10.
   - **Detailed Steps**:
     1. Initialize a table of size (n+1) x (k+1) with 0s.
     2. Fill the table using the formula C(i, j) = C(i-1, j-1) + C(i-1, j).
     3. The value C(5, 2) is found in the table.

2. **Assembly Line-Scheduling**

   - **Problem**: Schedule tasks on two assembly lines to minimize the total time.
   - **Algorithm**:
     1. Use dynamic programming to find the minimum time to complete tasks on each line.
     2. Combine the results to find the optimal schedule.
   - **Example**: For two lines with tasks [(2, 4), (3, 2)] and transfer times [(2, 3), (1, 2)], the minimum time is 9.
   - **Detailed Steps**:
     1. Initialize tables for the minimum time to complete tasks on each line.
     2. Fill the tables using the formula T1(i) = min(T1(i-1) + a1(i), T2(i-1) + t1(i) + a1(i)).
     3. Combine the results to find the minimum time.

3. **Knapsack Problem**

   - **Problem**: Given a set of items with weights and values, determine the maximum value that can be obtained with a given capacity.
   - **Algorithm**:
     1. Use a table to store the maximum value for each capacity and number of items.
     2. Fill the table using the formula max_value(i, w) = max(max_value(i-1, w), value[i] + max_value(i-1, w-weight[i])).
   - **Example**: For items [(value=60, weight=10), (value=100, weight=20), (value=120, weight=30)] and capacity 50, the maximum value is 220.
   - **Detailed Steps**:
     1. Initialize a table of size (n+1) x (capacity+1) with 0s.
     2. Fill the table using the formula max_value(i, w) = max(max_value(i-1, w), value[i] + max_value(i-1, w-weight[i])).
     3. The maximum value is found in the table.

4. **Matrix Chain Multiplication**

   - **Problem**: Find the most efficient way to multiply a chain of matrices.
   - **Algorithm**:
     1. Use a table to store the minimum number of scalar multiplications for each subchain.
     2. Fill the table using the formula m[i, j] = min(m[i, k] + m[k+1, j] + p[i-1]*p[k]*p[j]) for all k from i to j-1.
   - **Example**: For matrices with dimensions [10, 30, 5, 60], the minimum number of scalar multiplications is 4500.
   - **Detailed Steps**:
     1. Initialize a table of size n x n with 0s.
     2. Fill the table using the formula m[i, j] = min(m[i, k] + m[k+1, j] + p[i-1]*p[k]*p[j]) for all k from i to j-1.
     3. The minimum number of scalar multiplications is found in the table.

5. **Sum of Subset Problem**

   - **Problem**: Given a set of numbers, find if there is a subset with a given sum.
   - **Algorithm**:
     1. Use a table to store whether a subset with a given sum exists for each number of elements.
     2. Fill the table using the formula subset_sum(i, s) = subset_sum(i-1, s) or subset_sum(i-1, s-num[i]).
   - **Example**: For numbers [3, 34, 4, 12, 5, 2] and sum 9, a subset with the sum exists.
   - **Detailed Steps**:
     1. Initialize a table of size (n+1) x (sum+1) with False.
     2. Fill the table using the formula subset_sum(i, s) = subset_sum(i-1, s) or subset_sum(i-1, s-num[i]).
     3. The result is found in the table.

6. **Longest Common Subsequence**

   - **Problem**: Find the longest subsequence common to two sequences.
   - **Algorithm**:
     1. Use a table to store the length of the longest common subsequence for each pair of prefixes.
     2. Fill the table using the formula lcs(i, j) = lcs(i-1, j-1) + 1 if seq1[i] == seq2[j], otherwise max(lcs(i-1, j), lcs(i, j-1)).
   - **Example**: For sequences "AGGTAB" and "GXTXAYB", the longest common subsequence is "GTAB" with length 4.
   - **Detailed Steps**:
     1. Initialize a table of size (m+1) x (n+1) with 0s.
     2. Fill the table using the formula lcs(i, j) = lcs(i-1, j-1) + 1 if seq1[i] == seq2[j], otherwise max(lcs(i-1, j), lcs(i, j-1)).
     3. The length of the longest common subsequence is found in the table.

7. **Floyd-Warshall Algorithm**

   - **Problem**: Find the shortest paths between all pairs of vertices in a weighted graph.
   - **Algorithm**:
     1. Initialize the distance matrix with the weights of the edges.
     2. For each pair of vertices (i, j), update the distance using the formula dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]) for all k.
   - **Example**: For a graph with vertices [A, B, C, D] and edges [(A, B, 1), (A, C, 4), (B, C, 2), (C, D, 3)], the shortest path from A to D is A->B->C->D with a total distance of 6.
   - **Detailed Steps**:
     1. Initialize the distance matrix with the weights of the edges.
     2. For each pair of vertices (i, j), update the distance using the formula dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]) for all k.
     3. The shortest paths are found in the distance matrix.

8. **Bellman-Ford Algorithm**
   - **Problem**: Find the shortest paths from a source vertex to all other vertices in a weighted graph with negative weights.
   - **Algorithm**:
     1. Initialize the distance to the source vertex as 0 and to all other vertices as infinity.
     2. Relax all edges |V|-1 times, updating the distances.
     3. Check for negative weight cycles by relaxing all edges one more time.
   - **Example**: For a graph with vertices [A, B, C, D] and edges [(A, B, 1), (A, C, 4), (B, C, -2), (C, D, 3)], the shortest path from A to D is A->B->C->D with a total distance of 2.
   - **Detailed Steps**:
     1. Initialize distances: [A=0, B=∞, C=∞, D=∞].
     2. Relax all edges |V|-1 times, updating the distances.
     3. Check for negative weight cycles by relaxing all edges one more time.
     4. The shortest paths are found in the distance array.

### Introduction to Algorithms

**Algorithms** are step-by-step procedures or formulas for solving problems. They are fundamental to computer science and are used to perform calculations, data processing, and automated reasoning tasks.

### Analysis and Design Techniques of Algorithms

**Analysis of Algorithms**: The process of determining the computational complexity of algorithms. This involves studying the time and space requirements of an algorithm.

**Design Techniques**:
1. **Divide and Conquer**: Breaking down a problem into smaller subproblems, solving each subproblem recursively, and combining the solutions.
2. **Dynamic Programming**: Solving problems by breaking them down into simpler subproblems and storing the results of these subproblems to avoid redundant calculations.
3. **Greedy Algorithms**: Making a series of choices, each of which looks the best at the moment, to find an overall optimal solution.
4. **Backtracking**: Exploring all possible solutions by building them incrementally and abandoning partial solutions as soon as it is determined that they cannot lead to a valid solution.

### Model for Analysis of Algorithm: Random Access Machine (RAM)

The **Random Access Machine (RAM)** is an abstract model used to analyze the time complexity of algorithms. It assumes that each simple operation (like addition, multiplication, or memory access) takes constant time. This model helps in understanding the performance of algorithms independent of the specific hardware.

### Algorithm Analysis Techniques: Mathematics, Empirical and Asymptotic Analysis

**Mathematical Analysis**: Using mathematical techniques to derive the time complexity of an algorithm. This involves solving recurrence relations and using mathematical tools like summations and integrals.

**Empirical Analysis**: Measuring the performance of an algorithm by running it on a real machine and observing the time and space it uses. This is useful for understanding the practical performance of an algorithm.

**Asymptotic Analysis**: Studying the performance of an algorithm as the input size grows very large. This helps in understanding the scalability of an algorithm.

### Big-O, Big-Theta, and Big-Omega Notations

**Big-O Notation (O(n))**: Describes the upper bound of the time complexity of an algorithm. It represents the worst-case scenario.

**Big-Theta Notation (Θ(n))**: Describes the exact bound of the time complexity of an algorithm. It represents the average-case scenario.

**Big-Omega Notation (Ω(n))**: Describes the lower bound of the time complexity of an algorithm. It represents the best-case scenario.

### Worst, Average, and Best Case Analysis

**Worst-Case Analysis**: Determines the maximum time an algorithm will take to complete. It is useful for understanding the upper limit of the algorithm's performance.

**Average-Case Analysis**: Determines the expected time an algorithm will take to complete, assuming a random distribution of inputs. It is useful for understanding the typical performance of an algorithm.

**Best-Case Analysis**: Determines the minimum time an algorithm will take to complete. It is useful for understanding the lower limit of the algorithm's performance.

### Complexity Analysis of Iterative Algorithm: Primitive Operations, Analyzing Control Statements

**Primitive Operations**: Basic operations that take constant time, such as addition, subtraction, multiplication, division, and comparison.

**Analyzing Control Statements**:
- **Loops**: The time complexity of a loop is determined by the number of iterations it performs. For example, a loop that runs n times has a time complexity of O(n).
- **Nested Loops**: The time complexity of nested loops is the product of the time complexities of the individual loops. For example, two nested loops that each run n times have a time complexity of O(n^2).
- **Conditional Statements**: The time complexity of a conditional statement is the maximum time complexity of its branches.

### Complexity Analysis Techniques for Recurrence Relations

**Recurrence Relations**: Equations that define a function in terms of its smaller instances. They are used to describe the time complexity of recursive algorithms.

**Substitution Method**: Guessing the form of the solution and using mathematical induction to prove it.

**Iterative Method**: Unfolding the recurrence relation iteratively to find a pattern and then solving it.

**Recursion Tree Method**: Drawing a tree that represents the recursive calls of the algorithm and summing the costs at each level of the tree.

**Master’s Theorem**: A direct way to solve recurrence relations of the form T(n) = aT(n/b) + f(n), where a ≥ 1, b > 1, and f(n) is an asymptotically positive function.

### Basic Sorting Algorithms

#### Selection Sort

**Algorithm**:
1. Find the minimum element in the array and swap it with the first element.
2. Repeat the process for the remaining elements.

**Time Complexity**:
- Best Case: O(n^2)
- Average Case: O(n^2)
- Worst Case: O(n^2)

**Example**: Sorting [64, 25, 12, 22, 11]
- Step 1: [11, 25, 12, 22, 64]
- Step 2: [11, 12, 25, 22, 64]
- Step 3: [11, 12, 22, 25, 64]
- Step 4: [11, 12, 22, 25, 64]

#### Insertion Sort

**Algorithm**:
1. Start with the second element and compare it with the elements before it.
2. Insert the element in its correct position.
3. Repeat the process for the remaining elements.

**Time Complexity**:
- Best Case: O(n) (when the array is already sorted)
- Average Case: O(n^2)
- Worst Case: O(n^2)

**Example**: Sorting [12, 11, 13, 5, 6]
- Step 1: [11, 12, 13, 5, 6]
- Step 2: [11, 12, 13, 5, 6]
- Step 3: [5, 11, 12, 13, 6]
- Step 4: [5, 6, 11, 12, 13]

#### Bubble Sort

**Algorithm**:
1. Compare each pair of adjacent elements and swap them if they are in the wrong order.
2. Repeat the process for the remaining elements until the array is sorted.

**Time Complexity**:
- Best Case: O(n) (when the array is already sorted)
- Average Case: O(n^2)
- Worst Case: O(n^2)

**Example**: Sorting [64, 34, 25, 12, 22, 11, 90]
- Step 1: [64, 34, 25, 12, 22, 11, 90]
- Step 2: [34, 25, 12, 22, 11, 64, 90]
- Step 3: [25, 12, 22, 11, 34, 64, 90]
- Step 4: [12, 22, 11, 25, 34, 64, 90]
- Step 5: [12, 11, 22, 25, 34, 64, 90]
- Step 6: [11, 12, 22, 25, 34, 64, 90]
