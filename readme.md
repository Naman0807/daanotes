# Design and Analysis of Algorithms - Complete Summary

## Table of Contents

1. [Introduction to Algorithms](#introduction-to-algorithms)
   - [Analysis and Design Techniques](#analysis-and-design-techniques-of-algorithms)
   - [Model for Analysis: RAM](#model-for-analysis-of-algorithm-random-access-machine-ram)
   - [Algorithm Analysis Techniques](#algorithm-analysis-techniques-mathematics-empirical-and-asymptotic-analysis)
   - [Complexity Notations](#big-o-big-theta-and-big-omega-notations)
   - [Case Analysis](#worst-average-and-best-case-analysis)
   - [Complexity Analysis of Iterative Algorithms](#complexity-analysis-of-iterative-algorithm-primitive-operations-analyzing-control-statements)
   - [Complexity Analysis of Recursive Algorithms](#complexity-analysis-techniques-for-recurrence-relations)
   - [Basic Sorting Algorithms](#basic-sorting-algorithms)
2. [Greedy Algorithms](#greedy-algorithms)
   - [Greedy Choice and Optimal Substructure Property](#greedy-choice-and-optimal-substructure-property)
   - [Problem Solving Using Greedy Approach](#problem-solving-using-greedy-approach)
3. [Dynamic Programming](#dynamic-programming)
   - [The Principle of Optimality](#the-principle-of-optimality)
   - [Memorization and Bottom-Up Approach](#dynamic-programming-using-memorization-and-bottom-up-approach)
   - [Problem Solving Using Dynamic Programming](#problem-solving-using-dynamic-programming)
4. [Searching Algorithms and String Matching](#searching-algorithms-and-string-matching-algorithms)
   - [Branch and Bound](#branch-and-bound)
   - [String Matching Algorithms](#string-matching-algorithms)

---

## **Introduction to Algorithms**

**Algorithms** are step-by-step procedures or formulas for solving problems. They are fundamental to computer science and are used to perform calculations, data processing, and automated reasoning tasks. An algorithm takes input data, processes it through a sequence of well-defined operations, and produces an output. The key characteristics of a good algorithm include correctness, efficiency, simplicity, and optimality.

### Analysis and Design Techniques of Algorithms

**Analysis of Algorithms**: The process of determining the computational complexity of algorithms involves analyzing how the algorithm's resource consumption (time and space) scales with input size. This helps in comparing algorithms and selecting the most efficient one for a given problem.

**Design Techniques**:

1. **Divide and Conquer**:

   - Divides the problem into smaller, independent subproblems of the same type
   - Solves each subproblem recursively
   - Combines the solutions to subproblems to form the solution to the original problem
   - Examples: Merge Sort (divides array in half, sorts each half, merges results), Quick Sort (partitions array around pivot, sorts each partition)
   - Applications: Binary search, FFT (Fast Fourier Transform), Strassen's matrix multiplication

2. **Dynamic Programming**:

   - Breaks problems into overlapping subproblems
   - Stores solutions to subproblems in a table to avoid redundant calculations
   - Builds solution bottom-up or top-down with memoization
   - Key characteristics: Optimal substructure and overlapping subproblems
   - Applications: Shortest path problems, resource allocation, sequence alignment

3. **Greedy Algorithms**:

   - Makes locally optimal choices at each step to find global optimum
   - Never reconsiders previous choices
   - Simpler and more efficient than dynamic programming when applicable
   - May not always yield optimal solutions for all problems
   - Applications: Minimum spanning trees, Huffman coding, activity selection

4. **Backtracking**:
   - Builds solutions incrementally, abandoning a path when it determines the path cannot lead to a valid solution
   - Uses depth-first search approach to explore solution space
   - Prunes search space by rejecting invalid partial solutions early
   - Applications: Constraint satisfaction problems, combinatorial optimization

### Model for Analysis of Algorithm: Random Access Machine (RAM)

The **Random Access Machine (RAM)** model provides a theoretical framework for analyzing algorithm performance independent of specific hardware details.

**Detailed Assumptions**:

- Each simple operation (arithmetic, comparison, assignment) takes exactly one time unit
- Memory consists of infinitely many cells, each capable of storing any integer
- Memory access takes constant time regardless of the location
- Control flow operations (loops, conditionals) comprise sequences of simple operations
- The model abstracts away practical considerations like cache behavior, memory hierarchy, and instruction pipelining

This model allows for theoretical analysis of algorithms in a hardware-independent manner, though real-world performance may differ due to hardware specifics.

### Algorithm Analysis Techniques: Mathematics, Empirical and Asymptotic Analysis

- **Mathematical Analysis**:

  - Uses mathematical tools to derive exact expressions for algorithm performance
  - Involves counting primitive operations, solving recurrence relations
  - Provides rigorous proofs of algorithm behavior
  - Example: Analyzing the exact number of comparisons in a sorting algorithm

- **Empirical Analysis**:

  - Implements the algorithm and measures actual performance
  - Tests with various inputs and sizes to observe practical behavior
  - Accounts for real-world factors like caching, compiler optimizations
  - Uses profiling tools to identify bottlenecks
  - Limited by test cases and hardware-specific results

- **Asymptotic Analysis**:
  - Focuses on algorithm behavior with very large inputs
  - Ignores constant factors and lower-order terms
  - Provides a high-level understanding of scalability
  - Uses notation like Big-O, Big-Theta, and Big-Omega
  - Most commonly used method in theoretical computer science

### Big-O, Big-Theta, and Big-Omega Notations

These notations help classify algorithms according to their growth rates:

- **Big-O Notation (O(n))**:

  - Formally: f(n) = O(g(n)) if there exist positive constants c and n₀ such that 0 ≤ f(n) ≤ c·g(n) for all n ≥ n₀
  - Represents an upper bound on growth rate
  - Example: A linear search algorithm is O(n), meaning its running time grows at most linearly with input size
  - Common complexities (from fastest to slowest): O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)

- **Big-Theta Notation (Θ(n))**:

  - Formally: f(n) = Θ(g(n)) if there exist positive constants c₁, c₂, and n₀ such that c₁·g(n) ≤ f(n) ≤ c₂·g(n) for all n ≥ n₀
  - Represents tight bound (both upper and lower)
  - Example: Binary search is Θ(log n), meaning its running time grows exactly logarithmically with input size

- **Big-Omega Notation (Ω(n))**:
  - Formally: f(n) = Ω(g(n)) if there exist positive constants c and n₀ such that f(n) ≥ c·g(n) for all n ≥ n₀
  - Represents a lower bound on growth rate
  - Example: Any comparison-based sorting algorithm is Ω(n log n), meaning it cannot be faster than n log n in the worst case

### Worst, Average, and Best Case Analysis

Understanding algorithm performance across different input scenarios:

- **Worst-Case Analysis**:

  - Analyzes algorithm performance for the most unfavorable input
  - Guarantees an upper bound on running time
  - Important for real-time applications and safety-critical systems
  - Example: QuickSort has O(n²) worst-case when the pivot selection consistently results in unbalanced partitions

- **Average-Case Analysis**:

  - Analyzes expected performance across all possible inputs
  - Usually assumes a probability distribution over inputs
  - Often more representative of typical performance
  - Example: QuickSort has O(n log n) average-case with random pivot selection
  - Mathematically more complex than worst-case analysis

- **Best-Case Analysis**:
  - Analyzes performance for the most favorable input
  - Rarely used alone as it's often unrealistically optimistic
  - Can be useful to determine lower bounds on performance
  - Example: Bubble sort has O(n) best-case when the array is already sorted

### Complexity Analysis of Iterative Algorithm: Primitive Operations, Analyzing Control Statements

**Primitive Operations** form the basis for algorithm analysis:

- Basic arithmetic operations (addition, subtraction, multiplication, division)
- Comparison operations (equality and inequality tests)
- Assignment operations
- Array/variable access operations
- Function call overhead

**Analyzing Control Structures**:

- **Sequential Statements**: Sum the complexities of individual statements
- **If-Else Statements**: Take the maximum complexity of the if or else branch
- **For Loops**: Multiply the complexity of the loop body by the number of iterations
  - Example: `for(i=0; i<n; i++)` with O(1) body has O(n) complexity
- **Nested Loops**: Multiply the complexities of each loop
  - Example: Two nested loops each running n times: O(n²)
  - Three nested loops: O(n³)
- **While/Do-While Loops**: Analyze based on the number of iterations and body complexity
  - May require understanding the loop termination condition

### Complexity Analysis Techniques for Recurrence Relations

Recurrence relations often arise when analyzing recursive algorithms:

- **Substitution Method**:

  - Guess a bound and prove it using mathematical induction
  - Steps: (1) Guess the solution form, (2) Verify the base case, (3) Assume solution for smaller values, (4) Prove for the current value
  - Example: For T(n) = 2T(n/2) + n, guess T(n) = O(n log n) and prove by induction

- **Iterative Method**:

  - Expand the recurrence and look for patterns
  - Repeatedly substitute the definition until reaching base cases
  - Sum up the resulting expression
  - Example: T(n) = T(n-1) + 1 expands to T(n) = T(n-2) + 1 + 1 = ... = T(1) + (n-1) = O(n)

- **Recursion Tree Method**:

  - Draw a tree where each node represents a subproblem
  - Calculate cost at each level of the tree
  - Sum costs across all levels
  - Example: For T(n) = 2T(n/2) + n, the tree has log n levels, each with cost n, so total cost is n log n

- **Master's Theorem**:
  - Provides solutions for recurrences of the form T(n) = aT(n/b) + f(n), where a ≥ 1, b > 1
  - Three cases based on how f(n) compares to n^(log_b a):
    1. If f(n) = O(n^(log_b a-ε)) for some ε > 0, then T(n) = Θ(n^(log_b a))
    2. If f(n) = Θ(n^(log_b a)), then T(n) = Θ(n^(log_b a) log n)
    3. If f(n) = Ω(n^(log_b a+ε)) for some ε > 0, and af(n/b) ≤ cf(n) for c < 1, then T(n) = Θ(f(n))
  - Example: For T(n) = 2T(n/2) + n, a=2, b=2, f(n)=n, and n^(log_b a) = n^(log_2 2) = n^1 = n, so case 2 applies, giving T(n) = Θ(n log n)

### Basic Sorting Algorithms

#### Selection Sort

- **Detailed Algorithm**:
  1. Divide the array into sorted and unsorted regions (initially, sorted region is empty)
  2. Find the smallest element in the unsorted region
  3. Swap it with the first element of the unsorted region
  4. Expand sorted region by one element
  5. Repeat until entire array is sorted
- **Time Complexity Analysis**:
  - Finding minimum requires n-1 comparisons for first pass, n-2 for second, and so on
  - Total comparisons: (n-1) + (n-2) + ... + 1 = n(n-1)/2 = O(n²)
  - Best, average, and worst cases all require O(n²) time
- **Space Complexity**: O(1) as sorting is done in-place with only a few temporary variables
- **Key Properties**: Simple, stable (doesn't change relative order of equal elements), performs poorly on large lists

#### Insertion Sort

- **Detailed Algorithm**:
  1. Start with the second element
  2. Compare it with elements to its left
  3. Shift larger elements to the right
  4. Insert the element in its correct position
  5. Proceed with next element
  6. Repeat until entire array is sorted
- **Time Complexity Analysis**:
  - Best case: O(n) when array is already sorted (each element requires only one comparison)
  - Worst case: O(n²) when array is in reverse order (each element must be compared with all previous elements)
  - Average case: O(n²) for random input
- **Space Complexity**: O(1) as sorting is done in-place
- **Key Properties**: Efficient for small data sets, adaptive (performs better on partially sorted arrays), stable, online (can sort data as it arrives)

#### Bubble Sort

- **Detailed Algorithm**:
  1. Compare adjacent elements and swap if they're in wrong order
  2. After first pass, largest element is at the end
  3. Repeat the process for remaining elements
  4. Each pass places the next largest element in its final position
- **Time Complexity Analysis**:
  - Best case: O(n) with optimization to detect if any swaps were made
  - Worst case: O(n²) when array is in reverse order
  - Average case: O(n²) for random input
- **Space Complexity**: O(1) as sorting is done in-place
- **Key Properties**: Simple to implement, stable, inefficient for large lists, can be optimized by stopping early if no swaps occur in a pass

---

## **Greedy Algorithms**

### Greedy Choice and Optimal Substructure Property

**Greedy Choice Property** ensures that a locally optimal choice leads to a globally optimal solution:

- At each step, a greedy algorithm makes the choice that looks best at the moment
- The choice is made based on the current state without considering future consequences
- Once a choice is made, it's never reconsidered
- For this property to hold, the problem must be structured such that local optimality leads to global optimality
- Without this property, a greedy algorithm may not find the optimal solution

**Optimal Substructure Property** means:

- An optimal solution contains optimal solutions to its subproblems
- This allows problems to be solved by breaking them into smaller instances
- If a problem lacks this property, dynamic programming or other approaches may be needed
- Example: Shortest path problem has optimal substructure because any subpath of a shortest path is itself a shortest path

For a greedy algorithm to be applicable, both properties must typically be present and provable for the specific problem.

### Problem Solving Using Greedy Approach

1. **Making Change (Coin Change Problem)**

   - **Detailed Problem**: Given a set of coin denominations and a target amount, find the minimum number of coins needed to make up that amount.
   - **In-depth Algorithm**:
     1. Sort denominations in descending order (largest first)
     2. Initialize coins_used = 0
     3. For each denomination, starting with largest:
        a. Calculate max_coins = floor(remaining_amount / denomination)
        b. Add max_coins to coins_used
        c. Subtract max_coins × denomination from remaining_amount
     4. If remaining_amount > 0 after considering all denominations, no solution exists
   - **Why Greedy Works**: For standard coin systems (like US currency), the greedy approach works because each coin denomination is a multiple of the smaller ones. However, this doesn't work for all possible denomination sets.
   - **Limitations**: For arbitrary coin denominations (e.g., [1, 3, 4]), greedy can fail. Example: To make 6, greedy gives [4, 1, 1] (3 coins) while optimal is [3, 3] (2 coins).

2. **Fractional Knapsack**

   - **Detailed Problem**: Maximize value in a knapsack with weight capacity W using items that can be taken fractionally.
   - **In-depth Algorithm**:
     1. Calculate value-per-weight ratio (v/w) for each item
     2. Sort items by this ratio in descending order
     3. Initialize total_value = 0 and current_weight = 0
     4. For each item in the sorted list:
        a. If adding the entire item doesn't exceed capacity, add it completely
        b. Otherwise, add the fraction that fits
     5. Return total_value
   - **Proof of Optimality**: By always choosing items with highest value-per-weight ratio, we maximize the value for each unit of weight. Since items can be fractionally selected, we can always fill the knapsack to its exact capacity.
   - **Comparison with 0/1 Knapsack**: In 0/1 Knapsack, items must be taken entirely or not at all, making it unsuitable for greedy approach (requires dynamic programming).

3. **Job Scheduling with Deadlines**

   - **Detailed Problem**: Given jobs with profits and deadlines, schedule them to maximize total profit. Each job takes one unit of time, and only one job can be executed at a time.
   - **In-depth Algorithm**:
     1. Sort jobs in decreasing order of profit
     2. Initialize a time slot array of size max_deadline, filled with -1 (indicating empty slots)
     3. For each job in the sorted order:
        a. Find the latest available time slot before its deadline
        b. If found, assign the job to that slot
     4. Jobs assigned to slots are the scheduled jobs
   - **Time Complexity**: O(n²) where n is the number of jobs (sorting takes O(n log n), and for each job, finding a slot can take O(n))
   - **Example**: For jobs [(20,2), (15,2), (10,1), (5,3), (1,3)], the scheduling would be:
     - Slot 1: Job with profit 15
     - Slot 2: Job with profit 20
     - Slot 3: Job with profit 5
     - Total profit: 40

4. **Huffman Coding**

   - **Detailed Problem**: Construct minimum-redundancy codes for a set of characters based on their frequencies.
   - **In-depth Algorithm**:
     1. Create a leaf node for each character and add them to a min-priority queue based on frequency
     2. While the queue has more than one node:
        a. Extract two nodes with lowest frequencies (let's call them left and right)
        b. Create a new internal node with frequency = left.frequency + right.frequency
        c. Make left and right the children of this new node
        d. Add the new node back to the priority queue
     3. The remaining node is the root of the Huffman tree
     4. Traverse the tree to assign binary codes (0 for left, 1 for right)
   - **Properties of Huffman Codes**:
     - They are prefix-free (no code is a prefix of another)
     - More frequent characters get shorter codes
     - They provide optimal variable-length encoding
   - **Time Complexity**: O(n log n) where n is the number of characters (building heap takes O(n), and extracting/inserting nodes takes O(log n) up to n-1 times)

5. **Optimal Merge Pattern (External Sorting)**

   - **Detailed Problem**: Given a set of sorted files, find the optimal way to merge them into a single file by minimizing the total number of comparisons.
   - **In-depth Algorithm**:
     1. Create a min-heap with all file sizes
     2. While the heap has more than one element:
        a. Extract the two smallest files (let's call them A and B)
        b. Merge them, creating a new file of size A+B
        c. Insert the new file back into the heap
     3. The total cost is the sum of all merge operations
   - **Mathematical Basis**: This problem is equivalent to finding the minimum-weight external path length of a binary tree, where leaf nodes represent the files.
   - **Application**: Used in external sorting when data doesn't fit in memory and must be sorted in chunks, then merged.

6. **Graph Coloring**

   - **Detailed Problem**: Assign colors to vertices of a graph such that no two adjacent vertices have the same color, using the minimum number of colors.
   - **In-depth Algorithm**:
     1. Sort vertices in descending order of degree (number of adjacent vertices)
     2. Assign the first available color to each vertex:
        a. Start with the first vertex and assign color 1
        b. For each subsequent vertex, try colors 1, 2, 3, ... until finding one not used by any adjacent vertex
     3. Return the colored graph and the number of colors used
   - **Approximation Quality**: This greedy approach doesn't guarantee an optimal solution but provides a reasonable approximation. For general graphs, it can use at most d+1 colors where d is the maximum degree.
   - **Theoretical Bounds**: Finding the minimum number of colors (chromatic number) is NP-hard. The greedy algorithm can use up to twice as many colors as optimal in the worst case.

7. **Minimum Spanning Trees**

   - **Prim's Algorithm**:

     - **Detailed Problem**: Find a minimum spanning tree (MST) of a connected, weighted, undirected graph.
     - **In-depth Algorithm**:
       1. Initialize an empty MST and a min-priority queue of vertices
       2. Select an arbitrary start vertex and set its key to 0 (all others to infinity)
       3. While the queue is not empty:
          a. Extract the vertex u with minimum key value
          b. Add u to the MST
          c. For each adjacent vertex v not in MST:
          - If weight(u,v) < key[v], update key[v] = weight(u,v) and set parent[v] = u
       4. The MST consists of edges defined by the parent relationship
     - **Time Complexity**: O(E log V) with binary heap, where E is the number of edges and V is the number of vertices
     - **Properties**: Works well for dense graphs, grows MST one vertex at a time

   - **Kruskal's Algorithm**:

     - **Detailed Problem**: Same as Prim's, but with a different approach.
     - **In-depth Algorithm**:
       1. Sort all edges in non-decreasing order of weight
       2. Initialize an empty MST and a disjoint-set data structure for vertices
       3. For each edge (u,v) in the sorted order:
          a. If adding (u,v) doesn't create a cycle (u and v are in different sets):
          - Add edge (u,v) to the MST
          - Union the sets containing u and v
       4. Continue until MST has V-1 edges
     - **Time Complexity**: O(E log E) or O(E log V) for sorting edges, where E is the number of edges and V is the number of vertices
     - **Properties**: Works well for sparse graphs, builds MST one edge at a time

   - **Comparison**: Prim's algorithm starts with a vertex and grows the MST by adding vertices, while Kruskal's algorithm sorts all edges and adds them one by one if they don't create cycles.

8. **Shortest Path Algorithms**
   - **Dijkstra's Algorithm**:
     - **Detailed Problem**: Find the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edges.
     - **In-depth Algorithm**:
       1. Initialize distances: source = 0, all others = infinity
       2. Initialize a min-priority queue with all vertices
       3. While the queue is not empty:
          a. Extract vertex u with minimum distance
          b. For each neighbor v of u:
          - If distance[u] + weight(u,v) < distance[v]:
            - Update distance[v] = distance[u] + weight(u,v)
            - Update parent[v] = u
       4. The shortest paths can be constructed using the parent pointers
     - **Correctness Proof**: Based on the property that any subpath of a shortest path is itself a shortest path
     - **Time Complexity**: O(E log V) with binary heap, where E is the number of edges and V is the number of vertices
     - **Limitations**: Cannot handle negative edge weights (Bellman-Ford algorithm is used for such cases)

---

## **Dynamic Programming**

### The Principle of Optimality

The Principle of Optimality, formulated by Richard Bellman, is the theoretical foundation of dynamic programming:

- **Formal Definition**: An optimal policy has the property that, regardless of the initial state and decision, the remaining decisions must constitute an optimal policy with respect to the state resulting from the first decision.

- **Implications**:

  - If we know the optimal solution to a problem, then the solutions to all of its subproblems must also be optimal
  - We can build the optimal solution to the original problem by combining optimal solutions to its subproblems
  - This allows us to avoid redundant calculations by solving each subproblem only once

- **Mathematical Formulation**: If P(n) is the original problem and P(i) for i < n are its subproblems, then the optimal solution S(n) to P(n) contains within it the optimal solutions S(i) to problems P(i).

- **Contrast with Greedy Algorithms**: While both approaches exploit optimal substructure, dynamic programming considers all possible combinations of subproblem solutions, whereas greedy algorithms make an irrevocable choice at each step.

### Dynamic Programming Using Memorization and Bottom-Up Approach

Dynamic programming implementations typically follow one of two approaches:

- **Top-Down with Memoization**:

  - Start with the original problem
  - Break it down recursively into subproblems
  - Store solutions to subproblems in a table (memoize)
  - Check the table before solving a subproblem to avoid redundant work
  - Advantages:
    - Only solves subproblems that are actually needed
    - Often follows the natural recursive structure of the problem
    - Easier to understand for some problems
  - Disadvantages:
    - Recursive calls add overhead
    - May cause stack overflow for problems with large input sizes
    - Less efficient memory access patterns

- **Bottom-Up Approach**:

  - Identify the smallest subproblems
  - Solve them first and store their solutions
  - Gradually build up solutions to larger subproblems
  - Eventually solve the original problem
  - Advantages:
    - Avoids recursive overhead
    - Better memory locality
    - No risk of stack overflow
  - Disadvantages:
    - May solve subproblems that aren't needed for the final solution
    - Sometimes less intuitive than the recursive formulation

- **State Design**: Both approaches require careful design of:
  - State representation (what information defines a subproblem)
  - State transitions (how to build larger solutions from smaller ones)
  - Base cases (solutions to the smallest subproblems)

### Problem Solving Using Dynamic Programming

1. **Calculating Binomial Coefficient**

   - **Detailed Problem**: Calculate C(n,k), the number of ways to choose k items from n items, where order doesn't matter.
   - **Mathematical Definition**: C(n,k) = n! / (k! \* (n-k)!)
   - **Recurrence Relation**: C(n,k) = C(n-1,k-1) + C(n-1,k)
   - **Bottom-Up Algorithm**:
     1. Create a 2D table C[0...n][0...k]
     2. Initialize C[i][0] = 1 for all i (there's only one way to choose 0 elements)
     3. Initialize C[i][i] = 1 for all i (there's only one way to choose all elements)
     4. For i from 1 to n:
        a. For j from 1 to min(i,k):
        - C[i][j] = C[i-1][j-1] + C[i-1][j]
     5. Return C[n][k]
   - **Time and Space Complexity**: O(n\*k) for both time and space
   - **Application**: Calculating probabilities in binomial distribution, counting combinations

2. **Assembly Line Scheduling**

   - **Detailed Problem**: Given two assembly lines, each with n stations, and the time to process a car at each station, find the minimum time to assemble a car.
   - **Problem Parameters**:
     - a[i,j] = processing time at station j on line i
     - t[i,j] = transfer time from station j on line i to station j+1 on the other line
     - e[i] = entry time for line i
     - x[i] = exit time from line i
   - **Recurrence Relation**:
     - f1[j] = minimum time to reach station j on line 1
     - f2[j] = minimum time to reach station j on line 2
     - f1[1] = e[1] + a[1,1]
     - f2[1] = e[2] + a[2,1]
     - For j = 2 to n:
       - f1[j] = min(f1[j-1] + a[1,j], f2[j-1] + t[2,j-1] + a[1,j])
       - f2[j] = min(f2[j-1] + a[2,j], f1[j-1] + t[1,j-1] + a[2,j])
     - Final answer = min(f1[n] + x[1], f2[n] + x[2])
   - **Time and Space Complexity**: O(n) for both time and space
   - **Application**: Manufacturing processes, parallel pipeline optimization

3. **Knapsack Problem (0/1 Knapsack)**

   - **Detailed Problem**: Given n items with weights w₁,...,wₙ and values v₁,...,vₙ, find the most valuable subset that fits in a knapsack of capacity W.
   - **Key Distinction**: Items cannot be broken (either take an item completely or leave it)
   - **Recurrence Relation**:
     - Let K[i,w] = maximum value with first i items and capacity w
     - K[0,w] = 0 for all w (no items means no value)
     - K[i,0] = 0 for all i (zero capacity means no value)
     - For i = 1 to n:
       - For w = 1 to W:
         - If w_i > w: K[i,w] = K[i-1,w] (can't take item i)
         - Else: K[i,w] = max(K[i-1,w], K[i-1,w-w_i] + v_i) (maximum of not taking or taking item i)
   - **Bottom-Up Algorithm**:
     1. Create a 2D table K[0...n][0...W]
     2. Initialize first row and column to 0
     3. Fill the table using the recurrence relation
     4. Return K[n][W]
   - **Time Complexity**: O(nW), which is pseudo-polynomial (depends on the numeric value of W)
   - **Space Optimization**: Can be reduced to O(W) space by using only two rows

4. **Matrix Chain Multiplication**

   - **Detailed Problem**: Given a sequence of matrices A₁, A₂, ..., Aₙ, find the most efficient way to multiply them together.
   - **Context**: Matrix multiplication is associative but the order affects the number of scalar multiplications needed
   - **Example**: For matrices A(10×30), B(30×5), C(5×60), different parenthesization yields different costs:
     - (A×B)×C: (10×30×5) + (10×5×60) = 1,500 + 3,000 = 4,500 operations
     - A×(B×C): (30×5×60) + (10×30×60) = 9,000 + 18,000 = 27,000 operations
   - **Recurrence Relation**:
     - Let m[i,j] = minimum number of scalar multiplications needed to compute A_i × ... × A_j
     - m[i,i] = 0 (single matrix requires no multiplications)
     - For l = 2 to n (l is the chain length):
       - For i = 1 to n-l+1:
         - j = i+l-1
         - m[i,j] = min(m[i,k] + m[k+1,j] + p\_{i-1}×p_k×p_j) for all k from i to j-1
   - **Time and Space Complexity**: O(n³) time and O(n²) space
   - **Reconstruction**: Additional data structure needed to record the optimal split points for reconstructing the parenthesization

5. **Sum of Subset Problem**

   - **Detailed Problem**: Given a set of non-negative integers and a value sum, determine if there is a subset whose sum equals the given sum.
   - **Recurrence Relation**:
     - Let subset[i][j] = true if there is a subset of elements from set[0...i] with sum j
     - subset[i][0] = true for all i (empty subset has sum 0)
     - subset[0]set[0]] = true (first element can form a subset with itself)
     - For i = 1 to n-1:
       - For j = 1 to sum:
         - If j < set[i]: subset[i][j] = subset[i-1][j] (can't include current element)
         - Else: subset[i][j] = subset[i-1][j] OR subset[i-1]j-set[i]] (don't include OR include current element)
   - **Bottom-Up Algorithm**:
     1. Create a 2D boolean table subset[0...n][0...sum]
     2. Initialize first column to true (empty subset has sum 0)
     3. Initialize first row based on first element
     4. Fill the table using the recurrence relation
     5. Return subset[n-1][sum]
   - **Time and Space Complexity**: O(n×sum) for both
   - **Optimization**: Space can be reduced to O(sum) by using only one row and updating it in-place

6. **Longest Common Subsequence**

   - **Detailed Problem**: Given two sequences X and Y, find the longest subsequence present in both of them. A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.
   - **Example**: For X = "ABCBDAB" and Y = "BDCABA", the LCS is "BCBA" with length 4
   - **Recurrence Relation**:
     - Let LCS[i][j] = length of LCS of X[1...i] and Y[1...j]
     - LCS[i][0] = 0 for all i (any sequence with empty sequence has LCS 0)
     - LCS[0][j] = 0 for all j (empty sequence with any sequence has LCS 0)
     - For i = 1 to m:
       - For j = 1 to n:
         - If X[i] = Y[j]: LCS[i][j] = LCS[i-1][j-1] + 1 (characters match, extend LCS)
         - Else: LCS[i][j] = max(LCS[i-1][j], LCS[i][j-1]) (characters don't match, take best of excluding one character)
   - **Reconstructing the LCS**:
     1. Start from LCS[m][n]
     2. If X[i] = Y[j], include this character and move to LCS[i-1][j-1]
     3. Else, move to the larger of LCS[i-1][j] and LCS[i][j-1]
     4. Continue until reaching a cell with value 0
   - **Time and Space Complexity**: O(m×n) for both
   - **Applications**: DNA sequence alignment, file differencing, version control systems

7. **Floyd-Warshall Algorithm**

   - **Detailed Problem**: Find shortest paths between all pairs of vertices in a weighted directed graph.
   - **Algorithm Insight**: For vertices i and j, consider whether going through vertex k gives a shorter path than the current best path.
   - **Recurrence Relation**:
     - Let dist[i][j][k] = shortest path from i to j using vertices 0 to k as intermediate vertices
     - dist[i][j][0] = weight(i,j) if edge exists, otherwise infinity
     - For k = 1 to V-1:
       - For i = 0 to V-1:
         - For j = 0 to V-1:
           - dist[i][j][k] = min(dist[i][j][k-1], dist[i][k][k-1] + dist[k][j][k-1])
   - **Implementation Optimization**: Can use a single 2D array and update it in place:
     - For k = 0 to V-1:
       - For i = 0 to V-1:
         - For j = 0 to V-1:
           - dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
   - **Time and Space Complexity**: O(V³) time and O(V²) space
   - **Detecting Negative Cycles**: If dist[i][i] < 0 for any vertex i, the graph contains a negative cycle
   - **Applications**: Network routing, finding transitive closure of graphs

8. **Bellman-Ford Algorithm**
   - **Detailed Problem**: Find shortest paths from a source vertex to all other vertices in a weighted directed graph, allowing negative weight edges.
   - **Key Advantage Over Dijkstra**: Can handle negative edge weights and detect negative cycles
   - **Algorithm Steps**:
     1. Initialize distances: source = 0, all others = infinity
     2. Relax all edges V-1 times:
        a. For each edge (u,v) with weight w:
        - If dist[u] + w < dist[v]:
          - Update dist[v] = dist[u] + w
     3. Check for negative weight cycles:
        a. For each edge (u,v) with weight w:
        - If dist[u] + w < dist[v]:
          - Graph contains a negative weight cycle
   - **Time Complexity**: O(V×E) where V is the number of vertices and E is the number of edges
   - **Space Complexity**: O(V) for the distance array
   - **Applications**: Network routing protocols like RIP (Routing Information Protocol)

---

## **Searching Algorithms and String Matching Algorithms**

### Branch and Bound

Branch and Bound is a algorithmic paradigm for solving optimization problems by systematically enumerating all candidate solutions and discarding large subsets of fruitless candidates:

- **Core Principles**:

  - **Branching**: Divide the search space into smaller subproblems (branches)
  - **Bounding**: Calculate bounds on the best possible solution within a branch
  - **Pruning**: Eliminate branches that cannot contain the optimal solution

- **Key Components**:
  - **State Space Tree**: Represents all possible solutions
  - **Bounding Function**: Estimates the best possible solution within a branch
  - **Selection Strategy**: Determines which node to explore next (e.g., best-first, depth-first)

1. **The Assignment Problem**

   - **Detailed Problem**: Assign n workers to n jobs to minimize total cost, where each worker can do exactly one job and each job needs exactly one worker.
   - **Hungarian Algorithm Steps**:
     1. **Row Reduction**: Subtract the minimum value in each row from all elements in that row
     2. **Column Reduction**: Subtract the minimum value in each column from all elements in that column
     3. **Cover all zeros with minimum lines**:
        a. Find minimum number of horizontal and vertical lines to cover all zeros
        b. If number of lines equals n, optimal assignment found
        c. Otherwise, go to step 4
     4. **Create additional zeros**:
        a. Find smallest uncovered element
        b. Subtract it from all uncovered elements
        c. Add it to elements covered by two lines
        d. Return to step 3
     5. **Make the assignment**: Select zeros such that each row and column has exactly one selected zero
   - **Time Complexity**: O(n³) in the worst case
   - **Applications**: Resource allocation, job scheduling, personnel assignment

2. **The Knapsack Problem (Branch and Bound Approach)**
   - **Detailed Problem**: Same as the 0/1 Knapsack problem, but solved using branch and bound.
   - **Upper Bound Calculation**: For a partial solution, compute an upper bound by:
     1. Including all items that fit completely
     2. Adding a fraction of the next item (using the Fractional Knapsack approach)
   - **Branch and Bound Strategy**:
     1. Start with an empty solution and calculate its upper bound
     2. Create two branches: include or exclude the first item
     3. Calculate upper bounds for both branches
     4. Explore the branch with the higher upper bound first (best-first search)
     5. Keep track of the best complete solution found so far
     6. Prune branches whose upper bound is worse than the current best solution
   - **Advantages over Dynamic Programming**:
     - Can avoid exploring the entire solution space
     - Often faster for large inputs
     - Can handle additional constraints more flexibly
   - **Time Complexity**: Exponential in the worst case, but typically much better in practice due to pruning

### String Matching Algorithms

1. **The Naïve String-Matching Algorithm**

   - **Detailed Problem**: Find all occurrences of a pattern P[1...m] in a text T[1...n].
   - **Algorithm Details**:
     1. For each possible starting position i in T (from 1 to n-m+1):
        a. Check if P[1...m] matches T[i...i+m-1]
        b. If matches completely, report occurrence at position i
   - **Implementation Optimization**: Early termination when a mismatch is found
   - **Time Complexity**:
     - Worst case: O((n-m+1)×m) ≈ O(n×m) when many partial matches occur
     - Best case: O(n) when immediate mismatches occur at first character
   - **Space Complexity**: O(1) (excluding input storage)
   - **When to Use**: Simple to implement, works well for small patterns or when pattern rarely occurs in text

2. **The Rabin-Karp Algorithm**

   - **Detailed Problem**: Same as naïve matching, but uses hashing to speed up the search.
   - **Key Insight**: Use rolling hash to efficiently compare pattern with text windows
   - **Algorithm Details**:
     1. Compute hash value h(P) for the pattern
     2. Compute hash value h(T[1...m]) for the first m characters of the text
     3. For each position i from 1 to n-m+1:
        a. If h(P) = h(T[i...i+m-1]), check character by character for an actual match
        b. If i < n-m+1, compute h(T[i+1...i+m]) from h(T[i...i+m-1]) using the rolling hash property
   - **Rolling Hash Function**:
     - Typically uses polynomial hash: h(s) = s[0]×p^(m-1) + s[1]×p^(m-2) + ... + s[m-1] mod q
     - Where p is a prime number (typically 31 or 101) and q is a large prime
     - Rolling update: h(T[i+1...i+m]) = (h(T[i...i+m-1]) - T[i]×p^(m-1))×p + T[i+m] mod q
   - **Time Complexity**:
     - Average case: O(n+m) when few hash collisions occur
     - Worst case: O(n×m) when many hash collisions occur (can be made rare with good hash function)
   - **Applications**: Plagiarism detection, DNA sequence matching, file fingerprinting

3. **KMP Algorithm (Knuth-Morris-Pratt)**
   - **Detailed Problem**: Same as naïve matching, but with preprocessing to avoid redundant comparisons.
   - **Key Insight**: When a mismatch occurs, we already know part of the text, so we can skip some comparisons
   - **Preprocessing: Building the Prefix Table**:
     1. Compute LPS (Longest Proper Prefix which is also Suffix) array for the pattern
     2. LPS[i] = length of the longest proper prefix of P[0...i] which is also a suffix of P[0...i]
     3. Use this information to determine how many characters to skip when a mismatch occurs
   - **Matching Algorithm**:
     1. Preprocess pattern to build LPS array
     2. Initialize pattern index j = 0 and text index i = 0
     3. While i < n:
        a. If P[j] matches T[i], increment both i and j
        b. If j = m, report match at position i-m+1 and set j = LPS[j-1]
        c. If i < n and P[j] doesn't match T[i]:
        - If j > 0, set j = LPS[j-1]
        - Else, increment i
   - **Time Complexity**: O(n+m) in all cases
   - **Space Complexity**: O(m) for the LPS array
   - **Applications**: Text editors (find/replace), intrusion detection systems, genomic sequence alignment
