# Python Cheat Sheet for Coding Interviews

## Data Structures
- [Lists](#lists)
- [Dictionaries](#dictionaries)
- [Sets](#sets)
- [Tuples](#tuples)
- [Deque (Double-ended Queue)](#deque-double-ended-queue)
- [Heaps (Priority Queue)](#heaps-priority-queue)

## Common Algorithms & Techniques
- [Sorting](#sorting)
- [Binary Search](#binary-search)
- [Two Pointers](#two-pointers)
- [Sliding Window](#sliding-window)
- [Graph Traversals](#graph-traversals)
  - [BFS (Breadth-First Search)](#bfs-breadth-first-search)
  - [DFS (Depth-First Search)](#dfs-depth-first-search)
- [Dynamic Programming](#dynamic-programming)
  - [Memoization (Top-down)](#memoization-top-down)
  - [Tabulation (Bottom-up)](#tabulation-bottom-up)

## Time Complexity Reference

## Python Tips & Tricks
- [One-liners](#one-liners)
- [Constants and Infinity](#constants-and-infinity)
- [Built-ins for Interviews](#built-ins-for-interviews)
- [Common Regex Patterns](#common-regex-patterns)
- [Debugging Helpers](#debugging-helpers)

## Data Structures

### Lists
```python
# Creation
my_list = []
my_list = [1, 2, 3, 4, 5]
my_list = list(range(5))  # [0, 1, 2, 3, 4]

# Operations
my_list.append(6)         # Add to end: [1, 2, 3, 4, 5, 6]
my_list.insert(0, 0)      # Insert at index: [0, 1, 2, 3, 4, 5, 6]
my_list.pop()             # Remove and return last item: 6
my_list.pop(0)            # Remove at index: 0
my_list.remove(4)         # Remove first occurrence: [1, 2, 3, 5]
del my_list[1]            # Delete at index: [1, 3, 5]

# Slicing
my_list[start:end:step]   # start inclusive, end exclusive
my_list[::-1]             # Reverse list
my_list[2:]               # From index 2 to end
my_list[:3]               # From start to index 3 (exclusive)

# List comprehension
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Dictionaries
```python
# Creation
my_dict = {}
my_dict = {'key1': 'value1', 'key2': 'value2'}
my_dict = dict(key1='value1', key2='value2')

# Operations
my_dict['key3'] = 'value3'    # Add or update
my_dict.get('key4', 'default')  # Get with default
my_dict.pop('key1')           # Remove and return
my_dict.keys()                # View of keys
my_dict.values()              # View of values
my_dict.items()               # View of (key, value) pairs

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}
```

### Sets
```python
# Creation
my_set = set()
my_set = {1, 2, 3, 4, 5}

# Operations
my_set.add(6)               # Add element
my_set.remove(1)            # Remove element (raises error if not present)
my_set.discard(10)          # Remove if present (no error if absent)
my_set.pop()                # Remove and return arbitrary element

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1 | set2                 # Union: {1, 2, 3, 4, 5}
set1 & set2                 # Intersection: {3}
set1 - set2                 # Difference: {1, 2}
set1 ^ set2                 # Symmetric difference: {1, 2, 4, 5}
```

### Tuples
```python
# Creation (immutable)
my_tuple = ()
my_tuple = (1, 2, 3)
my_tuple = 1, 2, 3          # Parentheses optional

# Single element tuple needs trailing comma
single_tuple = (1,)
```

### Deque (Double-ended Queue)
```python
from collections import deque

# Creation
my_deque = deque([1, 2, 3])

# Operations (O(1) complexity)
my_deque.append(4)          # Add to right
my_deque.appendleft(0)      # Add to left
my_deque.pop()              # Remove from right
my_deque.popleft()          # Remove from left
```

### Heaps (Priority Queue)
```python
import heapq

# Create a min heap
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

# Pop smallest element
smallest = heapq.heappop(heap)  # 1

# Convert list to heap in-place
numbers = [3, 1, 5, 2, 4]
heapq.heapify(numbers)      # Now a min heap

# For max heap, negate values
max_heap = []
heapq.heappush(max_heap, -3)
max_val = -heapq.heappop(max_heap)  # 3
```

## Common Algorithms & Techniques

### Sorting
```python
# Built-in sort (Timsort: O(n log n))
sorted_list = sorted(my_list)
sorted_list = sorted(my_list, reverse=True)
sorted_list = sorted(my_list, key=lambda x: len(x))  # Sort by length

# In-place sort
my_list.sort()
my_list.sort(reverse=True)
my_list.sort(key=lambda x: len(x))
```

### Binary Search
```python
# On sorted list: O(log n)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found
```

### Two Pointers
```python
# Example: Find pair that sums to target in sorted array
def find_pair(arr, target):
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return [-1, -1]  # No pair found
```

### Sliding Window
```python
# Example: Find max sum subarray of size k
def max_subarray_sum(arr, k):
    n = len(arr)
    if n < k:
        return -1
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, n):
        window_sum = window_sum + arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### Graph Traversals

#### BFS (Breadth-First Search)
```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
```

#### DFS (Depth-First Search)
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    
    return result

# Iterative DFS
def dfs_iterative(graph, start):
    visited = set([start])
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        result.append(node)
        
        # Add neighbors in reverse order to maintain same order as recursive
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    
    return result
```

### Dynamic Programming

#### Memoization (Top-down)
```python
# Example: Fibonacci with memoization
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

#### Tabulation (Bottom-up)
```python
# Example: Fibonacci with tabulation
def fib_tab(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

## Time Complexity Reference

| Data Structure/Algorithm | Average Time Complexity |
|-------------------------|------------------------|
| Array/List Access       | O(1)                   |
| Array/List Search       | O(n)                   |
| Array/List Insertion    | O(n)                   |
| Array/List Deletion     | O(n)                   |
| Dictionary/Hash Access  | O(1)                   |
| Dictionary/Hash Search  | O(1)                   |
| Dictionary/Hash Insertion| O(1)                 |
| Dictionary/Hash Deletion| O(1)                  |
| Binary Search           | O(log n)               |
| Quicksort               | O(n log n)             |
| Mergesort               | O(n log n)             |
| Breadth-First Search    | O(V + E)               |
| Depth-First Search      | O(V + E)               |
| Dijkstra's Algorithm    | O((V + E) log V)       |

## Python Tips & Tricks

### One-liners
```python
# Swap values
a, b = b, a

# Flatten list of lists
flattened = [item for sublist in nested_list for item in sublist]

# Get frequency count
from collections import Counter
frequencies = Counter([1, 2, 2, 3, 3, 3])  # Counter({3: 3, 2: 2, 1: 1})

# Find most common element
most_common = max(set(my_list), key=my_list.count)

# Check if all/any elements satisfy condition
all_positive = all(x > 0 for x in numbers)
any_positive = any(x > 0 for x in numbers)
```

### Constants and Infinity
```python
# Python's infinity
float('inf')   # Positive infinity
float('-inf')  # Negative infinity

# Common numeric operations
max_val = max(my_list)
min_val = min(my_list)
sum_val = sum(my_list)
```

### Built-ins for Interviews
```python
# Useful built-in functions
enumerate(iterable)            # Returns (index, value) pairs
zip(list1, list2)              # Combines multiple iterables
map(function, iterable)        # Applies function to each element
filter(function, iterable)     # Filters elements by function
range(start, stop, step)       # Creates a sequence of numbers

# String operations
my_string.strip()              # Remove whitespace from ends
my_string.split(',')           # Split by delimiter
','.join(['a', 'b', 'c'])      # Join list into string
my_string.lower()/upper()      # Convert case
```

### Common Regex Patterns
```python
import re

# Basic patterns
re.search(r'\d+', text)        # Find digits
re.findall(r'[a-zA-Z]+', text) # Find words
re.sub(r'\s+', ' ', text)      # Replace multiple spaces with one
```

### Debugging Helpers
```python
# Print with labeled debug info
print(f"Debug - variable: {variable}, type: {type(variable)}")

# Time execution
import time
start = time.time()
# ... code to time ...
end = time.time()
print(f"Execution time: {end - start} seconds")
```
