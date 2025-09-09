# Ex. No: 18E - Count the Number of Triangles in an Undirected Graph

## AIM:
To write a Python program to **count the number of triangles** present in an **undirected graph** using matrix operations.

## ALGORITHM:

**Step 1**: Initialize a matrix `aux2` to store the square of the adjacency matrix (i.e., `graph²`).  
Also, initialize a matrix `aux3` to store the cube of the adjacency matrix (i.e., `graph³`).

**Step 2**: Multiply the adjacency matrix with itself to compute `aux2 = graph × graph`.

**Step 3**: Multiply `aux2` with the adjacency matrix again to compute `aux3 = aux2 × graph`.

**Step 4**: Compute the **trace** of the matrix `aux3` (i.e., the sum of diagonal elements of the matrix).

**Step 5**: Divide the trace by **6** to get the number of triangles in the graph.  
*(Each triangle is counted six times in the trace — twice per vertex and once per direction.)*

**Step 6**: Return the result.

## PYTHON PROGRAM

```
Name - kristipati shivani
Reg No - 212222060126

def multiply(A, B, C):
    global V
    for i in range(V):
        for j in range(V):
            C[i][j] = 0
            for k in range(V):
                C[i][j] += A[i][k] * B[k][j]

def getTrace(graph):
    global V trace = 0
    for i in range(V):
        trace += graph[i][i] return trace

def triangleInGraph(graph):
    global V aux2 = [[None] * V
    for i in range(V)] aux3=[[None]*V for i in range(V)]
        for i in range(V):
          for j in range(V): aux2[i][j] = aux3[i][j] = 0      multiply(graph, graph, aux2) multiply(graph,aux2,aux3) trace=getTrace(aux3) return trace // 6

V = int(input()) graph = [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]

print("Total number of Triangle in Graph :", triangleInGraph(graph))
```

## OUTPUT
<img width="882" height="158" alt="image" src="https://github.com/user-attachments/assets/950745cd-0cd8-43cf-9a1c-f283c41d4117" />

## RESULT
The program to count the number of triangles in an undirected graph has been implemented and executed successfully.
