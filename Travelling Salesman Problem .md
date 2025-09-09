# Ex. No: 18D - Travelling Salesman Problem (TSP)

## AIM:
To write a Python program to find the shortest possible route that visits every city exactly once and returns to the starting point using the **Travelling Salesman Problem (TSP)** approach.

## ALGORITHM:

**Step 1**: Start the program.

**Step 2**: Input the number of cities and the distance matrix.

**Step 3**: Set the starting city (e.g., city `0`).

**Step 4**: Generate all possible permutations of the remaining cities.

**Step 5**: For each permutation:
- Calculate the total cost of traveling through the permutation starting and ending at city `0`.
- Keep track of the **minimum cost** and the corresponding route.

**Step 6**: Return the **route** and the **minimum cost**.

**Step 7**: End the program.

## PYTHON PROGRAM

```
name: kristipati shivani
reg.no:212222060126

vertex = []
for i in range(V):
	if i != s:
		vertex.append(i)
min_path=maxsize
next_permutation=permutations(vertex)
for i in next_permutation:
    


	current_pathweight = 0
	k=s
	for j in i:
	    current_pathweight+=graph[k][j]
	    k=j
	current_pathweight+=graph[k][s]
	min_path=min(min_path, current_pathweight)
return min_path
```

## OUTPUT
<img width="324" height="146" alt="image" src="https://github.com/user-attachments/assets/a925f495-bae3-4bfa-8487-feaa6302297b" />

## RESULT
Thus the program to find the shortest possible route using the Travelling Salesman Problem (TSP) approach has been implemented and executed successfully.

