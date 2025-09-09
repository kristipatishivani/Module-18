# Ex. No: 18A - Prim's Minimum Spanning Tree (MST) Algorithm
# AIM
To write a Python program for Prim's Minimum Spanning Tree (MST) algorithm.

# ALGORITHM
Step 1: Initialize the key[] array to infinity, set the first vertex's key to 0, and create mstSet[] and parent[] arrays.

Step 2: Select the vertex with the smallest key value not yet included in mstSet.

Step 3: Add the selected vertex to mstSet.

Step 4: For all adjacent vertices:

If the edge weight is smaller than their current key value, and the vertex is not in mstSet, then: Update their key value Update their parent to the current vertex Step 5: Repeat Steps 2–4 until all vertices are included in the MST.

Step 6: Print the resulting Minimum Spanning Tree using the parent[] array.

# PYTHON PROGRAM
import sys 
class Graph():

	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)]
					for row in range(vertices)]

	
	def printMST(self, parent):
		print ("Edge   Weight")
		for i in range(1, self.V):
			print (parent[i], "-", i, "  ",self.graph[i][parent[i]])

	
	def minKey(self, key, mstSet):

		
		min = sys.maxsize

		for v in range(self.V):
			if key[v] < min and mstSet[v] == False:
				min = key[v]
				min_index = v

		return min_index

	
	def primMST(self):

	
		key = [sys.maxsize] * self.V
		parent = [None] * self.V 
		key[0] = 0
		mstSet = [False] * self.V

		parent[0] = -1 

		
		
		for cout in range(self.V):

			
			u = self.minKey(key, mstSet)

			
			mstSet[u] = True

		
			for v in range(self.V):

				
				if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
						key[v] = self.graph[u][v]
						parent[v] = u

		self.printMST(parent)
		
		

g = Graph(5)
g.graph = [ [0, 2, 0, 6, 0],
			[2, 0, 3, 8, 5],
			[0, 3, 0, 0, 7],
			[6, 8, 0, 0, 9],
			[0, 5, 7, 9, 0]]

g.primMST();
# OUTPUT
<img width="917" height="328" alt="image" src="https://github.com/user-attachments/assets/8a280a55-7b5a-4c8f-8aa3-5a699880d362" />

# RESULT
The program for Prim's Minimum Spanning Tree (MST) algorithm has been implemented and executed successfully.

# Ex. No: 18B - Kruskal's Minimum Spanning Tree (MST) Algorithm
# AIM
To write a Python program for Kruskal's algorithm to find the Minimum Spanning Tree (MST) of a given connected, undirected, and weighted graph.

# ALGORITHM
Step 1: Sort all the edges of the graph in non-decreasing order of their weights.

Step 2: Initialize the parent[] and rank[] arrays for each vertex to keep track of the disjoint sets.

Step 3: Iterate through the sorted edges and pick the smallest edge. Check whether including this edge will form a cycle using the union-find method:

If the vertices of the edge belong to different sets, include it in the MST. Perform a union of these two sets. Step 4: Repeat Step 3 until the MST contains exactly V-1 edges.

Step 5: Print the edges included in the MST and the total minimum cost.

# PYTHON PROGRAM
from collections import defaultdict
class Graph:

	def __init__(self, vertices):
		self.V = vertices # No. of vertices
		self.graph = [] # default dictionary
		# to store graph


	def addEdge(self, u, v, w):
		self.graph.append([u, v, w])


	def find(self, parent, i):
		if parent[i] == i:
			return i
		return self.find(parent, parent[i])


	def union(self, parent, rank, x, y):
		xroot = self.find(parent, x)
		yroot = self.find(parent, y)


		if rank[xroot] < rank[yroot]:
			parent[xroot] = yroot
		elif rank[xroot] > rank[yroot]:
			parent[yroot] = xroot


		else:
			parent[yroot] = xroot
			rank[xroot] += 1

	def KruskalMST(self):

		result = [] # This will store the resultant MST
		

		i = 0
		
		e = 0

		self.graph = sorted(self.graph,
							key=lambda item: item[2])

		parent = []
		rank = []
		for node in range(self.V):
		    parent.append(node)
		    rank.append(0)
	
		while e < self.V - 1:

			# Step 2: Pick the smallest edge and increment
			# the index for next iteration
			u, v, w = self.graph[i]
			i = i + 1
			x = self.find(parent, u)
			y = self.find(parent, v)
       
			      
			if x != y:
				e = e + 1
				result.append([u, v, w])
				self.union(parent, rank, x, y)
		

		minimumCost = 0
		print ("Edges in the constructed MST")
		for u, v, weight in result:
			minimumCost += weight
			print("%d -- %d == %d" % (u, v, weight))
		print("Minimum Spanning Tree" , minimumCost)

g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)

g.KruskalMST()
# OUTPUT
<img width="1035" height="348" alt="image" src="https://github.com/user-attachments/assets/b87a14f7-1e6b-46d4-b4f3-173e1286fb0b" />

# RESULT
Thus the program for Kruskal's algorithm to find the Minimum Spanning Tree (MST) has been implemented and executed successfully.

# Ex. No: 18C - Dijkstra's Single Source Shortest Path Algorithm
# AIM
To write a Python program for Dijkstra's single source shortest path algorithm.

# ALGORITHM
Step 1: Initialize a distance[] array with infinity for all vertices except the source, which is set to 0. Create a sptSet[] array (shortest path tree set) to keep track of vertices whose shortest distance from the source is finalized.

Step 2: Pick the vertex u with the minimum distance value from the set of vertices not yet processed.

Step 3: For every adjacent vertex v of the picked vertex u, if the current distance to v is greater than the distance to u plus the edge weight (u, v), then update the distance of v.

Step 4: Mark the vertex u as processed in sptSet.

Step 5: Repeat Steps 2–4 until all vertices are processed.

Step 6: Print the shortest distances from the source to all other vertices.

# PYTHON PROGRAM
import sys

class Graph():

	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)]
					for row in range(vertices)]

	def printSolution(self, dist):
		print("Vertex   Distance from Source")
		for node in range(self.V):
			print(node, "           ", dist[node])

	
	def minDistance(self, dist, sptSet):

	
		min = sys.maxsize

	
		for u in range(self.V):
			if dist[u] < min and sptSet[u] == False:
				min = dist[u]
				min_index = u

		return min_index

	def dijkstra(self, src):

		dist = [sys.maxsize] * self.V
		dist[src] = 0
		sptSet = [False] * self.V
		for cout in range(self.V):
		    x=self.minDistance(dist,sptSet)
		    sptSet[x]=True
		    for y in range(self.V):
		        if self.graph[x][y]>0  and sptSet[y]==False and dist[y]>dist[x]+self.graph[x][y]:
		            dist[y]=dist[x]+self.graph[x][y]

		

			

		self.printSolution(dist)


g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
		[4, 0, 8, 0, 0, 0, 0, 11, 0],
		[0, 8, 0, 7, 0, 4, 0, 0, 2],
		[0, 0, 7, 0, 9, 14, 0, 0, 0],
		[0, 0, 0, 9, 0, 10, 0, 0, 0],
		[0, 0, 4, 14, 10, 0, 2, 0, 0],
		[0, 0, 0, 0, 0, 2, 0, 1, 6],
		[8, 11, 0, 0, 0, 0, 1, 0, 7],
		[0, 0, 2, 0, 0, 0, 6, 7, 0]
		];

g.dijkstra(0);
# OUTPUT
<img width="1004" height="503" alt="image" src="https://github.com/user-attachments/assets/41691fa9-8766-4706-8b93-dd5edf575525" />

# RESULT
Thus the program for Dijkstra's single source shortest path algorithm has been implemented and executed successfully.

# Ex. No: 18D - Travelling Salesman Problem (TSP)
# AIM
To write a Python program to find the shortest possible route that visits every city exactly once and returns to the starting point using the Travelling Salesman Problem (TSP) approach.

# ALGORITHM
Step 1: Start the program.

Step 2: Input the number of cities and the distance matrix.

Step 3: Set the starting city (e.g., city 0).

Step 4: Generate all possible permutations of the remaining cities.

Step 5: For each permutation:

Calculate the total cost of traveling through the permutation starting and ending at city 0. Keep track of the minimum cost and the corresponding route. Step 6: Return the route and the minimum cost.

Step 7: End the program.

# PYTHON PROGRAM
from sys import maxsize
from itertools import permutations
V = 4
def travellingSalesmanProblem(graph, s):


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

		
if __name__ == "__main__":


	graph = [[0, 10, 15, 20], [10, 0, 35, 25],
			[15, 35, 0, 30], [20, 25, 30, 0]]
	s = int(input())
	print(travellingSalesmanProblem(graph, s))
# OUTPUT
<img width="804" height="215" alt="image" src="https://github.com/user-attachments/assets/57dd638d-1a7a-4a60-92ed-8fb5c5988929" />

# RESULT
Thus the program to find the shortest possible route using the Travelling Salesman Problem (TSP) approach has been implemented and executed successfully.

# Ex. No: 18E - Count the Number of Triangles in an Undirected Graph
# AIM
To write a Python program to count the number of triangles present in an undirected graph using matrix operations.

# ALGORITHM
Step 1: Initialize a matrix aux2 to store the square of the adjacency matrix (i.e., graph²). Also, initialize a matrix aux3 to store the cube of the adjacency matrix (i.e., graph³).

Step 2: Multiply the adjacency matrix with itself to compute aux2 = graph × graph.

Step 3: Multiply aux2 with the adjacency matrix again to compute aux3 = aux2 × graph.

Step 4: Compute the trace of the matrix aux3 (i.e., the sum of diagonal elements of the matrix).

Step 5: Divide the trace by 6 to get the number of triangles in the graph. (Each triangle is counted six times in the trace — twice per vertex and once per direction.)

Step 6: Return the result.

# PYTHON PROGRAM
def multiply(A, B, C):
	global V
	for i in range(V):
		for j in range(V):
			C[i][j] = 0
			for k in range(V):
				C[i][j] += A[i][k] * B[k][j]


def getTrace(graph):
	global V
	trace = 0
	for i in range(V):
		trace += graph[i][i]
	return trace

def triangleInGraph(graph):
	global V
	aux2 = [[None] * V for i in range(V)]
	aux3=[[None]*V for i in range(V)]
	for i in range(V):
		for j in range(V):
			aux2[i][j] = aux3[i][j] = 0
	multiply(graph, graph, aux2)
	multiply(graph,aux2,aux3)
  trace=getTrace(aux3)
	return trace // 6


V = int(input())
graph = [[0, 1, 1, 0],
		[1, 0, 1, 1],
		[1, 1, 0, 1],
		[0, 1, 1, 0]]

print("Total number of Triangle in Graph :",
					triangleInGraph(graph))

# OUTPUT
<img width="1261" height="241" alt="image" src="https://github.com/user-attachments/assets/e94613e6-bedf-4a7c-b17c-2dbc5dfad0bc" />

# RESULT
The program to count the number of triangles in an undirected graph has been implemented and executed successfully.
