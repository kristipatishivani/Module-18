# Ex. No: 18B - Kruskal's Minimum Spanning Tree (MST) Algorithm

## AIM:
To write a Python program for **Kruskal's algorithm** to find the Minimum Spanning Tree (MST) of a given connected, undirected, and weighted graph.

## ALGORITHM:

**Step 1**: Sort all the edges of the graph in non-decreasing order of their weights.

**Step 2**: Initialize the `parent[]` and `rank[]` arrays for each vertex to keep track of the disjoint sets.

**Step 3**: Iterate through the sorted edges and pick the smallest edge. Check whether including this edge will form a cycle using the union-find method:
- If the vertices of the edge belong to different sets, include it in the MST.
- Perform a union of these two sets.

**Step 4**: Repeat Step 3 until the MST contains exactly `V-1` edges.

**Step 5**: Print the edges included in the MST and the total minimum cost.

## PYTHON PROGRAM

```
name:kristipati shivani
reg.no:212222060126

from collections import defaultdict

# Class to represent a graph


class Graph:

	def __init__(self, vertices):
		self.V = vertices # No. of vertices
		self.graph = [] # default dictionary
		# to store graph

	# function to add an edge to graph
	def addEdge(self, u, v, w):
		self.graph.append([u, v, w])

	# A utility function to find set of an element i
	# (uses path compression technique)
	def find(self, parent, i):
		if parent[i] == i:
			return i
		return self.find(parent, parent[i])

	# A function that does union of two sets of x and y
	# (uses union by rank)
	def union(self, parent, rank, x, y):
		xroot = self.find(parent, x)
		yroot = self.find(parent, y)

		# Attach smaller rank tree under root of
		# high rank tree (Union by Rank)
		if rank[xroot] < rank[yroot]:
			parent[xroot] = yroot
		elif rank[xroot] > rank[yroot]:
			parent[yroot] = xroot

		# If ranks are same, then make one as root
		# and increment its rank by one
		else:
			parent[yroot] = xroot
			rank[xroot] += 1

	# The main function to construct MST using Kruskal's
		# algorithm
	def KruskalMST(self):

		result = [] # This will store the resultant MST
		
		# An index variable, used for sorted edges
		i = 0
		
		# An index variable, used for result[]
		e = 0

		# Step 1: Sort all the edges in
		# non-decreasing order of their
		# weight. If we are not allowed to change the
		# given graph, we can create a copy of graph
		self.graph = sorted(self.graph,
							key=lambda item: item[2])

		parent = []
		rank = []
		for node in range(self.V):
		    parent.append(node)
		    rank.append(0)
		while e<self.V-1:
		    u,v,w=self.graph[i]
		    i=i+1
		    x=self.find(parent,u)
		    y=self.find(parent,v)
		    if x!=y:
		        e=e+1
		        result.append([u,v,w])
		        self.union(parent,rank,x,y)
		minimumCost=0
		print("Edges in the constructed MST")
		for u,v,weight in result:
		    minimumCost+=weight
		    print("%d -- %d == %d" % (u,v,weight))
		print("Minimum Spanning Tree",minimumCost)    

# Driver code
g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)

# Function call
g.KruskalMST()

```

## OUTPUT
<img width="477" height="175" alt="image" src="https://github.com/user-attachments/assets/feceee04-27e8-459d-b26f-fdf9e53bcf7d" />

## RESULT
Thus the Python program for Kruskal's algorithm to find the Minimum Spanning Tree (MST) is implemented successfully.

