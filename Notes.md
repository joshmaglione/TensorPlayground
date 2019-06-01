# An introduction with context

Joint with Pete Brooksbank and James Wilson. Funded by NSF

$\textsf{TensorSpace}$ refers to a collection of Magma packages:
* [TensorSpace](https://github.com/thetensor-space/TensorSpace): data structures for tensors,
* [Sylver](https://github.com/thetensor-space/Sylver): algorithms for simultaneous Sylvester equations,
* [Densor](https://github.com/thetensor-space/Densor): algorithms to construct linear closures of tensor spaces.

# Constructing Tensors

**Goal:** Want flexible definition for tensors. 

**Examples:**

1. $V = K^d$ a finite $K$-vector space, and $b$ bilinear form on $V$:
$$$
b : V \times V \rightarrowtail K
$$$

2. $\psi$ an entangled system of qubits. Interpret:
$$$
\psi : \mathbb{C}^2\times \cdots \times \mathbb{C}^2 \rightarrowtail \mathbb{C}
$$$

3. $*$  an $A$-module action. Interpret:
$$$
* : V \times A \rightarrowtail V
$$$

4. $T$ a high-dimensional grid of bits. Interpret:
$$$
T : \mathbb{F}_2^{d_n} \times \cdots \times \mathbb{F}_2^{d_1} \rightarrowtail \mathbb{F}_2^{d_0}
$$$

Tensors in $\textsf{TensorSpace}$ require three things:
1. a *frame*,
2. a function,
3. a *category*.

If a category is not specified, then one is used by default. For now, we do not specify, but we will come back to it.

Essentially, we require the function to map a tuple of vectors to a vector. Something like the following. 
```
mult := func< x | x[1]*x[2] >;         // <x1, x2> |-> x1*x2
```

A *frame* is an ordered list of $K$-modules (here, just vector spaces). 
```
Q := Rationals();           
A := MatrixAlgebra(Q, 2);           // 2 x 2 matrices over Q
frame := [A, A, A];         
```

From these data, we can construct the tensor given by matrix multiplication in $\text{Mat}_{2\times 2}(\mathbb{Q})$. 
```
t := Tensor(frame, mult);
t;
```

This is the most general construction. 

## Tensors as multilinear maps

Let's do a brief sanity check, and verify `t` does what we expect.
```
X := A![2, 2/5, 0, -1];
Y := A![0, -3, 5, 7]; 
X, Y;
```
A quick test:
```
<X, Y> @ t;
X*Y;
```

Recall, the print statement for `t`:
```
t;
```
It states vector spaces, but we gave it matrices: `X` and `Y`. The frame gives context, so that we leaving the typing to the machines. 
```
<X, [0, -3, 5, 7]> @ t;
```

## Structure constants 

We define a tensor of the form 
$$$
\langle t| : K^{d_n} \times \cdots \times K^{d_1} \rightarrowtail K^{d_0}
$$$
by $d_0\cdots d_n$ elements in $K$: a $(d_n\times \cdots \times d_0)$-grid 
$$$
[t_{i_n\cdots i_1}^{j_0}].
$$$

Notational challenge to do this for general $n$. We do $n=2$:
$$$
(u, v)\mapsto \left( 
		\sum_{i=1}^{d_2}\sum_{j=1}^{d_1} u_{i}t_{ij}^{1} v_{j},
		\ldots,
		\sum_{i=1}^{d_2}\sum_{j=1}^{d_1} u_{i}t_{ij}^{d_0} v_{j}\right)\in K^{d_0}.
$$$


Now, let's construct an example in $\textsf{TensorSpace}$.
```
d := [3, 2, 5, 4];
grid := [1..3*2*5*4];
t := Tensor(Q, d, grid);
t;
```

We verify that the structure constants of `t` are what we expect.
```
StructureConstants(t)[1..5];         // Only first 5 entries
```

The grid model allows for easier understanding of data manipulation. 

We describe two operations on tensors that potentially change the frame.

#### Slicing

We grab any entry of the grid  $[t_{i_n\cdots i_1}^{j_0}]$ and consider the corresponding sub-grid, which can be interpreted as a tensor itself.
```
ind := [{3}, {2}, {5}, {4}];                  // last entry
Slice(t, ind);
```

Or any sequence of entries.
```
indices := [{1, 3}, {2}, {1, 3, 4}, {4}];     // 12 entries
Slice(t, indices);
```
Interpret as a matrix.
```
SliceAsMatrices(t, indices, 3, 1);
```

#### Shuffling 

We apply permutations to the frame. Suppose 
$$$
t : U_n \times \cdots \times U_1\rightarrowtail U_0.
$$$
is a tensor on finite-dimensional vector spaces. For a permutation $\sigma$ of $\{0, \ldots, n\}$, a $\sigma$-shuffle of $t$ is a tensor
$$$
t : U_{\sigma(n)} \times \cdots \times U_{\sigma(1)} \rightarrowtail U_{\sigma(0)}.
$$$
*Note*: shuffling requires dual-spaces. $\textsf{TensorSpace}$ handles this.

```
sigma := [2, 0, 3, 1];                // cycle: (0, 2, 3, 1)
s := Shuffle(t, sigma);
s;
```
We can take a look at the structure constants.
```
StructureConstants(s)[1..5];              // first 5 entries
```

## Algebras associated to tensors

These definitions apply in greater generality, but we fix a $K$-bilinear map $t : U\times V\rightarrowtail W$. 

Set $\Omega = \text{End}(U)\times \text{End}(V)\times \text{End}(W)$.

The first algebra we define is the *centroid*, and the second is the *derivation algebra*.
$$$
\begin{aligned}
\mathcal{C}_t &= \{ (X, Y, Z) \in\Omega \mid \forall u, v, Z\langle t | u, v\rangle = \langle t | Xu, v\rangle = \langle t | u, Yv\rangle \} \\
\mathcal{D}_t &= \{ (X, Y, Z) \in \Omega \mid \forall u, v, Z\langle t | u, v\rangle = \langle t | Xu, v\rangle + \langle t | u, Yv\rangle \}
\end{aligned}
$$$

**Fact:** $\mathcal{C}_t$ is a $K$-algebra with $1$, and $\mathcal{D}_t$ is a Lie algebra. 

#### GHK and W example

With these algebras, we can easily distinguish between two different quantum states. The Greenberger$-$Horne$-$Zeilinger (GHZ) state is 
$$$
\begin{aligned}
\sqrt{2} \cdot GHZ &= |000\rangle + |111\rangle \\
&\equiv (e_1\otimes e_1\otimes e_1) + (e_2\otimes e_2\otimes e_2) \\
&\equiv \begin{bmatrix}
(1,0) & (0,0) \\ (0,0) & (0,1)
\end{bmatrix}.
\end{aligned}
$$$

The W state is 
$$$
\begin{aligned}
\sqrt{3} \cdot W &= |001\rangle + |010\rangle + |100\rangle \\
&\equiv (e_1\otimes e_1\otimes e_2) + (e_1\otimes e_2\otimes e_1) + (e_2\otimes e_1\otimes e_1) \\
&\equiv \begin{bmatrix}
(0,1) & (1,0) \\ (1,0) & (0,0) 
\end{bmatrix}.
\end{aligned}
$$$

Both states are $(2\times 2\times 2)$-grids.

We interpret these states as 
$$$
\begin{aligned}
\langle GHZ| &: \mathbb{C}^2\times \mathbb{C}^2 \rightarrowtail \mathbb{C}^2 , &
\langle W| &: \mathbb{C}^2\times \mathbb{C}^2 \rightarrowtail \mathbb{C}^2
\end{aligned}
$$$
Typically interpreted as $\mathbb{C}^2 \times \mathbb{C}^2 \times \mathbb{C}^2 \rightarrowtail \mathbb{C}$.

**Question:** GHZ and W are contained in the same space. Maybe there is a change of basis (of each $\mathbb{C}^2$) that makes GHZ and W equivalent?

Let's construct the tensors (over $\mathbb{Q}$).
```
Q := Rationals();
d := [2, 2, 2];                       // dimensions of frame
GHZ := Tensor(Q, d, [1, 0, 0, 0, 0, 0, 0, 1]);
W := Tensor(Q, d, [0, 1, 1, 0, 1, 0, 0, 0]);
```

Compute their centroids.
```
C_GHZ := Centroid(GHZ);
C_W := Centroid(GHZ);
C_GHZ, C_W;
```
Let's play around with $\mathcal{C}_W$.
```
C1 := C_W.1;                   // 1st generator of the ring
C1;
```
We can get the individual blocks. 
```
X := C1 @ Induce(C_W, 2);                     // build map
Y := C1 @ Induce(C_W, 1);                     // for each
Z := C1 @ Induce(C_W, 0);                     // coordinate
X, Y, Z;
```

Build two vectors from our $2$-dimensional vector space. 
```
V := VectorSpace(Q, 2);
u := V![-1, 1/2];
v := V![1/2, 4/27];
u, v;
```
Verify the centroid satisfies the two equations.
```
<u*X, v> @ W eq <u, v*Y> @ W;
<u*X, v> @ W eq (<u, v> @ W)*Z;
```

Distinguish them by the existence of nilpotent elements. 
```
J_GHZ := JacobsonRadical(C_GHZ);
J_W := JacobsonRadical(C_W);
Dimension(J_GHZ), Dimension(J_W);
```
Because the dimension of `J_GHZ` is different from `J_W` these states are **inequivalent**.

With a little work, one shows that 
$$$
\begin{aligned}
  \mathcal{C}_{GHZ} &\cong \mathbb{C}^2, & \mathcal{C}_{W} &\cong \mathbb{C}[x]/\left(x^2\right).
\end{aligned}
$$$

The derivation algebras also do the trick. 
```
D_GHK := DerivationAlgebra(GHZ);
D_W := DerivationAlgebra(W);
Dimension(D_GHK), Dimension(D_W);
```

## What is really happening?

The equation $\langle t | Xu, v\rangle = \langle t | u, Yv\rangle$ becomes
$$$
XT^{(k)}_{**} - T^{(k)}_{**}Y = 0, 
$$$
where $T^{(k)}_{**}$ is the $k$th slice in the $0$-coordinate.

The equation $\langle t | Xu, v\rangle = Z\langle t | u, v\rangle$ becomes
$$$
XT^{*}_{*j} - T^{*}_{*j}Z = 0, 
$$$
where $T^{*}_{*j}$ is the $j$th slice in the $1$-coordinate.


These algorithms are part of a general family of Sylvester-like equations: $XA + BY = C$. 

We need to solve a simultaneous system of Sylvester-like equations. 

Naively, this requires $O(d^6)$ field operations for a $(d\times d\times d)$-grid. 

**Theorem (M.-Wilson 2018).** There is an algorithm to solve simultaneous Sylvester equations of a nondegenerate $(d\times d\times d)$-grid using $O(d^4)$ field operations.

**Open:** Extend this to general tensors of arbitrary valence and implement it.



# Tensor spaces and tensor categories

These algebras and general operators play an important role. A frame and a function do not give the full picture. 

Tensor spaces are the $K$-modules that contain tensors. Therefore, they are defined similarly to tensors. A tensor space $T$ is a $K$-module with 
1. a frame, 
2. an interpreter map $\langle \cdot | : T\rightarrow \left( U_n\times \cdots \times U_1\rightarrowtail U_0\right)$,
3. a category,
4. a ($K$-)basis.

The map $\langle \cdot |$ gives every $t\in T$ a multilinear map interpretation 
$$$
\langle t | : U_n\times \cdots \times U_1\rightarrowtail U_0.
$$$

### Tensor categories

Context hints us towards certain constraints, and we have already seen different examples. 

#### Examples

1. Equivalence up to a change of bases. For all $X, Y, Z\in \textsf{GL}_2(\mathbb{C})$, there exists $u,v\in\mathbb{C}^2$ such that 
$$$
\begin{aligned}
Z\langle GHZ | Xu, Yv\rangle \ne \langle W | u, v \rangle.
\end{aligned}
$$$

2. Equivalence of algebras: an isomorphism. There exists $\varphi\in\textsf{GL}(A)$ such that for all $a,b\in A$
$$$
\begin{aligned}
\varphi(ab) &= \varphi(a)\varphi(b).
\end{aligned}
$$$

3. Adjoint operators of a bilinear form $\langle\,,\,\rangle$. For $u,v\in V$, 
$$$
\begin{aligned}
\langle Au, v \rangle &= \langle u, A^*v\rangle.
\end{aligned}
$$$

For us a *tensor category* is 
1. a function $\textsf{A}: [n] \rightarrow \{-1, 0, 1\}$, and
2. a partition of $[n]$.

The function $\textsf{A}$ tells us which way the arrows go: $\downarrow$, $\parallel$, or $\uparrow$.

The partition tells us which coordinates are treated as equal.

Let's look at an example from algebra.
```
A := MatrixAlgebra(Q, 2);
t := Tensor(A);
TensorCategory(t);
```

Let's look at an example as a high-dimensional grid.
```
t := Tensor(GF(2), [2, 3, 4], [1..24]);
TensorCategory(t);                       // default category
```

Changing the tensor category changes the operators we allow. 


## Relating tensor spaces and operators

As seen in the GHZ and W-state example, their corresponding tensors were not equivalent because algebras we constructed were not isomorphic.

For these kinds of questions, it is helpful to consider the tensor subspace of all tensors sharing common operators. 

We will define it for a specific set of operators. Suppose $T$ is a tensor space with interpreter map $\langle \cdot |: T\rightarrow (U\times V \rightarrowtail W)$. Recall the derivation algebra of $t\in T$,
$$$
\begin{aligned}
\mathcal{D}_t &= \{ (X, Y, Z) \in \Omega \mid \forall u, v, Z\langle t | u, v\rangle = \langle t | Xu, v\rangle + \langle t | u, Yv\rangle \}
\end{aligned}
$$$

For $t\in T$, the *universal densor subsapce* of $t$ is the tensor subspace
$$$
\textsf{Dens}(t) = \{ s\in T \mid \mathcal{D}_t \subseteq \mathcal{D}_s \}. 
$$$

#### Example: Lie algebras

Now consider the Lie algebra of $3\times 3$ trace zero matrices over a field $K$, denoted $\mathfrak{sl}_3$. The tensor given by the matrix commutator is the following.
```
K := GF(7);
sl3 := LieAlgebra("A2", K);
t := Tensor(sl3);
t; 
```
Note the tensor category.
```
TensorCategory(t);
```
Therefore, its derivation algebra acts in the same way in each coordinate.
Its densor subspace is easily computed.
```
Dens := UniversalDensorSpace(t);
Dens;
```
The dimension of the densor subspace is tiny compared to the ambient space.
```
T := Parent(t);                   // full tensor space of t
Dimension(T), Dimension(Dens);
```


#### Example: matrix multiplication

Consider the full tensor space $T$ framed by $\left(K^{12}, K^8, K^6\right)$. Then there exists $t\in T$ such that 
$$$
\begin{aligned}
\langle t| &: \mathbb{M}_{3\times 4} \times \mathbb{M}_{4\times 2} \rightarrowtail \mathbb{M}_{3\times 2}
\end{aligned}
$$$
is matrix multiplication. 
```
K := Rationals();
t := MatrixMultiplication(K, [3, 4, 2]);
t;
```

It is contained in a $24$-dimensional tensor space.
```
T := Parent(t);
T;
```

However, its densor subspace is small.
```
Dens := UniversalDensorSpace(t); 
Dens subset T;
Dens;
```

Understanding the derivation algebra of $t$ is enough to understand its symmetries.

### Densor subspaces

s

# Summary

We are still uncovering new algebraic data from tensors, and this is a snapshot of what we can do in $\textsf{TensorSpace}$. 

Context through the frame and tensor category give flexibility in the data structures.

These algebras provide general tools to different kinds of applications.