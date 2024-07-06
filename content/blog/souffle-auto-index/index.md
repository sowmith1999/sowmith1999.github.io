+++
title = 'Souffle Auto Index'
date = 2024-06-14T10:57:38-05:00
tags = ['datalog']
draft = false
[params]
    math = true
+++
<!-- # Auto Indexing in Datalog -->
## Introduction
[Datalog](https://en.wikipedia.org/wiki/Datalog) rules are a bunch of relational queries, each involving of joins over multiple relations. Here is a simple example of a Datalog rule:

```prolog
path(x, y) :- path(x, z), edge(z, y).
```
The above query does Transitive closure over a graph, where `path` and `edge` are relations. The query is saying, if there is a path from `x` to `z` and there is an edge from `z` to `y`, then there is a path from `x` to `y`. This query involves a join over `path` and `edge` relations. [Larger programs](https://github.com/harp-lab/brouhaha/blob/master/analyze.slog) can have 100's of such queries, each involving joins over multiple relations.

For these queries to be performant, you need indexes to make the value look-ups faster and avoid linear scans of the tables.

Well, How do you go about creating the indexes?, Is there a general way that is optimal?. What considerations do you need to take into account when creating indexes?.

This is the paper, that sets up the [Minimum Index Selection Problem](https://souffle-lang.github.io/vldb19.html). The goal of the paper is, given a set of searches that are performed over a relation, figure out the minimum number of indices possible, such that every search is covered by an Index.

### How does a sample problem look like?
- **Input**
    - A relation R, has n columns, of those 3 are being used in different searches, `x, y, z`.
    - The input for the problem is the set of searches being used on a relation, `{x}, {x,y}, {x,z}, {x, y, z}`
    - To enable, performant look-up of these searches, you need to create indexes on the relation for these columns. Naively, at most 4 indexes are needed as there are 4 searches.
- **Problem**
    - Can you get away with fewer indexes without any linear scans of the table?
- **Observations/Note's**
    - "For example, the index \( \ell = x \prec y \prec z \) covers three primitive searches: \( S_1 = \sigma_{x=v_1} \), \( S_2 = \sigma_{x=v'_1, y=v'_2} \), and \( S_3 = \sigma_{x=v''_1, y=v''_2, z=v''_3} \)".
    - You can share indexes among searches, if the searches share a common prefix. For example, if you have searches `{x}, {x,y}`. You can create an index for \(x \prec y\) and use it for both searches.
    - Taking this a bit further, you can see, `{x}, {x, y}, {x, y, z}`, can share one index, \(x \prec y \prec z \) and `x,z` needs a separate index. Or another possibility is `x`, `x,z` and `x,z,y` share an index, and `x,y` a separate index.
    - Here you can intuitively see that you can get away with fewer indexes than the number of searches, by just finding the longest common prefixes among the searches.
- **Solution**
    - This finding the longest prefixes among the searches to cover all searches as to figure a minimum number of indexes is a simple definition of the Minimum Index Selection Problem(MISP).
    - As the paper reveals, this problem can be modelled as Minimum Chain Cover Problem(MCCP), which can be solved in polynomial time.
    - Hence, Our MISP too, can be solved in polynomial time.
    - As does the paper, we will look at the problem and the solution in more detail in the following sections.
## Details
### Definitions
- **Primitive Search** : A primitive search is like a SQL select statement that return tuples which satisfy a condition. For example, a equality check on a column, \( \sigma_{x=v} \) is a primitive search. 
- **Index** : An Index here refers to a clustered B-Tree index that covers a searach predicate. Index, \( \ell = x \prec y \prec z \) uses \( x \) followed by \( y \) followed by \( z \) as its key, and covers searches that share a common prefix with the index.
- **Search Chain** : "A sequence of \( k \) searches \(S_1, S_2, \ldots, S_k \) form a search chain if each search \( S_i \) is a proper subset of its immediate successor \( S_{i+1} \). As a result, all search in the same search chain can be covered by a single index."

### Content
Hopefully, I did a good job of explaining what, the problem we are trying to solve is and some intuition about how the paper looks to solve it.

#### Why not just create an index for each search?
- Well, you can, but that is super expensive both memory and compute, and you can do much better.

#### Why not just look through all the searches and figure the minimal set?
- Well, again you can, but this borders on not possible due to sheer number of possible combinations. The number of possible combinations is something like \( 2^{m!} \), where \( m \) is the number of columns used for searches over a relation.
- When you have \( m \) attributes involved, you have \( m! \) possible permutations of the attributes, and then, you have pick or not pick each of the \( m! \) permutations, hence the \( 2^{m!} \) possible minimal sets.


- Well, this is an image.
![joins](join.svg)


