+++
title = 'Souffle Auto Index'
date = 2024-06-14T10:57:38-05:00
tags = ['datalog']
draft = true
[params]
    math = true
+++
<!-- # Auto Indexing in Datalog -->

So Datalog rules have a bunch of queries and involve a lot of joins over multiple relations. All this involves looking up values in the relations, for these queries to run in practical amount of time, you need indexes to make the look-ups faster and avoid linear scans of the tables.

So one of the best ways to run a datalog program well is to have a strong indexing plan, so that all the searches are covered by indexes. But Indexes cost a lot of memory and compute to build and maintain. Now, we need to find some middle ground, where we are leaving very little to no performance on the table, and be able to do it, with minimal number of indexes possible.

This is the paper, that sets up the [Minimum Index Selection Problem](https://souffle-lang.github.io/vldb19.html). The goal of the paper is, given a set of searches that are performed over a relation, figure out the minimum number of indices possible, such that every search is covered by an Index.

### How does the problem look like?
- **Input**
    - A relation R, has 3 columns that are being used in different searches, `x, y, z`.
    - The input for the problem, the searches being used on a relation, `{x}, {x,y}, {x,z}, {x, y, z}`
- **Problem**
    - Here, How do you get rid of some searches, Idea is, if you have a search \( S_{1} \)
```python
define foo():
    return 1
```
Well, Another snippet
```cpp
bool getBit(int num, int i) {
    return ((num & (1<<i)) != 0);
}
```
- Well, this is an image.
![joins](join.svg)


