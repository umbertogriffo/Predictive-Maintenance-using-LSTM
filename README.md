# LSTMS for Predictive Maintenance
* Author: Umberto Griffo
* Twitter: @UmbertoGriffo

Regression models: How many more cycles an in-service engine will last before it fails?
## Trie definition

**Trie**[1] is an ordered tree data structure that uses strings as keys. It's an efficient information retrieval data structure that we can use to search a word in **O(M)** time, where **M** is maximum string length. However the penalty is on trie storage requirements.
A common application of a **trie** is storing a **predictive text** or **autocomplete dictionary**, such as found on a mobile telephone.
Such applications take advantage of a trie's ability to quickly search for, insert, and delete entries.

The following picture shows a trie with the keys "Joe", "John", "Johnny", "Johnny", "Jane", and "Jack"
<p align="center">
  <img src="https://github.com/umbertogriffo/Trie/blob/master/Trie_example.png" height="330" width="330" />
</p>


## Complexity (Average)

|Access|Search|Insertion|Deletion|String Similarity|
|----|----|----|----|----|
|O(k)|O(k)|O(k)|O(k)|O(k*n)|

where **k** is maximum string length and **n** is number of nodes in the trie


## References

- [1] Trie https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
- [2] Hackerrank challenge contacts https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
- [3] What is a trie? What is its importance and how it is implemented? https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#turbofan