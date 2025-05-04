#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <numeric> // for iota
#include <chrono>
#include <map>
#include <omp.h>
#include <mutex>
using namespace std;

using Permutation = vector<int>;

using IST = map<Permutation, Permutation>; // child -> parent

vector<IST> ist_trees; // one IST for each t




// Swap the position of symbol x in v with the next symbol
Permutation Swap(const Permutation &v, int x, const unordered_map<int, int> &v_inv) {
    
    int i = v_inv.at(x);
    if (i + 1 >= v.size()) 
    return v;
    Permutation p = v;
    swap(p[i], p[i + 1]);
    return p;
}



// Return the position of the first incorrect symbol from the right
// (i.e., the first symbol that is not in its correct position)
int r(const Permutation &v) {
    for (int i = v.size() - 1; i >= 0; --i) {
        if (v[i] != i + 1) {
            return i + 1;
        }
    }
    return -1;
}


// Compute inverse permutation (symbol to index map)
unordered_map<int, int> compute_inverse(const Permutation &v)
{
    unordered_map<int, int> inv;
    for (int i = 0; i < v.size(); ++i) {
        inv[v[i]] = i;
    }
    
    return inv;
}
Permutation FindPosition(const Permutation &v, int t, const unordered_map<int, int> &v_inv, int rval) {
    int n = v.size();
    Permutation id(n);
    iota(id.begin(), id.end(), 1);

    Permutation sw = Swap(v, t, v_inv);

    if (t == 2 && sw == id)
        return Swap(v, t - 1, v_inv);

    int vn_1 = v[n - 2];

    if (vn_1 == t || vn_1 == n - 1) {
        return Swap(v, rval, v_inv);
    }

    return sw;
}


Permutation Parent1(const Permutation &v, int t, const unordered_map<int, int> &v_inv, int rval) {
    int n = v.size();
    int vn = v[n - 1];
    int vn_1 = v[n - 2];

    Permutation id(n);
    iota(id.begin(), id.end(), 1);

    if (vn == n) {
        return (t != n - 1) ? FindPosition(v, t, v_inv, rval)
                           : Swap(v, vn_1, v_inv);
    } else {
        if (vn == n - 1 && vn_1 == n && Swap(v, n, v_inv) != id) {
            return (t == 1) ? Swap(v, n, v_inv)
                            : Swap(v, t - 1, v_inv);
        } else {
            return (vn == t) ? Swap(v, n, v_inv)
                             : Swap(v, t, v_inv);
        }
    }
}


void level_order_tree(const Permutation &root, const IST &tree) {
    map<Permutation, vector<Permutation>> children_map;

    // Build children map from parent-child pairs
    for (const auto &[child, parent] : tree) {
        children_map[parent].push_back(child);
    }

    // Sort children lexicographically for consistency
    for (auto &[parent, children] : children_map) {
        sort(children.begin(), children.end());
    }

    // BFS: Level-order traversal
    queue<Permutation> q;
    q.push(root);
    q.push(Permutation());  // Marker for end of level

    while (!q.empty()) {
        Permutation current = q.front();
        q.pop();

        if (current.empty()) {
            // End of current level; print a new line
            if (!q.empty()) {
                cout << "\n";
                q.push(Permutation());  // Add marker for next level
            }
        } else {
            // Print current permutation
            for (int val : current) {
                cout << val << " ";
            }
            cout << "| ";

            // Push children to queue
            for (const auto &child : children_map[current]) {
                q.push(child);
            }
        }
    }

    cout << "\n";
}


void print_level_order_all_ists(const vector<IST> &ist_trees, const Permutation &root) {
    for (int t = 0; t < ist_trees.size(); ++t) {
        cout << "\nLevel-order traversal of IST T" << (t + 1) << ":\n";
        level_order_tree(root, ist_trees[t]);
    }
}
void print_ists(const vector<IST> &ist_trees) {
    for (int t = 0; t < ist_trees.size(); ++t) {
        cout << "\nIST T" << (t + 1) << ":\n";
        for (const auto &[child, parent] : ist_trees[t]) {
            for (int c : child) cout << c;
            cout << " -> ";
            for (int p : parent) cout << p;
            cout << "\n";
        }
    }
}
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i)
        result *= i;
    return result;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    
    int n = 9;
    vector<Permutation> permutations;
    //permutations.reserve(factorial(n)); // Reserve space for all permutations
    Permutation base(n);
    iota(base.begin(), base.end(), 1); // fills base with 1..n

    do {
         
       
            permutations.push_back(base);
    } while (next_permutation(base.begin(), base.end()));


ist_trees.resize(n - 1); 
vector<std::mutex> tree_mutexes(n - 1);

#pragma omp parallel for num_threads(36)
for (int i = 0; i < permutations.size(); ++i) {
    if (permutations[i] == base) continue;

    const auto &v = permutations[i];
    auto v_inv = compute_inverse(v);     // ✅ Inverse cache
    int rval = r(v);                     // ✅ Precompute r(v)

    for (int t = 1; t <= n - 1; ++t) {
        Permutation parent = Parent1(v, t, v_inv, rval);  // ✅ Pass inverse & r
        std::lock_guard<std::mutex> lock(tree_mutexes[t - 1]);
        ist_trees[t - 1][v] = parent;
    }
}


auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;
std::cout << "Time taken: " << duration.count() << " seconds\n";
//print_ists(ist_trees); // Print all ISTs
//
//print_level_order_all_ists(ist_trees, base); // Print level-order traversal of all ISTs



    return 0;
}
