#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <numeric>
#include <omp.h>
#include <mutex>
#include <mpi.h>
using namespace std;

using Permutation = vector<int>;
using IST = map<Permutation, Permutation>;

int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i)
        result *= i;
    return result;
}

// Lehmer code: index to permutation
Permutation index_to_permutation(int index, int n) {
    vector<int> fact(n);
    iota(fact.begin(), fact.end(), 1);
    for (int i = 1; i < n; ++i)
        fact[i] *= fact[i - 1];

    vector<int> nums(n);
    iota(nums.begin(), nums.end(), 1);
    Permutation result;
    int idx = index;

    for (int i = n - 1; i >= 0; --i) {
        int f = (i == 0) ? 1 : fact[i - 1];
        int pos = idx / f;
        result.push_back(nums[pos]);
        nums.erase(nums.begin() + pos);
        idx %= f;
    }

    return result;
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

// Swap, r, inverse map as in your original
Permutation Swap(const Permutation &v, int x, const unordered_map<int, int> &v_inv) {
    int i = v_inv.at(x);
    if (i + 1 >= v.size()) return v;
    Permutation p = v;
    swap(p[i], p[i + 1]);
    return p;
}

int r(const Permutation &v) {
    for (int i = v.size() - 1; i >= 0; --i)
        if (v[i] != i + 1) return i + 1;
    return -1;
}

unordered_map<int, int> compute_inverse(const Permutation &v) {
    unordered_map<int, int> inv;
    for (int i = 0; i < v.size(); ++i) inv[v[i]] = i;
    return inv;
}

Permutation FindPosition(const Permutation &v, int t, const unordered_map<int, int> &v_inv, int rval) {
    int n = v.size();
    Permutation id(n); iota(id.begin(), id.end(), 1);
    Permutation sw = Swap(v, t, v_inv);
    if (t == 2 && sw == id) return Swap(v, t - 1, v_inv);
    int vn_1 = v[n - 2];
    if (vn_1 == t || vn_1 == n - 1) return Swap(v, rval, v_inv);
    return sw;
}

Permutation Parent1(const Permutation &v, int t, const unordered_map<int, int> &v_inv, int rval) {
    int n = v.size();
    int vn = v[n - 1], vn_1 = v[n - 2];
    Permutation id(n); iota(id.begin(), id.end(), 1);
    if (vn == n) {
        return (t != n - 1) ? FindPosition(v, t, v_inv, rval) : Swap(v, vn_1, v_inv);
    } else {
        if (vn == n - 1 && vn_1 == n && Swap(v, n, v_inv) != id) {
            return (t == 1) ? Swap(v, n, v_inv) : Swap(v, t - 1, v_inv);
        } else {
            return (vn == t) ? Swap(v, n, v_inv) : Swap(v, t, v_inv);
        }
    }
}
vector<int> serializeIST(const IST& ist, int n) {
    vector<int> data;
    for (const auto& [child, parent] : ist) {
        data.insert(data.end(), child.begin(), child.end());
        data.insert(data.end(), parent.begin(), parent.end());
    }
    return data;
}

IST deserializeIST(const vector<int>& data, int n) {
    IST ist;
    for (size_t i = 0; i < data.size(); i += 2 * n) {
        Permutation child(data.begin() + i, data.begin() + i + n);
        Permutation parent(data.begin() + i + n, data.begin() + i + 2 * n);
        ist[child] = parent;
    }
    return ist;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();  
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int n = 9;
    const int total_perms = factorial(n);
    int chunk_size = total_perms / world_size;
    int start = rank * chunk_size;
    int end = (rank == world_size - 1) ? total_perms : start + chunk_size;

    vector<IST> local_ists(n - 1);
    vector<mutex> tree_mutexes(n - 1);

    #pragma omp parallel for
    for (int i = start; i < end; ++i) {
        Permutation v = index_to_permutation(i, n);
        Permutation base(n); iota(base.begin(), base.end(), 1);
        if (v == base) continue;
        auto v_inv = compute_inverse(v);
        int rval = r(v);
        for (int t = 1; t <= n - 1; ++t) {
            Permutation parent = Parent1(v, t, v_inv, rval);
            lock_guard<mutex> lock(tree_mutexes[t - 1]);
            local_ists[t - 1][v] = parent;
        }
    }

    // Optionally serialize and gather ISTs (not shown, as IST is complex to serialize in pure MPI)
    // Instead: output to separate files per rank for post-processing
    vector<vector<int>> all_serialized_ists(n - 1);
    vector<int> local_counts(n - 1);
    vector<int> recv_counts(n - 1);
    vector<int*> recv_buffers(n - 1);

    for (int t = 0; t < n - 1; ++t) {
    vector<int> serialized = serializeIST(local_ists[t], n);
    local_counts[t] = serialized.size();

    // Gather sizes to rank 0
    vector<int> all_sizes(world_size);
    MPI_Gather(&local_counts[t], 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_size = 0;
    vector<int> displs(world_size, 0);
    if (rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            displs[i] = total_size;
            total_size += all_sizes[i];
        }
        recv_buffers[t] = new int[total_size];
        recv_counts[t] = total_size;
    }

    MPI_Gatherv(serialized.data(), local_counts[t], MPI_INT,
                recv_buffers[t], all_sizes.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<int> combined(recv_buffers[t], recv_buffers[t] + recv_counts[t]);
        all_serialized_ists[t] = move(combined);
        delete[] recv_buffers[t];
    }
}

    if (rank == 0) {
     double end_time = MPI_Wtime();  // ✅ End timer
    cout << "\nTotal execution time: " << (end_time - start_time) << " seconds.\n";
    
/*
vector<IST> final_ists;
for (int t = 0; t < n - 1; ++t) {
    IST ist = deserializeIST(all_serialized_ists[t], n);
    final_ists.push_back(move(ist));
}

// Use the root permutation (identity)
Permutation root(n);
iota(root.begin(), root.end(), 1);

print_level_order_all_ists(final_ists, root);
*/
}

    MPI_Finalize();
    return 0;
}

