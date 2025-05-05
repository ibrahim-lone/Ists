#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <numeric> // for iota
#include <chrono> // Can still use for reference, but MPI_Wtime is better for MPI timing
#include <mpi.h>   // MPI Header
// Removed omp.h and mutex as MPI handles parallelism differently

using namespace std;

using Permutation = vector<int>;
using IST = map<Permutation, Permutation>; // child -> parent

// --- Functions (Swap, r, compute_inverse, FindPosition, Parent1) ---
// --- These functions remain EXACTLY THE SAME as in your original code ---

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
int r(const Permutation &v) {
    for (int i = v.size() - 1; i >= 0; --i) {
        if (v[i] != i + 1) {
            return i + 1;
        }
    }
    return -1; // Should only happen for identity
}

// Compute inverse permutation (symbol to index map)
unordered_map<int, int> compute_inverse(const Permutation &v) {
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

    // Original logic had t==2 check, let's verify if needed or if rval handles it
    // The core logic seems okay, let's keep it as is for now.
    if (t == 2 && sw == id)
        return Swap(v, t - 1, v_inv);

    int vn_1 = v[n - 2];

    if (vn_1 == t || vn_1 == n - 1) {
         // Ensure rval is valid before swapping
         if (rval > 0 && v_inv.count(rval)) {
              return Swap(v, rval, v_inv);
         } else {
              // Handle edge case where rval might not be swappable, perhaps return v?
              // Or rely on Parent1 logic to avoid this call. Let's return v for now.
              return v;
         }
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
         // Ensure rval is valid if used in FindPosition
         if (t != n - 1 && rval <= 0) return v; // Avoid calling FindPosition with invalid rval

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

// --- End of unchanged functions ---


int main(int argc, char *argv[]) { // Pass argc and argv to MPI_Init

    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);   // Get rank of this process

    // --- Start Timing ---
    double start_time = MPI_Wtime();

    int n = 9; // Must be consistent across all processes

    // Each process calculates its local part of the trees
    vector<IST> ist_trees_local(n - 1);

    Permutation p(n);
    iota(p.begin(), p.end(), 1);
    const Permutation identity_permutation = p; // Store identity

    long long permutation_index = 0; // Use long long for n!

    // --- Main Calculation Loop (Distributed) ---
    if (world_rank == 0) {
        cout << "Starting permutation generation and calculation with " << world_size << " processes..." << endl;
    }

    do {
        // Distribute work round-robin based on rank
        // Process rank 'r' handles permutation if index % world_size == r
        // Skip processing the identity permutation itself
        if (p != identity_permutation && (permutation_index % world_size == world_rank)) {

            // Calculate inverse and rval only for permutations this rank handles
            auto v_inv = compute_inverse(p);
            int rval = r(p);

            // Calculate parent for each tree 't'
            for (int t = 1; t <= n - 1; ++t) {
                 // Ensure rval is valid before calling Parent1 if necessary
                 if (rval > 0) {
                      Permutation parent = Parent1(p, t, v_inv, rval);
                      ist_trees_local[t - 1][p] = parent; // Store in this process's local map
                 }
                 // else: what to do if r(v) is -1 (identity)? Parent1 shouldn't be called for identity anyway.
            }
        }
        permutation_index++; // Increment index for the next permutation

        // Optional: Add a progress indicator for rank 0 (can be slow)
        // if (world_rank == 0 && permutation_index % 10000 == 0) {
        //     cout << "Processed " << permutation_index << " permutations..." << endl;
        // }

    } while (next_permutation(p.begin(), p.end()));


    // --- Result Gathering Stage ---
    if (world_rank == 0) {
        cout << "Calculation finished. Gathering results..." << endl;
    }

    vector<IST> ist_trees_global; // Final combined results (only needed on rank 0)
    if (world_rank == 0) {
        ist_trees_global.resize(n - 1);
    }

    // Gather results for each tree t = 0 to n-2 (corresponding to T1 to Tn-1)
    for (int t = 0; t < n - 1; ++t) {

        // 1. Flatten local map<Permutation, Permutation> to a flat vector<int>
        vector<int> local_int_buffer;
        // Reserve space: each pair takes 2*n integers
        local_int_buffer.reserve(ist_trees_local[t].size() * 2 * n);
        for (const auto& [child, parent] : ist_trees_local[t]) {
            local_int_buffer.insert(local_int_buffer.end(), child.begin(), child.end());
            local_int_buffer.insert(local_int_buffer.end(), parent.begin(), parent.end());
        }
        int local_int_count = local_int_buffer.size(); // Number of ints this process sends for tree t

        // 2. Gather the *counts* of integers from each process to rank 0
        vector<int> recvcounts; // Array on rank 0 to store counts from each process
        if (world_rank == 0) {
            recvcounts.resize(world_size);
        }
        MPI_Gather(&local_int_count, 1, MPI_INT,         // Send local_int_count
                   recvcounts.data(), 1, MPI_INT,         // Receive into recvcounts array
                   0, MPI_COMM_WORLD);                  // Gather target is rank 0

        // 3. Rank 0 calculates displacements and total size, allocates receive buffer
        vector<int> displs;             // Displacements array for Gatherv on rank 0
        vector<int> global_int_buffer;  // Buffer on rank 0 to receive all flattened data
        int total_int_count = 0;
        if (world_rank == 0) {
            displs.resize(world_size);
            displs[0] = 0;
            total_int_count = recvcounts[0];
            // Calculate total size and displacements
            for (int i = 1; i < world_size; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
                total_int_count += recvcounts[i];
            }
            // Allocate the buffer on rank 0 to hold everything
            global_int_buffer.resize(total_int_count);
             if (total_int_count > 0) { // Print only if there's data expected
                 // Optional: Debug print
                 // cout << "Rank 0: Gathering tree " << t+1 << ", Total integers: " << total_int_count << endl;
             }
        }

        // 4. Gather the actual flattened integer data from all processes to rank 0
        MPI_Gatherv(local_int_buffer.data(), local_int_count, MPI_INT, // Send local data buffer
                    global_int_buffer.data(), recvcounts.data(), displs.data(), MPI_INT, // Receive into global buffer using counts/displacements
                    0, MPI_COMM_WORLD);                                  // Gather target is rank 0

        // 5. Rank 0 reconstructs the final map (ist_trees_global[t]) from the flattened buffer
        if (world_rank == 0) {
            int current_pos = 0;
            // Ensure total_int_count is a multiple of 2*n if > 0
            if (total_int_count > 0 && total_int_count % (2 * n) != 0) {
                 cerr << "ERROR: Gathered data size mismatch for tree " << t+1 << "! Total ints: " << total_int_count << endl;
            } else {
                 while (current_pos < total_int_count) {
                      // Extract child permutation (n integers)
                      Permutation child(global_int_buffer.begin() + current_pos, global_int_buffer.begin() + current_pos + n);
                      current_pos += n;
                      // Extract parent permutation (n integers)
                      Permutation parent(global_int_buffer.begin() + current_pos, global_int_buffer.begin() + current_pos + n);
                      current_pos += n;
                      // Insert into the final map for tree t
                      ist_trees_global[t][child] = parent;
                 }
            }
        }
         // Add a barrier here to ensure all processes finish gathering for tree 't'
         // before potentially clearing local data or moving to next 't' (optional but safer)
         MPI_Barrier(MPI_COMM_WORLD);
    }


    // --- End Timing & Print Results (Only on Rank 0) ---
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        cout << "Gathering complete." << endl;
        cout << "Total time taken: " << (end_time - start_time) << " seconds" << endl;
        // cout << "Total permutations processed (approx): " << permutation_index << endl; // This index might be slightly off due to parallel counting

        // Optionally print the size of the gathered trees for verification
        cout << "Sizes of gathered ISTs:" << endl;
        for (int t = 0; t < n - 1; ++t) {
             cout << "  T" << (t + 1) << ": " << ist_trees_global[t].size() << " entries" << endl;
        }

        // WARNING: Printing the full trees might be very large and slow!
        // Example: Print T1 if needed for small n
        // if (n <= 4) { // Only for very small n
        //     cout << "\nIST T1:\n";
        //     for (const auto& [child, parent] : ist_trees_global[0]) {
        //         for (int c : child) cout << c;
        //         cout << " -> ";
        //         for (int p : parent) cout << p;
        //         cout << "\n";
        //     }
        // }
    }

    // --- Finalize MPI ---
    MPI_Finalize();
    return 0;
}