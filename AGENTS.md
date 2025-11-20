Deep-learning-aided dismantling of interdependent networks — Agent Guide
========================================================================

Scope
-----
- This file applies to all code under `code/` (including `MultiDismantler_unit_cost/`, `MultiDismantler_degree_cost/`, `MRGNN/`, and the training / evaluation scripts).
- It does **not** cover files one level above (`cuda_detect.py`, top-level `results/`); treat those as external utilities / outputs.

Project overview
----------------
- The repository implements the methods from the paper **“Deep-learning-aided dismantling of interdependent networks”**.
- It focuses on **multilayer (interdependent) networks** with two layers, and learns a **node removal policy** that dismantles the network efficiently.
- There are **two closely related optimisation problems**, implemented as two parallel codebases with shared structure:
  - `MultiDismantler_unit_cost/` — **Unit cost**: each removed node has cost 1.
  - `MultiDismantler_degree_cost/` — **Degree cost**: each removed node’s cost depends on its (normalised) degree in the two layers.
- Training and experiments are orchestrated via `run.sh` at `code/run.sh`.

High-level architecture
-----------------------
- **Environment & graph**
  - `graph.py`
    - Builds synthetic interdependent networks via `GMM.GMM(N)` (two linked layers).
    - Stores adjacency lists (`adj_list`), edge lists, and `max_rank`, the size of the **initial largest mutually connected component** (LMCC) computed by `Mcc.MCC`.
    - In `MultiDismantler_degree_cost`, also computes per-node **weights**:
      - `self.weights[0][i] = deg_layer1(i) / max_deg_layer1`
      - `self.weights[1][i] = deg_layer2(i) / max_deg_layer2`
  - `mvc_env.py`
    - Implements the **RL environment** (state, step function, reward) for dismantling.
    - Maintains:
      - `covered_set` — removed nodes.
      - `action_list` — sequence of removed nodes.
      - `remove_edge` — edges removed due to interdependencies.
      - `MaxCCList` — LMCC size history (for computing AUDC curves).
    - Uses `Mcc.MCC` and NetworkX graphs to recompute the current LMCC size.
- **Policy / value network**
  - `MultiDismantler_torch.py`
    - Wraps the environment in a **DQN-like** training loop with n-step returns and a Graph Neural Network backbone.
    - Uses `PrepareBatchGraph.PrepareBatchGraph` to build batched graph inputs for the GNN.
    - Uses `MRGNN/` (encoders, aggregators, attention) to implement multi-relational GraphSAGE-style embeddings.
    - Provides:
      - `Train()` — training on synthetic graphs.
      - `Test()` / `GetSol()` — synthetic graph evaluation and solution extraction.
      - `EvaluateRealData()` — real-network evaluation, generating:
        - Node removal sequences.
        - LMCC decline curves.
        - Cost curves (degree-cost version only).
- **Entry scripts**
  - `train.py` — seeds RNGs, instantiates `MultiDismantler`, calls `Train()`.
  - `testReal.py` — evaluates pre-trained models on real datasets via `EvaluateRealData`.
  - `testSynthetic.py` — evaluates on synthetic datasets.
  - `drawUnweight.py` / `drawWeight.py` — generate LMCC decline plots.
  - `run.sh` — central driver; select between unit-cost / degree-cost pipelines and between train/test modes.

Two optimisation problems
-------------------------

### 1. Unit cost dismantling (`MultiDismantler_unit_cost`)

**Goal**
- Dismantle the interdependent network (drive LMCC to collapse) with **as few node removals as possible**.
- Informal objective: minimise the **number of removed nodes** required to fragment the network, while being robust to network randomness.

**Reward definition (unit cost)**
- Implemented in `MultiDismantler_unit_cost/mvc_env.py:getReward`:
  - Let:
    - `rank_t` = LMCC size after taking action at time `t`.
    - `max_rank` = LMCC size in the original graph before any removals.
    - `N` = number of nodes.
  - Reward per step:
    - `r_t = - rank_t / (max_rank * N)`
  - Interpretation:
    - LMCC size is normalised by the initial LMCC and by graph size.
    - The agent is penalised proportionally to the current LMCC; long trajectories with slow LMCC decay produce more negative cumulative rewards.
    - **Cost of an action itself (node removal) is implicitly unit cost** — every step removes exactly one node, and the reward does not depend on node degree.

**Evaluation / cost metrics (unit cost)**
- `self.test_env.score` accumulates `-r_t` during `stepWithoutReward`.
- In `MultiDismantler_unit_cost.MultiDismantler`:
  - `Test()` returns:
    - `self.test_env.score + len(remaining_nodes) / (max_rank * N)`
    - The extra term treats remaining nodes as if they were removed at the end with uniform unit cost.
  - `GetSol()` returns `(score, sol, cost_value)` where:
    - `sol` — sequence of removed nodes.
    - `cost_value = len(sol) / N` — final **normalised node count** cost.
- `EvaluateRealData()`:
  - Runs `GetSolution()` multiple times, aggregates LMCC curves and AUDC statistics, and writes them to `results/unitcost/...`.

### 2. Degree cost dismantling (`MultiDismantler_degree_cost`)

**Goal**
- Dismantle the interdependent network with **minimal total removed degree**, i.e. preferentially remove low-degree nodes when possible, while still collapsing the LMCC.
- Each node has a **degree-based removal cost** derived from its degree in both layers.

**Node degree weights**
- Implemented in `MultiDismantler_degree_cost/graph.py:cal_degree`:
  - For each layer ℓ ∈ {1, 2}:
    - `deg_ℓ(i)` — degree of node `i` in layer ℓ.
    - `maxDeg_ℓ` — maximum degree over all nodes in that layer.
    - Normalised degree:
      - `w_ℓ(i) = deg_ℓ(i) / maxDeg_ℓ  ∈ [0, 1]`
  - Stored in:
    - `self.weights[0][i] = w_1(i)`
    - `self.weights[1][i] = w_2(i)`
- For Graphs built from real data (`Graph_test`), the same logic is used after computing `max_rank`.

**Reward definition (degree cost)**
- Implemented in `MultiDismantler_degree_cost/mvc_env.py:getReward`:
  - Let:
    - `rank_t` = LMCC size after removing node `a`.
    - `max_rank` = initial LMCC size.
    - `w̄(a)` = average **normalised degree weight** of node `a` across both layers:
      - `w̄(a) = ( w_1(a) / Σ_i w_1(i) + w_2(a) / Σ_i w_2(i) ) / 2`
  - Reward per step:
    - `r_t = - (rank_t / max_rank) * w̄(a)`
  - Function returns a pair: `(r_t, rank_t / max_rank)`:
    - `r_t` — used for RL updates.
    - `rank_t / max_rank` — stored in `MaxCCList` for LMCC normalisation and plotting.
- Interpretation:
  - The agent is penalised by **LMCC size × relative degree cost** of the removed node.
  - Removing a high-degree node is “expensive” (high `w̄(a)`), so the policy is encouraged to dismantle the network using cheaper (lower-degree) nodes if possible.

**Evaluation / cost metrics (degree cost)**
- `MultiDismantler_degree_cost.MultiDismantler`:
  - `Test()`:
    - After generating `sol` and `remain_nodes`, computes a **remaining-cost penalty**:
      - `remain_score = Σ_{Node ∈ remain_nodes} [ 1 / max_rank × w̄(Node) ]`
    - Returns `self.test_env.score + remain_score`.
  - `GetSol()`:
    - Returns `(score, sol, total_cost_value)` where:
      - `total_cost_value = Σ_{Node ∈ sol} w̄(Node)` (over the trajectory).
  - `EvaluateRealData()`:
    - Writes three outputs per dataset:
      - Node removal sequence (`Solution_...txt`).
      - Normalised LMCC curve (`NormalizedLMCC_...txt`).
      - Cumulative **degree-cost curve** (`Cost_...txt`) defined over `sol + remain_nodes[:-1]`.
    - The last element of `cost` is set to `score` (AUDC-like metric under degree cost).

Key design invariants
---------------------
When modifying or extending the code, preserve the following invariants:

1. **Unit vs degree cost separation**
   - `MultiDismantler_unit_cost/` and `MultiDismantler_degree_cost/` should remain **logically parallel** and separable.
   - Do **not** mix cost logic between them:
     - Unit cost env must not depend on `graph.weights`.
     - Degree cost env must always use the degree-based weights defined in `graph.py`.
   - If introducing a new cost model, mirror the structure by creating a new directory (e.g. `MultiDismantler_custom_cost/`) rather than entangling the existing two.

2. **Reward and evaluation consistency**
   - Whenever you change `MvcEnv.getReward` or LMCC computation:
     - Make sure `step`, `stepWithoutReward`, `Test()`, `GetSol()`, and `EvaluateRealData()` remain **mutually consistent**:
       - `self.test_env.score` should always represent the **cumulative dismantling penalty** under the current cost model.
       - The various “final score” outputs should be interpretable as AUDC-like integrals over LMCC, under the chosen cost definition.
   - Do not silently change the meaning of returned scores without updating:
     - File outputs in `results/unitcost/...` and `results/degreecost/...`.
     - Any documentation that refers to AUDC or robustness metrics.

3. **LMCC definition**
   - LMCC is always computed via `Mcc.MCC(G1, G2, remove_edge)` and `Mcc.find_max_set_length(...)`.
   - `graph.max_rank` must remain the LMCC size **before any removals**.
   - Any change to `Mcc` should be done cautiously and kept identical between unit-cost and degree-cost variants unless there is a deliberate, documented reason.

4. **Graph representation**
   - `Graph` and `Graph_test` are the **single source of truth** for:
     - Adjacency (`adj_list`) and edge lists.
     - Number of nodes / edges.
     - Degree-based weights (degree-cost version).
   - **Extension allowed**: You may extend these classes to store **pre-computed static features** (e.g., Louvain community partition maps, centrality measures) required for new research experiments. These should be computed *once* (offline) and stored, to avoid runtime overhead.

5. **Batch preparation & GNN inputs**
   - `PrepareBatchGraph.py` constructs:
     - Per-layer graph structures (via `GraphStruct`).
     - Auxiliary scalar features per environment (fraction of removed nodes, edge coverage, etc.).
     - Sparse tensors / dicts used by `MRGNN` encoders and aggregators.
   - **Extension allowed**: If changing feature engineering (e.g. appending community features):
     - Ensure the input dimension `input_dim` in the GNN constructor matches the new feature size.
     - Modify `PrepareBatchGraph.py` to handle the concatenation of these new features efficiently.

6. **Reproducibility**
   - Entry scripts (`train.py`, `testReal.py`, `testSynthetic.py`) explicitly set:
     - `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed`.
   - Preserve or extend this seeding pattern when adding new randomness sources (e.g., new sampling procedures).
   - Avoid introducing non-deterministic operations unless strictly necessary for performance.

Coding & style guidelines for this repo
---------------------------------------
- **Python & dependencies**
  - Target Python version: **3.9** (`requires-python = "~=3.9.0"` in `pyproject.toml`).
  - Core dependencies include PyTorch, NetworkX, pandas, matplotlib, torch-scatter/torch-sparse/torch-geometric.
  - Do **not** introduce heavy new dependencies without a clear reason; prefer using the existing stack.
- **File / module boundaries**
  - Prefer extending existing modules rather than creating many small ones; the current layout has a small, well-known set of entrypoints.
  - Keep `MRGNN/` self-contained as the GNN backbone; modifications there should be architecture-specific and not entangled with environment logic.
- **Code style**
  - Keep changes minimal and consistent with existing style:
    - Use descriptive variable names (avoid single-letter names for anything non-trivial).
    - Use type hints where the surrounding code already uses them (e.g. in `mvc_env.py`, `PrepareBatchGraph.py`).
  - Avoid adding surplus inline comments; aim for clear code and high-level documentation instead.
  - Follow the existing pattern of using small helper classes (`SparseMatrix`, `GraphStruct`, `GSet`) rather than ad-hoc dictionaries.
- **GPU / CUDA handling**
  - `cuda_detect.py` is a standalone utility and should remain independent.
  - In training code:
    - Respect the existing device selection logic (`torch.device("cuda:0" if available else "cpu")`).
    - When adding new tensors, ensure they are moved to the correct device.

Workflow notes for future modifications
---------------------------------------
- To **train** a model from scratch:
  - From `code/`:
    - Unit cost: `./run.sh MultiDismantler_unit_cost train`
    - Degree cost: `./run.sh MultiDismantler_degree_cost train`
- To **evaluate pre-trained models**:
  - Use the relevant `testReal.py` or `testSynthetic.py` script under the chosen cost directory, with `--output` pointing to a directory under `results/unitcost/` or `results/degreecost/`.
- When implementing **new experiments or algorithms**:
  - Decide explicitly which cost model you are targeting (unit vs degree).
  - Treat `research_framework_report.md` as a conceptual design document (D-CHAD 2.0); integrate new ideas incrementally so as not to break existing reproducibility.
  - Always check that:
    - `run.sh` still works end-to-end for both unit-cost and degree-cost modes.
    - Output files preserve their expected structure (solution, LMCC curve, cost curve where applicable).

If in doubt, prioritise **reproducibility of the original paper’s results** and **clear separation of cost models** over structural refactoring or aggressive optimisation.

