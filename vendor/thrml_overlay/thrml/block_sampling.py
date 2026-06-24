import dataclasses
from collections import defaultdict
from typing import Mapping, Sequence, Type, TypeAlias

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree, Shaped

from thrml.block_management import (
    Block,
    BlockSpec,
    block_state_to_global,
    verify_block_state,
)
from thrml.interaction import InteractionGroup
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES, AbstractNode

from .conditional_samplers import AbstractConditionalSampler, _SamplerState
from .observers import AbstractObserver, ObserveCarry, StateObserver

# A SuperBlock is a collection of blocks that will be sampled at the same "time"
# specifically, they will be sampled separately, but without updating the state
# in between (so same algorithmic, but not computation time)
# This could be used if you have need to break blocks up for e.g. different
# samplers/nodes
SuperBlock: TypeAlias = tuple[Block, ...] | Block
_SD: TypeAlias = Mapping[Type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]

# HTDML patch-live MARKER (mirror exp15 is_patch_live). The reversible forward/reverse symmetrized
# block-Gibbs scan ( K = 1/2(P_AB + P_BA) ) + the v2 shared/per-chain order-coin toggle live in
# `sample_blocks` below. A detector asserts this constant is present + the v2 toggle is in the source.
REVERSIBLE_SCAN_MARKER = "HTDML-REVERSIBLE-SCAN-v2:fwd-rev-symmetrized-block-gibbs;K=half(P_AB+P_BA);order-coin-toggle"


class BlockGibbsSpec(BlockSpec):
    """
    A BlockGibbsSpec is a type of BlockSpec which contains additional information
    on free and clamped blocks.

    This entity also supports `SuperBlock`s, which are merely groups of blocks
    which are sampled at the same time algorithmically, but not programmatically.
    That is to say, superblock = (block1, block2) means that the states input to
    block1 and block2 are the same, but they are not executed at the same time.
    This may be because they are the same color on a graph, but require vastly
    different sampling methods such that JAX SIMD approaches are not feasible
    to parallelize them.

    A recurring theme in `thrml` is the importance of implicit indexing. One
    such example can be seen here. Because global states are created by
    concatenating lists of free and clamped blocks, providing the inputs
    in the same order as the blocks are defined is essential. This is almost
    always taken care of internally, but when writing custom functions or
    interfaces this is important to keep in mind.

    **Attributes:**

    - `free_blocks`: the list of free blocks (in order)
    - `sampling_order`: a list of `len(superblocks)` lists, where each
        `sampling_order[i]` is the index of `free_blocks` to sample.
        Sampling is done by iterating over this order and sampling each
        sublist of free blocks at the same algorithmic time.
    - `clamped_blocks`: the list of clamped blocks
    - `superblocks`: the list of superblocks
    """

    free_blocks: list[Block]
    sampling_order: list[list[int]]
    clamped_blocks: list[Block]
    superblocks: list[tuple[Block, ...]]

    def __init__(
        self,
        free_super_blocks: Sequence[SuperBlock],
        clamped_blocks: list[Block],
        node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES,
    ):
        """Create a Gibbs specification from free and clamped blocks.

        **Arguments:**

        - `free_super_blocks`: An ordered sequence where each element is either
            a single `Block`, or a tuple of blocks that must share the same global
            state when calling their individual samplers.
        - `clamped_blocks`: Blocks whose nodes stay fixed during sampling.
        - `node_shape_dtypes`: Mapping from node class to a PyTree of
            `jax.ShapeDtypeStruct`; identical to the argument in `BlockSpec`.
        """
        free_blocks = []
        sampling_order = []
        superblocks = []
        i = 0
        for super_block in free_super_blocks:
            if isinstance(super_block, Block):
                blocks = (super_block,)
            else:
                blocks = super_block

            superblocks.append(blocks)
            sampling_group = []
            for block in blocks:
                free_blocks.append(block)
                sampling_group.append(i)
                i += 1
            sampling_order.append(sampling_group)

        super().__init__(free_blocks + clamped_blocks, node_shape_dtypes)
        self.free_blocks = free_blocks
        self.clamped_blocks = clamped_blocks
        self.sampling_order = sampling_order
        self.superblocks = superblocks


def _tree_slice(x, sl):
    if eqx.is_array(x):
        return jnp.take(x, sl, axis=0)
    return x


class BlockSamplingProgram(eqx.Module):
    """A PGM block-sampling program.

    This class encapsulates everything that is needed to run a PGM block sampling program in THRML.
    `per_block_interactions` and `per_block_interaction_active` are parallel to the free blocks in `gibbs_spec`, and
    their members are passed directly to a sampler when the state of the corresponding free block is being updated
    during a sampling program. `per_block_interaction_global_inds` and `per_block_interaction_global_slices` are
    also parallel to the free blocks, and are used to slice the global state of the program to produce the
    state information required to update the state of each block alongside the static information contained in the
    interactions.

    **Attributes:**

    - `gibbs_spec`: A division of some PGM into free and clamped blocks.
    - `samplers`: A sampler to use to update every free block in `gibbs_spec`.
    - `per_block_interactions`: All the interactions that touch each free block in `gibbs_spec`.
    - `per_block_interaction_active`: indicates which interactions are real
        and which interactions are not part of the model and have been added to pad data structures so that they
        can be rectangular.
    - `per_block_interaction_global_inds`: how to find the information required to update each block within the global
        state list
    - `per_block_interaction_global_slices`: how to slice each array in the global state list to find the information
        required to update each block
    """

    gibbs_spec: BlockGibbsSpec
    samplers: list[AbstractConditionalSampler]
    per_block_interactions: list[list[PyTree]]
    per_block_interaction_active: list[list[Array]]
    per_block_interaction_global_inds: list[list[list[int]]]
    per_block_interaction_global_slices: list[list[list[Array]]]

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: list[AbstractConditionalSampler],
        interaction_groups: list[InteractionGroup],
    ):
        """Construct a `BlockSamplingProgram`.

        This code is the beating heart of THRML, and the chance that you should be
        modifying it or trying to understand it deeply are very low (as this would
        basically correspond to re-writing the library). This code takes in a set of
        information that implicitly defines a sampling program and manipulates it into
        a shape that is appropriate for practical vectorized block-sampling program.
        This involves reindexing, slicing, and often padding.

        **Arguments:**

        - `gibbs_spec`: A division of some PGM into free and clamped blocks.
        - `samplers`: The update rule to use for each free block in `gibbs_spec`.
        - `interaction_groups`: A list of `InteractionGroups` that define how the
            variables in your sampling program affect one another.
        """

        self.gibbs_spec = gibbs_spec
        self.samplers = samplers

        # first, construct a map from every head node to each interaction it
        # shows up in and where it shows up in that interaction

        head_node_map = defaultdict(list)

        for i, interaction_group in enumerate(interaction_groups):
            for j, node in enumerate(interaction_group.head_nodes.nodes):
                head_node_map[node].append((i, j))

        # now, let's organize this information on the interactions into a block format

        interaction_inds = []
        max_n_interactions = []

        for block in gibbs_spec.free_blocks:
            this_block_interaction_info = [
                [[] for _ in range(len(block.nodes))] for _ in range(len(interaction_groups))
            ]
            for j, node in enumerate(block.nodes):
                this_node_interaction_info = head_node_map[node]
                for info in this_node_interaction_info:
                    this_block_interaction_info[info[0]][j].append(info[1])
            interaction_inds.append(this_block_interaction_info)
            this_max_n = [max([len(x) for x in this_int]) for this_int in this_block_interaction_info]
            max_n_interactions.append(this_max_n)

        # now, take the block-arranged interaction structure and use it to construct the block-arranged interactions
        # and slicers for the global state

        # if you are reading this, god help you

        per_block_interactions = []
        per_block_interaction_active = []
        per_block_interaction_global_inds = []
        per_block_interaction_global_slices = []

        for block, block_interact_inds, block_n_interactions in zip(
            gibbs_spec.free_blocks, interaction_inds, max_n_interactions
        ):
            this_block_interactions = []
            this_block_active = []
            this_block_global_inds = []
            this_block_global_slices = []
            for interaction_group, interact_inds, n_interactions in zip(
                interaction_groups, block_interact_inds, block_n_interactions
            ):
                if n_interactions > 0:
                    n_nodes = len(block.nodes)
                    interaction_slices = np.zeros((n_nodes, n_interactions), dtype=int)

                    global_inds = []
                    global_slices = []
                    for tail_block in interaction_group.tail_nodes:
                        global_inds.append(gibbs_spec.node_global_location_map[tail_block.nodes[0]][0])
                        global_slices.append(np.zeros((n_nodes, n_interactions), dtype=int))

                    active = np.zeros((n_nodes, n_interactions), dtype=bool)
                    for i, inds in enumerate(interact_inds):
                        for j, ind in enumerate(inds):
                            interaction_slices[i, j] = ind
                            active[i, j] = 1

                            for k, tail_block in enumerate(interaction_group.tail_nodes):
                                s = gibbs_spec.node_global_location_map[tail_block.nodes[ind]][1]
                                global_slices[k][i, j] = s

                    interaction_slices = jnp.array(interaction_slices)

                    sliced_interaction = jax.tree.map(
                        lambda x: _tree_slice(x, interaction_slices),  # shape -> (n, m, …)
                        interaction_group.interaction,
                    )

                    this_block_interactions.append(sliced_interaction)
                    this_block_active.append(jnp.array(active))
                    this_block_global_inds.append(global_inds)
                    this_block_global_slices.append([jnp.array(x) for x in global_slices])
            per_block_interactions.append(this_block_interactions)
            per_block_interaction_active.append(this_block_active)
            per_block_interaction_global_inds.append(this_block_global_inds)
            per_block_interaction_global_slices.append(this_block_global_slices)

        self.per_block_interactions = per_block_interactions
        self.per_block_interaction_active = per_block_interaction_active
        self.per_block_interaction_global_inds = per_block_interaction_global_inds
        self.per_block_interaction_global_slices = per_block_interaction_global_slices


_State: TypeAlias = PyTree[Shaped[Array, "nodes ?*state"], "_State"]


def sample_single_block(
    key: Key[Array, ""],
    state_free: list[_State],
    clamp_state: list[_State],
    program: BlockSamplingProgram,
    block: int,
    sampler_state: _SamplerState,
    global_state: list[PyTree] | None = None,
) -> tuple[_State, _SamplerState]:
    """Samples a single block within a Gibbs sampling program based on the current
    states and program configurations. It extracts neighboring states, processes
    required data, and applies a sampling function to generate output samples.

    **Arguments:**

    - `key`: Pseudo-random number generator key to ensure reproducibility of sampling.
    - `state_free`: Current states of free blocks, representing the values to be
        updated during sampling.
    - `clamp_state`: Clamped states that remain fixed during the sampling process.
    - `program`: The Gibbs sampling program containing specifications, samplers,
        neighborhood information, and parameters.
    - `block`: Index of the block to be sampled in the current iteration.
    - `sampler_state`: The current state of the sampler that will be used to
        perform the update.
    - `global_state`: Optionally precomputed global state for the concatenated
        free and clamped blocks; when omitted the function constructs it internally.

    **Returns:**

    - Updated block state and sampler state for the specified block.
    """
    # flatten the state and extract neighbor states
    if global_state is None:
        global_state = block_state_to_global(state_free + clamp_state, program.gibbs_spec)
    per_interaction_global_inds = program.per_block_interaction_global_inds[block]
    per_interaction_slices = program.per_block_interaction_global_slices[block]

    all_interaction_states = []
    for interaction_global_inds, interaction_slices in zip(per_interaction_global_inds, per_interaction_slices):
        this_interaction_states = []
        for ind, sl in zip(interaction_global_inds, interaction_slices):
            this_interaction_states.append(
                jax.tree.map(
                    lambda x: jnp.take(x, sl, axis=0),  # shape -> (n, m, …)
                    global_state[ind],
                )
            )
        all_interaction_states.append(this_interaction_states)

    this_block = program.gibbs_spec.free_blocks[block]

    node_type = this_block.node_type
    template_sd = program.gibbs_spec.node_shape_struct[node_type]

    def _resize_sd(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct((len(this_block.nodes), *leaf.shape), leaf.dtype)
        return leaf

    sd_to_pass = jax.tree.map(_resize_sd, template_sd)

    sampler = program.samplers[block]
    out_samples, out_sampler_state = sampler.sample(
        key,
        program.per_block_interactions[block],
        program.per_block_interaction_active[block],
        all_interaction_states,
        sampler_state,
        sd_to_pass,
    )
    return out_samples, out_sampler_state


def sample_blocks(
    key: Key[Array, ""],
    state_free: list[_State],
    clamp_state: list[_State],
    program: BlockSamplingProgram,
    sampler_state: list[_SamplerState],
    order_subkey: Key[Array, ""] | None = None,
) -> tuple[list[_State], list[_SamplerState]]:
    """Perform one iteration of sampling, visiting every block.

    **Arguments:**

    - `key`: The JAX PRNG key.
    - `state_free`: The state of the free blocks.
    - `clamp_state`: The state of the clamped blocks.
    - `program`: The Gibbs program.
    - `sampler_state`: The state of the sampler.

    **Returns:**

    - Updated free-block state list and sampler-state list.
    """

    # gaurdrail state/block compatability here as everything else calls this

    sds = {node_type: jax.tree.unflatten(*sd) for node_type, sd in program.gibbs_spec.node_shape_dtypes.items()}
    verify_block_state(program.gibbs_spec.free_blocks, state_free, sds, -1)
    verify_block_state(program.gibbs_spec.clamped_blocks, clamp_state, sds, -1)

    # HTDML-REVERSIBLE-SCAN PATCH v2 (was: a single deterministic for-loop over sampling_order, i.e.
    # alternating/deterministic-scan = non-reversible = A2-excluded/F3-regime).
    # Forward/reverse symmetrized block-Gibbs scan; A2-satisfied. A per-sweep fair coin selects the
    # FULL superblock visitation order forward vs exactly reversed:
    #   coin=0 -> P_fwd = P_{g_{M-1}} ... P_{g_0};  coin=1 -> P_rev = P_{g_0} ... P_{g_{M-1}}.
    # Each single-block Gibbs update P_i resamples from pi_theta(x_i | x_{-i}) => P_i is pi-reversible
    # (self-adjoint in L2(pi)). Hence (P_fwd)* = P_rev, and M = 1/2(P_fwd + P_rev) = 1/2(P_fwd + P_fwd*)
    # is self-adjoint => pi-reversible. Only the application ORDER is randomized (per-block keys[i] fixed
    # per sweep), so each chain's MARGINAL kernel is exactly 1/2(P_fwd+P_rev). Single-superblock => no-op.
    # The DTM negative phase has 4 superblocks (upper_hidden, lower_hidden, image_output, label_output).
    # v2 ORDER-COIN TOGGLE (identical per-chain marginal kernel either way; differs ONLY in cross-chain
    # coin correlation, which the single-chain self-adjointness gate cannot and need not see):
    #   * order_subkey is None -> PER-CHAIN coin (diagnostics): coin from THIS chain's key, so under
    #     vmap-over-chains the predicate is batched => lax.cond computes BOTH sweeps (~2x). Keeps the
    #     frozen-theta diagnostic across-chain SEM (P5) exactly independent.
    #   * order_subkey given   -> SHARED coin (training): coin from a key shared across chains (non-
    #     batched under vmap) => lax.cond stays true control flow => ONE sweep (~1x, the speedup).
    #     Per-block Gibbs NOISE stays per-chain (from `key`); only the visitation order is shared.
    # Justification + numerical self-adjointness gate:
    #   <thermo-wiki>/experiments/internal-exp/patches/reversible-scan.md
    if order_subkey is None:
        order_key, block_key = jax.random.split(key)
        coin = jax.random.bernoulli(order_key)
    else:
        block_key = key
        coin = jax.random.bernoulli(order_subkey)
    keys = jax.random.split(block_key, (len(program.gibbs_spec.free_blocks),))
    fwd_order = list(program.gibbs_spec.sampling_order)
    rev_order = list(reversed(fwd_order))

    def _sweep(order, state_free, sampler_state):
        for sampling_group in order:
            global_state = block_state_to_global(state_free + clamp_state, program.gibbs_spec)
            state_updates = {}
            for i in sampling_group:
                state_updates[i], sampler_state[i] = sample_single_block(
                    keys[i], state_free, clamp_state, program, i, sampler_state[i], global_state
                )

            for i, state in state_updates.items():
                state_free[i] = state
        return state_free, sampler_state

    state_free, sampler_state = jax.lax.cond(
        coin,
        lambda sf, ss: _sweep(rev_order, sf, ss),
        lambda sf, ss: _sweep(fwd_order, sf, ss),
        state_free, sampler_state,
    )
    return state_free, sampler_state


def _run_blocks(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    n_iters: int,
    sampler_states: list[_SamplerState],
    order_key: Key[Array, ""] | None = None,
) -> tuple[list[PyTree[Shaped[Array, "n_iters nodes ?*state"]]], list[_SamplerState]]:
    """
    Perform `n_iters` steps of block sampling.

    EXP4: `order_key=None` -> each sweep uses the PER-CHAIN order coin (sample_blocks draws it). If
    given, it is split into one shared order-subkey per sweep => SHARED (across-chain) coin (training).
    """
    if n_iters == 0:
        return init_chain_state, sampler_states

    keys = jax.random.split(key, n_iters)

    if order_key is None:
        def body_fn(states, _key):
            state_free, sampler_state = states
            return sample_blocks(_key, state_free, state_clamp, program, sampler_state), None

        return jax.lax.scan(body_fn, (init_chain_state, sampler_states), keys)[0]

    order_subkeys = jax.random.split(order_key, n_iters)

    def body_fn(states, scan_in):
        _key, _ok = scan_in
        state_free, sampler_state = states
        return sample_blocks(_key, state_free, state_clamp, program, sampler_state, order_subkey=_ok), None

    return jax.lax.scan(body_fn, (init_chain_state, sampler_states), (keys, order_subkeys))[0]


@dataclasses.dataclass
class SamplingSchedule:
    """
    Represents a sampling schedule for a process.

    **Attributes:**

    - `n_warmup`: The number of warmup steps to run before collecting samples.
    - `n_samples`: The number of samples to collect.
    - `steps_per_sample`: The number of steps to run between each sample.
    """

    n_warmup: int
    n_samples: int
    steps_per_sample: int

    def __hash__(self) -> int:
        return hash((self.n_warmup, self.n_samples, self.steps_per_sample))


def sample_with_observation(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    observation_carry_init: ObserveCarry,
    f_observe: AbstractObserver,
    order_key: Key[Array, ""] | None = None,
) -> tuple[ObserveCarry, list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]]:
    """Run the full chain and call an Observer after every recorded sample.

    **Arguments:**

    - `key`: RNG key.
    - `program`: The sampling program.
    - `schedule`: Warm-up length, number of samples, number of steps between samples.
    - `init_chain_state`: Initial free-block state.
    - `state_clamp`: Clamped-block state.
    - `observation_carry_init`: Initial carry handed to `f_observe`.
    - `f_observe`: Observer instance.

    **Returns:**

    - Tuple `(final_observer_carry, samples)` where `samples` is a PyTree whose
        leading axis has size `schedule.n_samples`.
    """
    # run warmup
    sampler_states = jax.tree.map(
        lambda x: x.init(),
        program.samplers,
        is_leaf=lambda a: isinstance(a, AbstractConditionalSampler),
    )
    if order_key is not None:
        order_key, ok_warmup = jax.random.split(order_key, 2)
    else:
        ok_warmup = None
    key, subkey = jax.random.split(key, 2)
    warmup_state, warmup_sampler_states = _run_blocks(
        subkey,
        program,
        init_chain_state,
        state_clamp,
        schedule.n_warmup,
        sampler_states,
        order_key=ok_warmup,
    )
    mem, warmup_observation = f_observe(program, warmup_state, state_clamp, observation_carry_init, jnp.array(0))

    if schedule.n_samples <= 1:
        warmup_observation = jax.tree.map(lambda x: x[None], warmup_observation)
        return mem, warmup_observation

    # collect samples

    if order_key is not None:
        ok_samples = jax.random.split(order_key, schedule.n_samples - 1)
    else:
        ok_samples = None

    def body_fn(carry, input):
        (prev_state, prev_sampler_state), _mem = carry

        if order_key is not None:
            _key, i, _ok = input
        else:
            _key, i = input
            _ok = None

        new_state, new_sampler_state = _run_blocks(
            _key,
            program,
            prev_state,
            state_clamp,
            schedule.steps_per_sample,
            prev_sampler_state,
            order_key=_ok,
        )
        _mem, observe_out = f_observe(program, new_state, state_clamp, _mem, i)
        new_carry = ((new_state, new_sampler_state), _mem)
        return new_carry, observe_out

    keys = jax.random.split(key, schedule.n_samples - 1)
    outer_iters = jnp.arange(1, schedule.n_samples)

    if order_key is not None:
        inputs = (keys, outer_iters, ok_samples)
    else:
        inputs = (keys, outer_iters)

    (_, mem_out), observed_results = jax.lax.scan(body_fn, ((warmup_state, warmup_sampler_states), mem), inputs)

    # need to prepend the first observation from the warmup
    def prepend_warmup_observation(_warmup, _rest):
        return jnp.concatenate([_warmup[None], _rest], axis=0)

    observed_results = jax.tree.map(prepend_warmup_observation, warmup_observation, observed_results)

    return mem_out, observed_results


def sample_states(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state_free: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    nodes_to_sample: list[Block],
) -> list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]:
    """Convenience wrapper to collect state information for *nodes_to_sample* only.

    Internally builds a [`thrml.StateObserver`][], runs
    [`thrml.sample_with_observation`][], and returns a stacked tensor of shape
    `(schedule.n_samples, ...)`.
    """
    f_observe = StateObserver(nodes_to_sample)
    carry_init = f_observe.init()

    mem_out, results_out = sample_with_observation(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        carry_init,
        f_observe,
    )

    return results_out
