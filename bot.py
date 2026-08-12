# bot.py

from functools import partial
from typing import NamedTuple
import pickle

import jax
import jax.numpy as jnp
import flax.linen as nn
import mctx


# ============================================================
# Model board: 8 rows × 10 columns
#
# game.Instance:
#   field[x][y]
#   x = 0..9
#   y = 0..7
#
# Model:
#   board[y][x]
# ============================================================

H = 8
W = 10
N = H * W
MOVES_COUNT = 3


# ============================================================
# State
# ============================================================


class VirusState(NamedTuple):
    alive0: jax.Array
    dead0: jax.Array
    alive1: jax.Array
    dead1: jax.Array
    player: jax.Array
    phase: jax.Array


# ============================================================
# Game logic
# ============================================================


def neighbours(mask):
    up = jnp.pad(
        mask[:, 1:, :],
        ((0, 0), (0, 1), (0, 0)),
    )

    down = jnp.pad(
        mask[:, :-1, :],
        ((0, 0), (1, 0), (0, 0)),
    )

    left = jnp.pad(
        mask[:, :, 1:],
        ((0, 0), (0, 0), (0, 1)),
    )

    right = jnp.pad(
        mask[:, :, :-1],
        ((0, 0), (0, 0), (1, 0)),
    )

    return up | down | left | right


def controlled_territory(alive, dead):
    owned = alive | dead

    def cond(carry):
        reached, frontier = carry

        new_frontier = neighbours(frontier) & owned & ~reached

        return jnp.any(new_frontier)

    def body(carry):
        reached, frontier = carry

        new_frontier = neighbours(frontier) & owned & ~reached

        return (
            reached | new_frontier,
            new_frontier,
        )

    reached, _ = jax.lax.while_loop(
        cond,
        body,
        (alive, alive),
    )

    return reached


def current_masks(state):
    p0 = state.player[:, None, None] == 0

    my_alive = jnp.where(
        p0,
        state.alive0,
        state.alive1,
    )

    my_dead = jnp.where(
        p0,
        state.dead0,
        state.dead1,
    )

    enemy_alive = jnp.where(
        p0,
        state.alive1,
        state.alive0,
    )

    return (
        my_alive,
        my_dead,
        enemy_alive,
    )


def legal_actions_mask(state):
    my_alive, my_dead, enemy_alive = current_masks(state)

    controlled = controlled_territory(
        my_alive,
        my_dead,
    )

    occupied = state.alive0 | state.dead0 | state.alive1 | state.dead1

    empty = ~occupied

    legal = neighbours(controlled) & (empty | enemy_alive)

    return legal.reshape(
        state.player.shape[0],
        N,
    )


def terminal_mask(state):
    return ~jnp.any(
        legal_actions_mask(state),
        axis=1,
    )


def apply_actions(state, actions):
    batch = state.player.shape[0]

    r = actions // W
    c = actions % W

    selected = (
        jnp.zeros(
            (batch, H, W),
            dtype=jnp.bool_,
        )
        .at[
            jnp.arange(batch),
            r,
            c,
        ]
        .set(True)
    )

    p0 = state.player[:, None, None] == 0

    capture0 = p0 & selected & state.alive1

    capture1 = ~p0 & selected & state.alive0

    # Actions passed here are legal, therefore
    # non-captures are necessarily growth.
    grow0 = p0 & selected & ~state.alive1

    grow1 = ~p0 & selected & ~state.alive0

    alive0 = (state.alive0 & ~capture1) | grow0

    alive1 = (state.alive1 & ~capture0) | grow1

    dead0 = state.dead0 | capture0

    dead1 = state.dead1 | capture1

    end_turn = state.phase == 2

    next_player = jnp.where(
        end_turn,
        1 - state.player,
        state.player,
    ).astype(jnp.int32)

    next_phase = jnp.where(
        end_turn,
        0,
        state.phase + 1,
    ).astype(jnp.int32)

    return VirusState(
        alive0,
        dead0,
        alive1,
        dead1,
        next_player,
        next_phase,
    )


def transition_info(state, actions):
    next_state = apply_actions(
        state,
        actions,
    )

    done = terminal_mask(next_state)

    player_changed = next_state.player != state.player

    # If no next atomic move exists:
    #
    # same player -> that player loses
    # switched player -> opponent loses
    terminal_reward = jnp.where(
        player_changed,
        1.0,
        -1.0,
    )

    reward = jnp.where(
        done,
        terminal_reward,
        0.0,
    ).astype(jnp.float32)

    discount = jnp.where(
        done,
        0.0,
        jnp.where(
            player_changed,
            -1.0,
            1.0,
        ),
    ).astype(jnp.float32)

    return (
        next_state,
        reward,
        discount,
        done,
    )


# ============================================================
# Instance -> model state
# ============================================================


def instance_to_state(instance):
    alive0 = jnp.zeros(
        (H, W),
        dtype=jnp.bool_,
    )

    dead0 = jnp.zeros(
        (H, W),
        dtype=jnp.bool_,
    )

    alive1 = jnp.zeros(
        (H, W),
        dtype=jnp.bool_,
    )

    dead1 = jnp.zeros(
        (H, W),
        dtype=jnp.bool_,
    )

    # game:
    #   x = 0..9
    #   y = 0..7
    #
    # model:
    #   row = y
    #   col = x

    for x in range(10):
        for y in range(8):
            cell = instance.field[x][y]

            if cell.isFree:
                continue

            if cell.player == 0:
                if cell.isAlive:
                    alive0 = alive0.at[y, x].set(True)
                else:
                    dead0 = dead0.at[y, x].set(True)

            else:
                if cell.isAlive:
                    alive1 = alive1.at[y, x].set(True)
                else:
                    dead1 = dead1.at[y, x].set(True)

    # movesLeft:
    #
    # 3 -> phase 0
    # 2 -> phase 1
    # 1 -> phase 2
    phase = MOVES_COUNT - instance.movesLeft

    return VirusState(
        alive0=alive0[None],
        dead0=dead0[None],
        alive1=alive1[None],
        dead1=dead1[None],
        player=jnp.asarray(
            [instance.currentPlayer],
            dtype=jnp.int32,
        ),
        phase=jnp.asarray(
            [phase],
            dtype=jnp.int32,
        ),
    )


# ============================================================
# Neural network
# ============================================================


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x, train=False):
        residual = x

        x = nn.Conv(
            self.channels,
            (3, 3),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)

        x = nn.BatchNorm(
            use_running_average=not train,
            name="bn1",
        )(x)

        x = nn.relu(x)

        x = nn.Conv(
            self.channels,
            (3, 3),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(x)

        x = nn.BatchNorm(
            use_running_average=not train,
            name="bn2",
        )(x)

        return nn.relu(residual + x)


class VirusNet(nn.Module):
    channels: int = 64
    blocks: int = 6

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            self.channels,
            (3, 3),
            padding="SAME",
            use_bias=False,
            name="trunk_conv",
        )(x)

        x = nn.BatchNorm(
            use_running_average=not train,
            name="trunk_bn",
        )(x)

        x = nn.relu(x)

        for i in range(self.blocks):
            x = ResidualBlock(
                self.channels,
                name=f"resblock_{i}",
            )(x, train=train)

        # Policy
        p = nn.Conv(
            2,
            (1, 1),
            use_bias=False,
            name="policy_conv",
        )(x)

        p = nn.BatchNorm(
            use_running_average=not train,
            name="policy_bn",
        )(p)

        p = nn.relu(p)

        p = p.reshape((p.shape[0], -1))

        logits = nn.Dense(
            N,
            name="policy_dense",
        )(p)

        # Value
        v = nn.Conv(
            1,
            (1, 1),
            use_bias=False,
            name="value_conv",
        )(x)

        v = nn.BatchNorm(
            use_running_average=not train,
            name="value_bn",
        )(v)

        v = nn.relu(v)

        v = v.reshape((v.shape[0], -1))

        v = nn.Dense(
            256,
            name="value_dense1",
        )(v)

        v = nn.relu(v)

        v = nn.Dense(
            1,
            name="value_dense2",
        )(v)

        value = jnp.tanh(v[:, 0])

        return logits, value


net = VirusNet()


# ============================================================
# 11-channel encoding used during training
# ============================================================


def encode_states(state):
    p0 = state.player[:, None, None] == 0

    control0 = controlled_territory(
        state.alive0,
        state.dead0,
    )

    control1 = controlled_territory(
        state.alive1,
        state.dead1,
    )

    occupied = state.alive0 | state.dead0 | state.alive1 | state.dead1

    empty = ~occupied

    frontier0 = neighbours(control0) & (empty | state.alive1)

    frontier1 = neighbours(control1) & (empty | state.alive0)

    board = jnp.stack(
        [
            jnp.where(
                p0,
                state.alive0,
                state.alive1,
            ),
            jnp.where(
                p0,
                state.dead0,
                state.dead1,
            ),
            jnp.where(
                p0,
                state.alive1,
                state.alive0,
            ),
            jnp.where(
                p0,
                state.dead1,
                state.dead0,
            ),
            jnp.where(
                p0,
                control0,
                control1,
            ),
            jnp.where(
                p0,
                control1,
                control0,
            ),
            jnp.where(
                p0,
                frontier0,
                frontier1,
            ),
            jnp.where(
                p0,
                frontier1,
                frontier0,
            ),
        ],
        axis=-1,
    ).astype(jnp.float32)

    # P1 sees canonical board rotated 180°.
    board = jnp.where(
        (state.player == 1)[:, None, None, None],
        jnp.flip(
            board,
            axis=(1, 2),
        ),
        board,
    )

    phase = jax.nn.one_hot(
        state.phase,
        3,
        dtype=jnp.float32,
    )

    phase = jnp.broadcast_to(
        phase[:, None, None, :],
        (
            state.player.shape[0],
            H,
            W,
            3,
        ),
    )

    return jnp.concatenate(
        [board, phase],
        axis=-1,
    )


# ============================================================
# Network prediction
# ============================================================


def predict(variables, state):
    logits, value = net.apply(
        variables,
        encode_states(state),
        train=False,
    )

    batch = state.player.shape[0]

    board = logits.reshape(
        batch,
        H,
        W,
    )

    # Convert canonical coordinates
    # back to real/model-board coordinates.
    board = jnp.where(
        (state.player == 1)[:, None, None],
        jnp.flip(
            board,
            axis=(1, 2),
        ),
        board,
    )

    logits = board.reshape(
        batch,
        N,
    )

    legal = legal_actions_mask(state)

    logits = jnp.where(
        legal,
        logits,
        jnp.finfo(logits.dtype).min,
    )

    return logits, value


# ============================================================
# mctx
# ============================================================


def select_state(mask, a, b):
    def choose(x, y):
        shape = mask.shape + (1,) * (x.ndim - 1)

        return jnp.where(
            mask.reshape(shape),
            x,
            y,
        )

    return jax.tree_util.tree_map(
        choose,
        a,
        b,
    )


def recurrent_fn(
    variables,
    rng_key,
    action,
    state,
):
    del rng_key

    already_terminal = terminal_mask(state)

    (
        candidate,
        reward,
        discount,
        done,
    ) = transition_info(
        state,
        action,
    )

    next_state = select_state(
        already_terminal,
        state,
        candidate,
    )

    reward = jnp.where(
        already_terminal,
        0.0,
        reward,
    )

    discount = jnp.where(
        already_terminal,
        0.0,
        discount,
    )

    done = already_terminal | done

    logits, value = predict(
        variables,
        next_state,
    )

    value = jnp.where(
        done,
        0.0,
        value,
    )

    return (
        mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ),
        next_state,
    )


qtransform = partial(
    mctx.qtransform_completed_by_mix_value,
    rescale_values=False,
    value_scale=0.1,
    maxvisit_init=50.0,
    use_mixed_value=True,
)


# ============================================================
# Bot
# ============================================================


class VirusBot:
    def __init__(
        self,
        model_path,
        simulations=128,
    ):
        with open(
            model_path,
            "rb",
        ) as f:
            checkpoint = pickle.load(f)

        self.variables = {
            "params": jax.tree_util.tree_map(
                jax.device_put,
                checkpoint["params"],
            ),
            "batch_stats": jax.tree_util.tree_map(
                jax.device_put,
                checkpoint["batch_stats"],
            ),
        }

        self.simulations = simulations
        self.rng = jax.random.PRNGKey(1)

        def search(
            variables,
            rng_key,
            state,
        ):
            logits, value = predict(
                variables,
                state,
            )

            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=state,
            )

            legal = legal_actions_mask(state)

            return mctx.gumbel_muzero_policy(
                params=variables,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=simulations,
                invalid_actions=~legal,
                max_num_considered_actions=16,
                # Deterministic play.
                gumbel_scale=0.0,
                qtransform=qtransform,
            )

        self._search = jax.jit(search)

    def get_move(self, instance):
        """
        Returns (x, y) suitable for:

            instance.move(x, y)
        """

        state = instance_to_state(instance)

        if bool(terminal_mask(state)[0]):
            raise RuntimeError("No legal moves.")

        self.rng, key = jax.random.split(self.rng)

        result = self._search(
            self.variables,
            key,
            state,
        )

        result.action.block_until_ready()

        action = int(result.action[0])

        # model:
        #   row = y
        #   col = x
        #
        # game:
        #   field[x][y]
        x = action % W
        y = action // W

        return x, y
