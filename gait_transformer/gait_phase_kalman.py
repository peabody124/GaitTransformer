# originally implemented in PhaseKalman-20210827.ipynb

import jax
from jax import vmap, jit, jacfwd
from jax import numpy as jnp
import functools

# for convenience defining these dynamics as file wide global variiables

M = 2  # 2 state variables
Ma = 3  # 3 augmented state variables
F = jnp.block([[jnp.array([[0.0, 1.0], [0.0, 0.0]]), jnp.zeros((M, Ma))], [jnp.zeros((Ma, M + Ma))]])

# phase observation noise
N = 8  # number of phase observations
Rp = jnp.eye(N) * 1.0

# process noise
Q = jnp.diag(jnp.array([0.5, 0.5, 0.1, 0.1, 0.1]))

I = jnp.eye(M + Ma)


@jit
def predict(x, P, dt, Q=Q):
    Pdot = F @ P + P @ F.transpose() + Q
    P = P + Pdot * dt

    xdot = F @ x
    x = x + xdot * dt
    return x, P


@jit
def measurements(x):
    omega, _, phi1, phi2, phi3 = x

    return jnp.array(
        [
            jnp.cos(omega),
            jnp.sin(omega),
            jnp.cos(omega + phi1),
            jnp.sin(omega + phi1),
            jnp.cos(omega + phi2),
            jnp.sin(omega + phi2),
            jnp.cos(omega + phi3),
            jnp.sin(omega + phi3),
        ]
    )


@jit
def update_phases(x, P, z, Rp=Rp):
    I = jnp.eye(len(x))

    h = measurements(x)
    H = jacfwd(measurements)(x)
    K = P @ H.transpose() @ jnp.linalg.inv(H @ P @ H.transpose() + Rp)

    x = x + K @ (z - h)
    P = (I - K @ H) @ P

    return x, P


@jit
def rauch_tung_striebel_update(x, P, x_priori_next, P_priori_next, x_posteriori, P_posteriori):
    C = P_posteriori @ F.transpose() @ jnp.linalg.inv(P_priori_next)
    x = x_posteriori + C @ (x - x_priori_next)
    P = P_posteriori + C @ (P - P_priori_next) @ C.transpose()
    return x, P


# @functools.partial(jax.jit, static_argnums=(2,))
# don't jit this as it will retrace every time it is called with
# a different size
def gait_kalman_smoother(phases, dt=1.0 / 30.0, smoothing=True):

    P = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    x = jnp.array([0.0, 0.0, jnp.pi, 3 * jnp.pi / 4, jnp.pi + 3 * jnp.pi / 4])

    a_priori = []
    a_posteriori = []

    @jit
    def constrain(x):
        from jax.nn import relu

        return jnp.array([x[0], relu(x[1]), *x[2:]])

    @jit
    def forward_update(carry, phase, dt=1.0 / 30.0):
        x, P = carry
        x, P = predict(x, P, dt)  # , Q=Q)
        x = constrain(x)

        # TODO: check that some phase is greater than zero
        x2, P2 = update_phases(x, P, phase)

        return (x2, P2), {"a_priori_x": x, "a_priori_P": P, "a_posteriori_x": x2, "a_posteriori_P": P2, "state": x2}

    # warm up a few times, seems to help
    for i in range(5):
        (x2, _), filtered = jax.lax.scan(forward_update, (x, P), phases[:300])
        x = x2.at[0].set(x[0])

    (x, P), filtered = jax.lax.scan(forward_update, (x, P), phases)

    if smoothing:
        backpass = {
            "a_priori_x": filtered["a_priori_x"][1:],
            "a_priori_P": filtered["a_priori_P"][1:],
            "a_posteriori_x": filtered["a_posteriori_x"][:-1],
            "a_posteriori_P": filtered["a_posteriori_P"][:-1],
        }

        last_x = x

        @jit
        def backward_update(carry, y):
            x, P = carry

            # x, P = rauch_tung_striebel_update(x, P, a_priori[i+1]['state'], a_priori[i+1]['covariance'], a_posteriori[i]['state'], a_posteriori[i]['covariance'])
            x, P = rauch_tung_striebel_update(x, P, y["a_priori_x"], y["a_priori_P"], y["a_posteriori_x"], y["a_posteriori_P"])

            return (x, P), {"state": x}

        (x, P), filtered = jax.lax.scan(backward_update, (x, P), backpass, reverse=True)

        state = jax.numpy.concatenate([filtered["state"], last_x[None, :]], axis=0)
    else:
        state = filtered["state"]

    predictions = vmap(measurements)(state)
    errors = jnp.mean((phases - predictions) ** 2, axis=1)

    return state, predictions, errors


def compute_phases(states):
    left_foot_down_phase = jnp.mod(states[:, 0], 2 * jnp.pi)
    right_foot_down_phase = jnp.mod(states[:, 0] + states[:, 2], 2 * jnp.pi)
    left_foot_up_phase = jnp.mod(states[:, 0] + states[:, 3], 2 * jnp.pi)
    right_foot_up_phase = jnp.mod(states[:, 0] + states[:, 4], 2 * jnp.pi)

    return jnp.stack([left_foot_down_phase, right_foot_down_phase, left_foot_up_phase, right_foot_up_phase], axis=1)


def compute_event_times(phase, timestamps, velocity):
    crossings = jnp.where(jnp.diff(phase) < -jnp.pi)[0]
    remainders = jnp.pi * 2 - jnp.take(phase, crossings)
    residual_times = remainders / ((jnp.take(velocity, crossings) + jnp.take(velocity, crossings + 1)) / 2)

    times = jnp.take(timestamps, crossings) + residual_times
    return times


def get_event_times(states, timestamps):
    velocity = states[:, 1]
    phases = compute_phases(states).transpose()
    events = [compute_event_times(p, timestamps, velocity) for p in phases]
    return {k: v for k, v in zip(["left_down", "right_down", "left_up", "right_up"], events)}
