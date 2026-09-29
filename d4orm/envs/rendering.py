"""Matplotlib exports following the mbd-multi trajectory and animation style."""

import matplotlib
import numpy as np
from matplotlib import animation, patches
from matplotlib import pyplot as plt

COLORMAPS = ("Reds", "Greens", "Purples", "Oranges", "Blues")


def _color(agent_index, group_ids=None):
    index = (
        agent_index if group_ids is None else int(group_ids[agent_index]) + 1
    )
    return matplotlib.colormaps[COLORMAPS[index % len(COLORMAPS)]](0.6)


def _make_scene(environment, static=False, spatial=False):
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d" if spatial else None)
    # Center a box with side length 1.5 times the environment diameter.
    half_extent = 0.65 * environment.diameter
    limits = (-half_extent, half_extent)
    axes.set(xlim=limits, ylim=limits)
    if spatial:
        axes.set(zlim=limits, xlabel="X", ylabel="Y", zlabel="Z")
        axes.set_box_aspect((1, 1, 1))
    else:
        axes.set_aspect("equal")
        if static and environment.pos_dim_agent == 2:
            axes.set(xticks=[], yticks=[])
        for index, (center, radius) in enumerate(
            zip(
                np.asarray(environment.obstacle_centers),
                np.asarray(environment.obstacle_radii),
            )
        ):
            axes.add_patch(
                patches.Circle(
                    center[:2],
                    radius,
                    color="black",
                    alpha=0.5,
                    label="Obstacle" if index == 0 else None,
                )
            )
        if environment.num_obstacles:
            axes.legend(fontsize=15)
    return figure, axes


def _body(environment, axes, position, color, zorder=5):
    spatial = environment.pos_dim_agent == 3
    outer = patches.Circle(
        position[:2],
        environment.agent_radius,
        color=color,
        alpha=0.7 if spatial else 1.0,
        zorder=zorder,
        linewidth=0,
    )
    axes.add_patch(outer)
    bodies = [outer]
    if spatial:
        inner = patches.Circle(
            position[:2],
            environment.agent_radius_original,
            color=color,
            zorder=zorder + 0.1,
            linewidth=0,
        )
        axes.add_patch(inner)
        bodies.append(inner)
    return bodies


def _draw_paths(environment, axes, states, group_ids=None):
    spatial = environment.pos_dim_agent == 3
    for agent_index in range(environment.num_agents):
        color = _color(agent_index, group_ids)
        positions = states[:, agent_index, :2]
        axes.plot(
            *positions.T,
            color=color,
            linewidth=1,
            alpha=0.5 if spatial else 0.8,
        )
        _body(environment, axes, positions[0], color)
    # Use simulation collision predicates, including overlap/boundary contact.
    collisions = np.asarray(environment.collision_matrix(states)).any(axis=-1)
    collisions |= np.asarray(environment.obstacle_collision_matrix(states)).any(
        axis=-1
    )
    positions = states[..., :2][collisions]
    if positions.size:
        axes.plot(*positions.T, "rx", markersize=10, markeredgewidth=1)
    goals = np.asarray(environment.xg).reshape(environment.num_agents, -1)
    axes.plot(
        *goals[:, :2].T,
        "+",
        color="black",
        alpha=0.5,
        markersize=10,
        markeredgewidth=1,
        zorder=10,
    )


def render_trajectory(
    environment, trajectory, gif_path, image_path, group_ids=None
):
    """Saves reference-style GIF and static plot, projecting 3D paths onto XY.

    Static plots show starts, goals, and collisions. Animations show moving
    bodies and headings. A None GIF path skips animation.
    """
    states = np.asarray(trajectory).reshape(
        -1, environment.num_agents, environment.obsv_dim_agent
    )
    if not len(states):
        raise ValueError("Trajectory must contain at least one state.")
    offset = max(1, int(environment.offset))
    if gif_path is not None and str(gif_path) != "None":
        figure, axes = _make_scene(environment)
        try:
            # Keep the shared scene bounds fixed throughout the animation.
            figure.set_size_inches(6.0, 6.0)
            figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
            axes.set(
                xticks=[],
                yticks=[],
            )
            bodies, headings = [], []
            for index in range(environment.num_agents):
                color = _color(index, group_ids)
                bodies.append(_body(environment, axes, states[0, index], color))
                headings.append(
                    axes.plot([], [], color="black", linewidth=1.5, zorder=10)[
                        0
                    ]
                )

            def update(frame):
                for index in range(environment.num_agents):
                    state = states[frame, index]
                    for layer, body in enumerate(bodies[index]):
                        body.set_center(state[:2])
                        if environment.pos_dim_agent == 3:
                            body.set_zorder(float(state[2]) + layer * 0.1)
                    headings[index].set_data(
                        *environment.get_heading_line(state, state[:2], index)
                    )
                return [body for layers in bodies for body in layers] + headings

            frames = list(range(0, len(states), offset))
            if frames[-1] != len(states) - 1:
                frames.append(len(states) - 1)
            movie = animation.FuncAnimation(
                figure, update, frames=frames, interval=100
            )
            movie.save(
                gif_path,
                writer=animation.PillowWriter(fps=max(1, 10 // offset)),
            )
        finally:
            plt.close(figure)
    figure, axes = _make_scene(environment, static=True)
    try:
        _draw_paths(environment, axes, states[::offset], group_ids)
        figure.savefig(image_path, bbox_inches="tight", pad_inches=0.05)
    finally:
        plt.close(figure)


def show_trajectory(environment, trajectory):
    """Shows a trajectory, retaining a rotatable 3D view for spatial robots."""
    states = np.asarray(trajectory).reshape(
        -1, environment.num_agents, environment.obsv_dim_agent
    )
    spatial = environment.pos_dim_agent == 3
    figure, axes = _make_scene(environment, static=True, spatial=spatial)
    try:
        if spatial:
            goals = np.asarray(environment.xg).reshape(
                environment.num_agents, -1
            )
            for index in range(environment.num_agents):
                axes.plot(
                    *states[:, index, :3].T, color=_color(index), linewidth=1
                )
                axes.scatter(*states[0, index, :3], color=_color(index))
            axes.scatter(*goals[:, :3].T, color="black", marker="+")
        else:
            _draw_paths(environment, axes, states)
        plt.show()
    finally:
        plt.close(figure)
