import numpy as np
import matplotlib.pyplot as plt

class DALPolicyOPCVisualizer:
    def __init__(self, buffer_ref, var_names, interval=100):
        self.buffer_ref = buffer_ref
        self.var_names = var_names
        self.interval = interval
        self.state_objective_mapping = DAL4TankEnv.state_objectives_mapping
        self.n_vars = len(var_names) - len(self.state_objective_mapping)

        self.fig, self.axes = plt.subplots(
            self.n_vars, 1, sharex=True, figsize=(14, 1.2 * self.n_vars), gridspec_kw={"hspace": 0.05}
        )
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, wspace=0.05)
        self.fig.suptitle("DAL RL Policy Dashboard")
        if self.n_vars == 1:
            self.axes = [self.axes]
        self.lines = []
        self.objective_lines = []
        state_limits = np.array(DAL4TankEnv.state_limits)
        for i, (ax, name) in enumerate(zip(self.axes, var_names)):
            # State variable line (blue)
            (line,) = ax.plot([], [], label=name, color="C0")
            ax.set_ylabel(name)
            ymin, ymax = state_limits[i]
            ax.set_ylim(ymin, ymax)
            # Objective variable line (tomato), if applicable
            obj_line = None
            if name in self.state_objective_mapping:
                obj_var = self.state_objective_mapping[name]
                if obj_var in var_names:
                    obj_idx = var_names.index(obj_var)
                    (obj_line,) = ax.plot([], [], label=f"{obj_var} (objective)", color="tomato", linestyle="--")
            self.lines.append(line)
            self.objective_lines.append(obj_line)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.5)
        self.axes[-1].set_xlabel("Step")

    def init_plot(self):
        for ax in self.axes:
            ax.set_xlim(0, max(10, len(self.buffer_ref)))
            ax.set_ylabel("")
        for line in self.lines:
            line.set_data([], [])
        for obj_line in self.objective_lines:
            if obj_line is not None:
                obj_line.set_data([], [])
        return self.lines + [l for l in self.objective_lines if l is not None]

    def update(self, frame):
        data = np.array(self.buffer_ref)
        if data.size == 0:
            return self.lines + [l for l in self.objective_lines if l is not None]
        state_limits = np.array(DAL4TankEnv.state_limits)
        x = np.arange(data.shape[0])
        for i, line in enumerate(self.lines):
            line.set_data(x, data[:, i])
            self.axes[i].set_xlim(0, max(10, len(self.buffer_ref)))
            # Draw objective if present
            obj_line = self.objective_lines[i]
            if obj_line is not None:
                # Find index of objective variable in buffer
                obj_var = self.state_objective_mapping[self.var_names[i]]
                obj_idx = self.var_names.index(obj_var)
                obj_line.set_data(x, data[:, obj_idx])
                self.axes[i].set_ylim(
                    max(state_limits[obj_idx, 0], data[-1, obj_idx] - 5.0),
                    min(state_limits[obj_idx, 1], data[-1, obj_idx] + 5.0),
                )
        return self.lines + [l for l in self.objective_lines if l is not None]

    def start(self):
        ani = animation.FuncAnimation(
            self.fig, self.update, init_func=self.init_plot, interval=self.interval, blit=False
        )
        plt.show()