import sys
import os
import time
import subprocess
from collections import deque
from statistics import mean

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph import mkBrush, ViewBox, PlotCurveItem, DateAxisItem
import zmq
import numpy as np

import logging
import logging_setup


# ---------- ZMQ subscriber thread ----------
class ZmqListener(QtCore.QThread):
    """
    SUBscribes to both engine and training publishers and emits each message as a dict.
    """
    message = QtCore.pyqtSignal(dict)

    def __init__(self, addresses, parent=None):
        super().__init__(parent)
        self._addresses = addresses
        self._running = True

        self.log = logging.getLogger('MyRLApp.GUI')

    def run(self):
        self.log.debug("ZmqListener: Starting ZMQ Listener thread")
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        # connect to each publisher
        for addr in self._addresses:
            sub.connect(addr)
        # sub.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics

        self.log.debug("ZmqListener: Subscribed to addresses.")

        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)

        self.log.debug("ZmqListener: Going into listening loop.")
        while self._running:
            socks = dict(poller.poll(timeout=500))  # 0.5 s timeout so we can shut down cleanly
            if sub in socks and socks[sub] == zmq.POLLIN:
                msg = sub.recv_json()
                if not isinstance(msg, dict):
                    self.log.debug(f"ZmqListener: Received invalid message from engine -> {msg}")
                    continue
                self.message.emit(msg)

        self.log.debug("ZmqListener: Exited listening loop.")
        sub.close()
        ctx.term()

    def stop(self):
        self._running = False
        self.wait()


# ---------- Main application window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RLlib + Engine Monitor")
        self.log = logging.getLogger('MyRLApp.GUI')
        self.log.info(f"GUI, PID={os.getpid()}")

        #========= DEFINE WHICH ALGO TO RUN HERE ==========================================
        algo = 'IMPALA'

        # launch Master.py (which is set up to spawn custom_run.py, shared_memory_env_runner.py, minion.py)
        script_dir = os.path.dirname('/Users/rodrigohadlich/PycharmProjects/RayProject/')
        master_path = os.path.join(script_dir, "Master.py")
        cmd = [
            sys.executable,
            master_path,
            "--algo", algo,
            "--enable-zmq", "True",
        ]
        # use the same Python interpreter
        self.master_proc = subprocess.Popen(cmd)

        # create containers for plot parameters
        self.plot_colors = ["#e60049", "#0bb4ff", "#50e991", "#ffa300", "#9b19f5", "#dc0ab4", "#b3d4ff", "#00bfa0"]
        self.plot_line_width = 5

        # data structures for plotting
        self._max_points = 3000
        # engine: dynamic curves keyed by metric name
        self.engine_curves = {}
        self.engine_data = {}
        self.engine_x = deque(maxlen=self._max_points)
        self.engine_count = 0
        self.evaluation_count = 0
        self.evaluation_x = deque(maxlen=self._max_points)
        self.policy_x = {
            'delta in minion': deque(maxlen=self._max_points),
            'ratio_max': deque(maxlen=self._max_points),
        }

        # manually set fields to be plotted
        self.engine_data["imep"] = deque(maxlen=self._max_points)
        self.engine_data["mprr"] = deque(maxlen=self._max_points)
        self.engine_data["target imep"] = deque(maxlen=self._max_points)
        self.engine_data["mean sampled imep"] = deque(maxlen=self._max_points)
        self.engine_data["evaluation error"] = deque(maxlen=self._max_points)
        self.policy_data = {
            'delta in minion': deque(maxlen=self._max_points),
            'ratio_max': deque(maxlen=self._max_points),
        }
        # self.engine_data["ratio_p99"] = deque(maxlen=self._max_points)

        # training: one curve for reward vs iteration
        self.training_curve = None
        self.training_x = []
        self.training_y = []

        # set up the UI
        central = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        alpha = 225

        # engine metrics (load tracking) plot
        self.load_plot = pg.PlotWidget(title="Load Tracking (minion.py)")
        legend_load = self.load_plot.addLegend()
        legend_load.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        # legend_load.setFrame(True)
        self.load_plot.showGrid(x=True, y=True)
        self.load_plot.setBackground('w')
        vlay.addWidget(self.load_plot)
        # make curves for load tracking plot
        pen = pg.mkPen(color=self.plot_colors[0], width=self.plot_line_width)
        curve = self.load_plot.plot(name='imep', pen=pen)
        self.engine_curves['imep'] = curve
        pen = pg.mkPen(color=self.plot_colors[1], width=self.plot_line_width)
        curve = self.load_plot.plot(name='target imep', pen=pen)
        self.engine_curves['target imep'] = curve
        pen = pg.mkPen(color=self.plot_colors[2], width=self.plot_line_width)
        curve = self.load_plot.plot(name='mean sampled imep', pen=pen)
        self.engine_curves['mean sampled imep'] = curve

        # engine metrics (safety) plot
        self.safety_plot = pg.PlotWidget(title="Safety (minion.py)")
        legend_safety = self.safety_plot.addLegend()
        legend_safety.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.safety_plot.showGrid(x=True, y=True)
        self.safety_plot.setBackground('w')
        vlay.addWidget(self.safety_plot)
        # make curve for safety plot
        pen = pg.mkPen(color=self.plot_colors[3], width=self.plot_line_width)
        curve = self.safety_plot.plot(name='mprr', pen=pen)
        self.engine_curves['mprr'] = curve

        # engine metrics (safety) plot
        self.evaluation_plot = pg.PlotWidget(title="Evaluation (minion.py)")
        legend_evaluation = self.evaluation_plot.addLegend()
        legend_evaluation.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.evaluation_plot.showGrid(x=True, y=True)
        self.evaluation_plot.setBackground('w')
        vlay.addWidget(self.evaluation_plot)
        # make curve for safety plot
        pen = pg.mkPen(color=self.plot_colors[4], width=self.plot_line_width)
        curve = self.evaluation_plot.plot(name='evaluation error', pen=pen)
        self.engine_curves['evaluation error'] = curve

        # training reward plot
        self.training_plot = pg.PlotWidget(title="Training Reward (custom_run.py)")
        legend_training = self.training_plot.addLegend()
        legend_training.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.training_plot.showGrid(x=True, y=True)
        self.training_plot.setBackground('w')
        vlay.addWidget(self.training_plot)

        # policy update plots
        date_axis = DateAxisItem(orientation='bottom')
        self.t0 = time.time()
        self.policy_plot = pg.PlotWidget(
            axisItems={'bottom': date_axis},
            title="Policy Update"
        )
        legend_policy = self.policy_plot.addLegend()
        legend_policy.setBrush(mkBrush(255, 255, 255, alpha))  # RGBA, 200 alpha
        self.policy_plot.showGrid(x=True, y=True)
        self.policy_plot.setBackground('w')
        vlay.addWidget(self.policy_plot)
        self.policy_plot.showAxis('right')
        self.policy_plot.getAxis('right').setLabel('ratio_max', **{'color': self.plot_colors[6]})
        self.ratio_vb = ViewBox()
        self.policy_plot.scene().addItem(self.ratio_vb)
        self.policy_plot.getAxis('right').linkToView(self.ratio_vb)
        self.ratio_vb.setXLink(self.policy_plot.getViewBox())
        self.policy_plot.getViewBox().sigResized.connect(
            lambda: self.ratio_vb.setGeometry(self.policy_plot.getViewBox().sceneBoundingRect())
        )
        # add curves
        pen = pg.mkPen(color=self.plot_colors[5], width=self.plot_line_width)
        curve = self.policy_plot.plot(name='delta in minion', pen=pen)
        self.engine_curves['delta in minion'] = curve
        pen = pg.mkPen(color=self.plot_colors[6], width=self.plot_line_width)
        curve = PlotCurveItem(name='ratio_max', pen=pen)
        self.ratio_vb.addItem(curve)
        self.engine_curves['ratio_max'] = curve
        # pen = pg.mkPen(color=self.plot_colors[6], width=self.plot_line_width)
        # curve = self.policy_plot.plot(name='ratio_p99', pen=pen)
        # self.engine_curves['ratio_p99'] = curve

        # ── Insert a horizontal layout at the top for controls ──
        controls = QtWidgets.QHBoxLayout()
        self.stop_btn = QtWidgets.QPushButton("Stop Processes")
        self.stop_btn.clicked.connect(self.stop_processes)
        controls.addWidget(self.stop_btn)
        # you could add more buttons here later (e.g. “Pause”, “Restart”)

        # Insert controls above the plots
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.addLayout(controls)
        main_layout.addWidget(self.load_plot)
        main_layout.addWidget(self.safety_plot)
        main_layout.addWidget(self.evaluation_plot)
        main_layout.addWidget(self.training_plot)
        main_layout.addWidget(self.policy_plot)
        self.setCentralWidget(central)

        # 3) Start ZMQ listener thread
        #    (adjust these addresses if you used tcp:// or a different ipc path)
        addresses = [
            "ipc:///tmp/engine.ipc",
            "ipc:///tmp/training.ipc",
        ]
        self.listener = ZmqListener(addresses)
        self.listener.message.connect(self.on_zmq_message)
        self.listener.start()

        # 4a) Setup a QTimer to refresh the high-speed plots at 10 Hz
        self._refresh_timer_HS = QtCore.QTimer(self)
        self._refresh_timer_HS.setInterval(100)  # 100 ms => 10 Hz
        self._refresh_timer_HS.timeout.connect(self._refresh_plots_hs)
        self._refresh_timer_HS.start()

        # 4a) Setup a QTimer to refresh the low-speed plots at 1 Hz
        self._refresh_timer_LS = QtCore.QTimer(self)
        self._refresh_timer_LS.setInterval(1000)  # 1000 ms => 1 Hz
        self._refresh_timer_LS.timeout.connect(self._refresh_plots_ls)
        self._refresh_timer_LS.start()

        self.log.debug("GUI: Done with init.")

    def update_views(self):
        """When the main plot is resized, update the secondary VB to match."""
        self.ratio_vb.setGeometry(self.policy_plot.getViewBox().sceneBoundingRect())
        self.ratio_vb.linkedViewChanged(self.policy_plot.getViewBox(), self.ratio_vb.XAxis)

    def stop_processes(self):
        if self.listener.isRunning():
            self.listener.stop()

        if self.master_proc.poll() is None:
            self.master_proc.terminate()
            try:
                self.master_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.master_proc.kill()

    def closeEvent(self, event):
        # Clean up ZMQ thread
        self.listener.stop()
        # Terminate Master.py (and thus its children)
        if self.master_proc.poll() is None:
            self.master_proc.terminate()
        super().closeEvent(event)

    @QtCore.pyqtSlot()
    def _refresh_plots_hs(self):
        x = list(self.engine_x)
        self.engine_curves['imep'].setData(x, list(self.engine_data['imep']))
        self.engine_curves['target imep'].setData(x, list(self.engine_data['target imep']))
        self.engine_curves['mprr'].setData(x, list(self.engine_data['mprr']))
        self.engine_curves['mean sampled imep'].setData(x, list(self.engine_data['mean sampled imep']))

    @QtCore.pyqtSlot()
    def _refresh_plots_ls(self):
        self.engine_curves['evaluation error'].setData(
            list(self.evaluation_x),
            list(self.engine_data['evaluation error'])
        )
        self.engine_curves['delta in minion'].setData(
            list(self.policy_x['delta in minion']),
            list(self.policy_data['delta in minion'])
        )
        self.engine_curves['ratio_max'].setData(
            list(self.policy_x['ratio_max']),
            list(self.policy_data['ratio_max'])
        )
        if self.training_curve:
            self.training_curve.setData(self.training_x, self.training_y)

    @QtCore.pyqtSlot(dict)
    def on_zmq_message(self, msg):
        topic = msg.get("topic", "")
        if topic == "engine":
            self._update_engine(msg)
        elif topic == "training":
            self._update_training(msg)
        elif topic == "evaluation":
            self._update_evaluation(msg)
        elif topic == "policy":
            self._update_policy(msg)

    def _update_policy(self, msg):
        # self.log.debug(f"GUI: In _update_policy.")
        t = time.time() - self.t0

        for key in msg.keys():
            if key != "topic":
                # self.log.debug(f"GUI: In _update_policy key: {key}, value = {msg[key]}")
                self.policy_data[key].append(msg[key])
                self.policy_x[key].append(t)
        # self.log.debug(f"GUI: Done with _update_evaluation.")

    def _update_evaluation(self, msg):
        # self.log.debug(f"GUI: In _update_evaluation.")
        self.evaluation_count += 1
        self.evaluation_x.append(self.evaluation_count)

        data_list = self.engine_data['evaluation error']
        data_list.append(np.abs(msg["current imep"] - msg["target"]))
        # self.log.debug(f"GUI: Done with _update_evaluation.")

    def _update_engine(self, msg):
        # self.log.debug(f"GUI: In _update_engine.")
        self.engine_count += 1
        self.engine_x.append(self.engine_count)
        # self.log.debug(f"GUI (_update_engine): msg -> {msg}.")

        data = {
            "imep": msg["current imep"],
            "mprr": msg["mprr"],
            "target imep": msg["target"],
            "mean sampled imep": mean(self.engine_data['target imep']) if list(
                self.engine_data['target imep']) != [] else 0,
        }

        # if list(self.engine_data['target imep']) is not []:
        #     data.update({"mean sampled imep": mean(self.engine_data['target imep'])})

        # self.log.debug(f"GUI (_update_engine): data -> {data}.")
        for i, (k, v) in enumerate(data.items()):
            # append data
            data_list = self.engine_data[k]
            data_list.append(v)

        # self.log.debug(f"GUI: Done with _update_engine.")

    def _update_training(self, msg):
        # Determine iteration & reward keys
        # Adjust these if you used different JSON keys

        # self.log.debug(f"GUI (_update_engine): msg -> {msg}.")

        if "iteration" in msg:
            x = msg["iteration"]
        else:
            return

        if "mean_return" in msg:
            y = msg["mean_return"]
        else:
            return

        if "eval_return" in msg:
            y2 = msg["eval_return"]

        self.training_x.append(x)
        self.training_y.append(y)
        if len(self.training_x) > self._max_points:
            self.training_x.pop(0)
            self.training_y.pop(0)

        if self.training_curve is None:
            # first time: create it
            pen = pg.mkPen(color=self.plot_colors[-1], width=self.plot_line_width)
            self.training_curve = self.training_plot.plot(name="Reward", pen=pen)
            self.training_plot.addLegend()

        # self.training_curve.setData(self.training_x, self.training_y)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
