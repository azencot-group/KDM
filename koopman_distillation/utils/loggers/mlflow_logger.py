import os
import warnings
from pathlib import Path

from .base_logger import BaseLogger
from typing import Dict, Any, List, Optional
from pprint import pprint
import numpy as np
from PIL import Image


def is_basic(x):
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def convert_no_basic_to_str(sub_dict: Dict[str, Any]):
    return {k: v if is_basic(v)
    else str(v) if not isinstance(v, dict) else convert_no_basic_to_str(v)
            for k, v in sub_dict.items()}


class MlflowLogger(BaseLogger):
    # http://132.72.40.201:5000
    def __init__(self, ip, port, project=None, *args, **kwargs):
        super(MlflowLogger, self).__init__(*args, **kwargs)
        if self.rank != 0:
            return
        import mlflow
        from mlflow import log_metric, log_param, log_params, log_artifacts, log_figure, log_image
        from matplotlib import pyplot as plt
        mlflow.set_tracking_uri(f"http://{ip}:{port}")
        if project is None:
            local_path_api_project = Path('neptune') / 'project.txt'
            if local_path_api_project.exists():
                project = local_path_api_project.read_text().strip()
            else:
                warnings.warn('''Please create a file at neptune/project.txt with your Neptune project name''')
                raise FileNotFoundError('Neptune project not found')
        mlflow.set_experiment(experiment_name=project)
        self.log_metric = log_metric
        self.log_param = log_param
        self._log_params = log_params
        self.log_artifacts = log_artifacts
        self.log_figure = log_figure
        self.log_image = log_image
        self.mlflow = mlflow
        self.run = mlflow.start_run()
        self.plt = plt
        self.step_tracker = {}

    def stop(self):
        if self.rank == 0:
            self.mlflow.end_run()

    def log(self, name: str, data: Any, step=None):
        if self.rank == 0:
            self.log_metric(name, data, step)

    def _log_fig(self, name: str, fig: Any):
        if self.rank == 0:
            if name not in self.step_tracker:
                self.step_tracker[name] = 1
            else:
                self.step_tracker[name] += 1
            if isinstance(fig, self.plt.Figure):
                canvas = fig.canvas
                canvas.draw()
                image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
                self.log_image(image=image, key=name, step=self.step_tracker[name])
            elif isinstance(fig, np.ndarray):
                if fig.dtype != np.uint8:
                    fig = fig * 255
                    fig = fig.astype(np.uint8)
                fig = Image.fromarray(fig)
                self.log_image(image=fig, key=name, step=self.step_tracker[name])
            elif isinstance(fig, Image.Image):
                self.log_image(image=fig, key=name, step=self.step_tracker[name])

    def log_params(self, params: Dict[str, Any]):
        if self.rank == 0:
            for k, v in params.items():
                if isinstance(v, dict):
                    self._log_params({f'{k}/{kk}': vv for kk, vv in v.items()})
                else:
                    self.log_param(k, v)
            self.mlflow.log_dict(convert_no_basic_to_str(params), 'params.json')

    def add_tags(self, tags: List[str]):
        if self.rank == 0:
            self.mlflow.set_tags({'tags': tags})

    def log_hparams(self, params: Dict[str, Any]):
        if self.rank == 0:
            params = convert_no_basic_to_str(params)
            self.mlflow.log_params(params)

    def log_name_params(self, name: str, params: Any):
        if self.rank == 0:
            self.mlflow.log_metric(name, params)