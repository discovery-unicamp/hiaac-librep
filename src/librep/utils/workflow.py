from dataclasses import dataclass
from typing import List, Union
import time
import warnings
from pathlib import Path

from librep.base.transform import Transform
from librep.base.data import Dataset
from librep.base.estimator import Estimator
from librep.base.evaluators import SupervisedEvaluator, Evaluators
from librep.config.type_definitions import ArrayLike, PathLike
from librep.datasets.multimodal import TransformMultiModalDataset, MultiModalDataset


class SimpleTrainEvalWorkflow:
    def __init__(
        self,
        estimator: Estimator,
        do_fit: bool = True,
        evaluator: SupervisedEvaluator = None,
        save_model_path: PathLike = None,
    ):
        self.estimator = estimator
        self.do_fit = do_fit
        self.evaluator = evaluator
        self.save_model_path = Path(save_model_path) if save_model_path else None

    def __call__(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike = None,
        y_val: ArrayLike = None,
        X_test: ArrayLike = None,
        y_test: ArrayLike = None,
        **fit_params,
    ) -> Union[ArrayLike, dict]:
        if self.do_fit:   
            fit_args = {
                "X": X_train,
                "y": y_train,
            } 
            if X_val is not None and y_val is not None:
                fit_args["X_val"] = X_val
                fit_args["y_val"] = y_val
                
            fit_params.update(fit_args)
            self.estimator.fit(**fit_params)
            
            if self.save_model_path and hasattr(self.estimator, "save"):
                self.estimator.save(self.save_model_path)

        y_pred = self.estimator.predict(X_test)

        if self.evaluator is not None:
            result = self.evaluator.evaluate(y_test, y_pred)
            return result
        else:
            return y_pred  

class SimpleTrainEvalWorkflowMultiModal(SimpleTrainEvalWorkflow):       
    def __call__(
        self,
        train_dataset: ArrayLike,
        validation_dataset: ArrayLike = None,
        test_dataset: ArrayLike = None,
        **fit_params 
    ) -> Union[ArrayLike, List[ArrayLike]]:
        fit_params = {
            "X_train": train_dataset[:][0],
            "y_train": train_dataset[:][1],
            "X_val": validation_dataset[:][0] if validation_dataset is not None else None,
            "y_val": validation_dataset[:][1] if validation_dataset is not None else None,
            "X_test": test_dataset[:][0] if test_dataset is not None else None,
            "y_test": test_dataset[:][1] if test_dataset is not None else None,
        }

        fit_params.update(fit_params)
        return super().__call__(**fit_params)
    

class MultiRunWorkflow:
    def __init__(
        self,
        workflow: SimpleTrainEvalWorkflowMultiModal,
        num_runs: int = 1,
        debug: bool = False,
    ):
        self.workflow = workflow
        self.num_runs = num_runs
        self.debug = debug

    def __call__(
        self,
        train_dataset: ArrayLike,
        validation_dataset: ArrayLike = None,
        test_dataset: ArrayLike = None,
        **fit_params 
    ):
        runs = []
        for i in range(self.num_runs):
            if self.debug:
                print(f"----- Starting run {i+1} / {self.num_runs} ------")
            start = time.time()
            result = self.workflow(
                train_dataset=train_dataset, 
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
                **fit_params
            )
            end = time.time()
            if self.debug:
                print(result)
                print(
                    f"----- Finished run {i+1} / {self.num_runs}. It took: {end-start:.3f} seconds -----\n"
                )
            runs.append(
                {
                    "run id": i + 1,
                    "start": start,
                    "end": end,
                    "time taken": end - start,
                    "result": result,
                }
            )
        return {"runs": runs}
