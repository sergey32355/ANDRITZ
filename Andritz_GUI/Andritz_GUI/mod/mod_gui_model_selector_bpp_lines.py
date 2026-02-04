"""TBC"""

import os
import random
import json
import numpy as np
import joblib
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from mod.mod_autoencoder import AutoencoderEstimator
import gui_files.config as config


class ModelSelectorBPPLines:
    """TBC"""

    def __init__(self, fe, random_seed=None, save_model=config.SAVE_MODEL, models_dir=config.MODELS_FOLDER, model_kind=config.MODEL_KIND, device="cpu", verbose=False):
        self.fe = fe
        self.random_seed = random_seed if random_seed is not None else random.randint(0, 1000)
        self.save_model = save_model
        self.models_dir = models_dir
        self.model_kind = model_kind
        self.device = device
        self.verbose = verbose

        #####################################
        ### HERE YOU CAN CHANGE THE MODEL ###
        #####################################
        
        if self.model_kind == "autoencoder":
            clf = AutoencoderEstimator(
                input_dim=fe.X_good_train.shape[1],
                random_seed=random_seed,
                device=device,
            )
            params = config.AE_PARAMS
            self.trained_model, self.best_model_params = self.train_pipeline(
                clf = clf,
                params = params,
            )

        self.mean_pred_snip_proba = self.get_snip_proba_and_update_fe()
        
        self.best_threshold_params, self.best_threshold_scores = (
            self.tune_threshold_params()
        )

        self.test_proba, self.test_pred, self.test_scores = self.get_seg_proba_and_pred_scores(
            df_bad_locations=self.fe.df_bad_test_locations,
            df_good_locations=self.fe.df_good_test_locations,
            **self.best_threshold_params,
        )

        if self.verbose:
            print("Best thresholding parameters:")
            print(json.dumps(self.best_threshold_params, indent=2, ensure_ascii=False))
            print("Best scores of trained model on thresholding dataset:")
            print(json.dumps(self.best_threshold_scores, indent=2, ensure_ascii=False))
        print("Scores of trained model on test dataset:")
        print(json.dumps(self.test_scores, indent=2, ensure_ascii=False))

        if save_model:
            dt = datetime.now(tz=ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(self.models_dir, exist_ok=True)
            model_name = f"{self.model_kind}_{self.fe.dl.data_dir}_{dt}.pkl"
            self.model_path = os.path.join(self.models_dir, model_name)
            print(f"\nSaving model to {self.model_path}")
            # Save both trained model and threshold parameters
            model_data = {
            'trained_model': self.trained_model,
            'best_threshold_params': self.best_threshold_params
            }
            joblib.dump(model_data, self.model_path)
            # Check that saving and loading works correctly
            loaded_data = joblib.load(self.model_path)
            loaded_model = loaded_data['trained_model']
            loaded_threshold_params = loaded_data['best_threshold_params']
            assert type(loaded_model) == type(self.trained_model)
            assert loaded_threshold_params == self.best_threshold_params
            # assert self.trained_model.get_params() == loaded_model.get_params() # Does not work
            original_state = self.trained_model["clf"].model_.state_dict()
            loaded_state = loaded_model["clf"].model_.state_dict()
            assert original_state.keys() == loaded_state.keys()
            assert all(torch.equal(original_state[key], loaded_state[key]) for key in original_state.keys())    

    def train_pipeline(self, clf, params, scale=True):

        print("Training model... (this might take a while)\n")

        if sum([len(param_list) for param_list in params.values()]) > len(params):

            # Define pipeline
            pipe = Pipeline(
                [
                    ("std", StandardScaler()),
                    ("clf", clf),
                ]
                if scale
                else [("clf", clf)]
            )

            # Define integrated hyperparameter tuning with GridSearchCV
            gscv = GridSearchCV(
                pipe,
                params,
                cv=4,
                n_jobs=-1,
                refit=True,
                return_train_score=True,
                verbose=2,
            )

            gscv.fit(self.fe.X_good_train)
            trained_model = gscv.best_estimator_
            best_model_params = gscv.best_params_

        else:

            best_model_params = {
                key.split("__")[-1]: value[0] for key, value in params.items()
            }
            clf.__init__(
                input_dim=self.fe.X_good_train.shape[1],
                random_seed=self.random_seed,
                device=self.device,
                **best_model_params,
            )

            # Define pipeline
            pipe = Pipeline(
                [
                    ("std", StandardScaler()),
                    ("clf", clf),
                ]
                if scale
                else [("clf", clf)]
            )

            pipe.fit(self.fe.X_good_train)
            trained_model = pipe

        return trained_model, best_model_params

    def get_snip_proba_and_update_fe(self):
        """TBC"""

        y_bad_pred_threshold = self.trained_model.predict_proba(self.fe.X_bad_threshold)
        y_good_pred_threshold = self.trained_model.predict_proba(
            self.fe.X_good_threshold
        )
        self.fe.df_bad_threshold_locations["pred_proba"] = y_bad_pred_threshold
        self.fe.df_good_threshold_locations["pred_proba"] = y_good_pred_threshold

        y_bad_pred_test = self.trained_model.predict_proba(self.fe.X_bad_test)
        y_good_pred_test = self.trained_model.predict_proba(self.fe.X_good_test)
        self.fe.df_bad_test_locations["pred_proba"] = y_bad_pred_test
        self.fe.df_good_test_locations["pred_proba"] = y_good_pred_test

        y_good_pred_train = self.trained_model.predict_proba(self.fe.X_good_train)

        mean_pred_snip_proba = {
            "bad_threshold": round(np.mean(y_bad_pred_threshold), 4),
            "bad_test": round(np.mean(y_bad_pred_test), 4),
            "good_threshold": round(np.mean(y_good_pred_threshold), 4),
            "good_test": round(np.mean(y_good_pred_test), 4),
            "good_train": round(np.mean(y_good_pred_train), 4),
        }

        return mean_pred_snip_proba

    def get_seg_proba_and_pred_scores(
        self, df_bad_locations, df_good_locations, m=4, seg_threshold=0.1
    ):
        """TBC"""
        seg_proba = {}
        seg_pred = {}
        scores = {}
        # Group by segment and compute mean of top m prediction probabilities
        bad_seg_proba = df_bad_locations.groupby(
            ["plate", "segment_type", "segment_number"]
        )["pred_proba"].apply(lambda x: x.nlargest(m).mean())
        good_seg_proba = df_good_locations.groupby(
            ["plate", "segment_type", "segment_number"]
        )["pred_proba"].apply(lambda x: x.nlargest(m).mean())

        # Apply seg_threshold to get binary predictions at segment level
        bad_seg_pred = (bad_seg_proba >= seg_threshold).astype(int)
        good_seg_pred = (good_seg_proba >= seg_threshold).astype(int)

        # Compute recall for bad and good segments
        recall_bad = bad_seg_pred.sum() / len(bad_seg_pred)
        recall_good = 1 - good_seg_pred.sum() / len(good_seg_pred)

        # Average of recalls (balanced accuracy)
        balanced_accuracy = (recall_bad + recall_good) / 2

        seg_proba = {
            "bad_seg_proba": bad_seg_proba,
            "good_seg_proba": good_seg_proba,
        }
        seg_pred = {
            "bad_seg_pred": bad_seg_pred,
            "good_seg_pred": good_seg_pred,
        }
        scores = {
            "balanced_accuracy": round(balanced_accuracy,4),
            "recall_bad": round(recall_bad,4),
            "recall_good": round(recall_good,4),
        }
        return seg_proba, seg_pred, scores

    def tune_threshold_params(self):

        best_threshold_params = {}
        best_threshold_scores = {}
        # Iterate over all combinations of thresholding parameters
        for m, seg_threshold in product(
            config.THRESHOLD_PARAMS["m"], config.THRESHOLD_PARAMS["seg_threshold"]
        ):
            _, _, new_scores = self.get_seg_proba_and_pred_scores(
                df_bad_locations=self.fe.df_bad_threshold_locations,
                df_good_locations=self.fe.df_good_threshold_locations,
                m=m,
                seg_threshold=seg_threshold,
            )
            # Update best_threshold_scores and best_threshold_params
            if (
                len(best_threshold_scores) == 0
                or new_scores["balanced_accuracy"]
                > best_threshold_scores["balanced_accuracy"]
            ):
                best_threshold_scores = new_scores
                best_threshold_params = {
                    "m": m,
                    "seg_threshold": seg_threshold,
                }

        return best_threshold_params, best_threshold_scores
    

class PredictorScorerBPPLines:
    """TBC"""

    def __init__(self, fe, model, threshold_parameters, verbose=True):
        self.fe = fe
        self.trained_model = model
        self.threshold_parameters = threshold_parameters
        self.verbose = verbose
        self.evaluate = (self.fe.X_bad_test is not None)
        self.get_snip_proba_and_update_fe()

        if self.evaluate:
            print("Predicting classes for loaded dataset and comparing with provided labels.\n")
            self.bad_seg_proba, self.bad_seg_pred = self.get_seg_proba_and_pred(self.fe.df_bad_test_locations)
            self.good_seg_proba, self.good_seg_pred = self.get_seg_proba_and_pred(self.fe.df_good_test_locations)
            self.test_scores = self.get_pred_scores()
            print("Scores of model on labelled dataset:")
            print(json.dumps(self.test_scores, indent=2, ensure_ascii=False))
        else:
            print("Predicting classes for loaded data. No labels were provided for evaluation.\n")
            self.all_seg_proba, self.all_seg_pred = self.get_seg_proba_and_pred(self.fe.df_all_locations)
            print("List of segments predicted to be defective:")
            self.pred_defect_seg = self.all_seg_pred[self.all_seg_pred==1].index.tolist()
            defects_list = [
                f"Plate: {plate}, Segment: {seg_type}_{seg_number}"
                for plate, seg_type, seg_number in self.pred_defect_seg
            ]
            print(json.dumps(defects_list, indent=2, ensure_ascii=False))

    def get_snip_proba_and_update_fe(self):
        """TBC"""

        if self.evaluate:
            y_bad_pred_test = self.trained_model.predict_proba(self.fe.X_bad_test)
            y_good_pred_test = self.trained_model.predict_proba(self.fe.X_good_test)
            self.fe.df_bad_test_locations["pred_proba"] = y_bad_pred_test
            self.fe.df_good_test_locations["pred_proba"] = y_good_pred_test
        else:
            y_all_pred = self.trained_model.predict_proba(self.fe.X_all)
            self.fe.df_all_locations["pred_proba"] = y_all_pred

    def get_seg_proba_and_pred(self, df):
        """TBC"""
        m = self.threshold_parameters["m"]
        seg_threshold = self.threshold_parameters["seg_threshold"]

        # Group by segment and compute mean of top m prediction probabilities
        seg_proba = df.groupby(
            ["plate", "segment_type", "segment_number"]
        )["pred_proba"].apply(lambda x: x.nlargest(m).mean())

        # Apply seg_threshold to get binary predictions at segment level
        seg_pred = (seg_proba >= seg_threshold).astype(int)

        return seg_proba, seg_pred
    
    def get_pred_scores(self):
        """TBC"""

        # Compute recall for bad and good segments
        recall_bad = self.bad_seg_pred.sum() / len(self.bad_seg_pred)
        recall_good = 1 - self.good_seg_pred.sum() / len(self.good_seg_pred)

        # Average of recalls (balanced accuracy)
        balanced_accuracy = (recall_bad + recall_good) / 2

        scores = {
            "balanced_accuracy": round(balanced_accuracy,4),
            "recall_bad": round(recall_bad,4),
            "recall_good": round(recall_good,4),
        }

        return scores
