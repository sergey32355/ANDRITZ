"""TBC"""
import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

class LoggerBPPLines:
     
    def __init__(self, notes="", do_log_results=True, do_log_summary=True):
        """TBC"""
        self.notes = notes
        self.do_log_results = do_log_results
        self.do_log_summary = do_log_summary
        self.summary_results = pd.DataFrame(columns=['balanced_accuracy', 'recall_bad', 'recall_good'])

    def log_results(self, ms):
        """ TBC """

        anom_bad_segs = f"{ms.test_preds["bad_seg_pred"].sum()}/{len(ms.test_preds["bad_seg_pred"])}"
        anom_good_segs = f"{ms.test_preds["good_seg_pred"].sum()}/{len(ms.test_preds["good_seg_pred"])}"
            
        if self.do_log_results:

            logs_file_path = os.path.join("sensor_tests","anomaly_logs","anomaly_logs_ae.xlsx")
            sheet_name = "anomaly_detection_ae_bpp"
            
            new_row = {
                "datetime": datetime.now(tz=ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S"),
                "notes": self.notes,
                "sensor": ms.fe.dl.sensor_id if ms.fe.dl.ddc is None else f"{ms.fe.dl.sensor_id}_{ms.fe.dl.ddc}",
                "random_seed": ms.random_seed,
                "model": ms.model,
                "model_parameters": str(ms.best_model_params),
                "threshold_parameters": str(ms.best_threshold_params),
                "n_plates": len(ms.fe.dl.list_bpp),
                "balanced_accuracy": round(ms.test_scores["balanced_accuracy"],3),
                "anom_bad_segments_TP/(TP+FN)": anom_bad_segs,
                "anom_bad_segments_frac": round(ms.test_scores["recall_bad"],3),
                "anom_good_segments_FP/(FP+TN)": anom_good_segs,
                "anom_good_segments_frac": round(1-ms.test_scores["recall_good"],3),
                "mean_pred_snip_probas": str(ms.mean_pred_snip_probas),
            }

            try:
                logs_df = pd.read_excel(logs_file_path, sheet_name=sheet_name)
                new_logs_df = pd.concat([logs_df, pd.DataFrame([new_row])], ignore_index=True)
            except (FileNotFoundError):
                new_logs_df = pd.DataFrame([new_row])

            new_logs_df.to_excel(logs_file_path, sheet_name=sheet_name, index=False)

        print(f"Prediction with parameters {str(ms.best_model_params | ms.best_threshold_params)}:")

        print(f"Summary of performance (average of recall values):\t\t\t\t={round(100*ms.test_scores["balanced_accuracy"],2)}%")

        print(f"Anomalous segments correctly identified (recall: TP/(TP+FN)):\t{anom_bad_segs}\t\t={round(100*ms.test_scores["recall_bad"],2)}%")
        print(f"Anomalous segments mistakenly identified (FP/(FP+TN)):\t\t{anom_good_segs}\t\t={round(100*(1-ms.test_scores["recall_good"]),2)}%\n")

        if self.do_log_summary:

            summary_row = pd.DataFrame(index=[ms.random_seed], data={k:[v] for k,v in ms.test_scores.items()})
            if self.summary_results.empty:
                self.summary_results = summary_row
            else:
                self.summary_results = pd.concat([self.summary_results, summary_row], ignore_index=False)

        return
    
    def log_summary(self, sensor, random_seeds, n_plates, model="autoencoder"):
        """ TBC """

        if self.do_log_summary:

            summary_logs_file_path = os.path.join("sensor_tests","anomaly_logs","anomaly_logs_ae_sweeps.xlsx")
            sheet_name = "anomaly_detection_ae_bpp_sweeps"

            new_row = {
                "datetime": datetime.now(tz=ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S"),
                "notes": self.notes,
                "sensor": sensor,
                "random_seeds": random_seeds,
                "model": model,
                "n_plates": n_plates,
            }

            # Calculate column-wise statistics
            means = self.summary_results.mean()
            stds = self.summary_results.std()

            print("Summary Results Statistics:")
            print("=" * 40)
            for col in self.summary_results.columns:
                print(f"{col}: {means[col]:.3f} ± {stds[col]:.3f}")
                new_row[col] = f"{means[col]:.3f} ± {stds[col]:.3f}"
                
            try:
                logs_df = pd.read_excel(summary_logs_file_path, sheet_name=sheet_name)
                new_logs_df = pd.concat([logs_df, pd.DataFrame([new_row])], ignore_index=True)
            except (FileNotFoundError):
                new_logs_df = pd.DataFrame([new_row])

            new_logs_df.to_excel(summary_logs_file_path, sheet_name=sheet_name, index=False)