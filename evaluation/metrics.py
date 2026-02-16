"""
Evaluation Metrics for Cognitive Smart Grid
Comprehensive evaluation including forecasting,
peak detection, and demand response optimization.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SmartGridEvaluator:
    """
    Comprehensive evaluation framework for
    Cognitive Smart Grid system.
    """

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Forecasting Evaluation
    # ------------------------------------------------------------------
    def evaluate_forecasting(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """Evaluate forecasting performance"""

        # Prevent division by zero in MAPE
        epsilon = 1e-8
        y_true_safe = np.where(y_true == 0, epsilon, y_true)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        r2 = r2_score(y_true, y_pred)

        self.results[f"{model_name}_forecasting"] = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "MAPE (%)": round(float(mape), 2),
            "R2": round(float(r2), 3),
        }

        return self.results[f"{model_name}_forecasting"]

    # ------------------------------------------------------------------
    # Peak Detection Evaluation
    # ------------------------------------------------------------------
    def evaluate_peak_detection(
        self,
        y_true_peak: np.ndarray,
        y_pred_peak: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate binary peak detection performance"""

        tp = np.sum((y_true_peak == 1) & (y_pred_peak == 1))
        fp = np.sum((y_true_peak == 0) & (y_pred_peak == 1))
        tn = np.sum((y_true_peak == 0) & (y_pred_peak == 0))
        fn = np.sum((y_true_peak == 1) & (y_pred_peak == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(y_true_peak)

        self.results["peak_detection"] = {
            "Precision": round(float(precision), 3),
            "Recall": round(float(recall), 3),
            "F1-Score": round(float(f1), 3),
            "Accuracy": round(float(accuracy), 3),
        }

        return self.results["peak_detection"]

    # ------------------------------------------------------------------
    # Demand Response Evaluation
    # ------------------------------------------------------------------
    def evaluate_demand_response(
        self,
        baseline_load: np.ndarray,
        optimized_load: np.ndarray,
        prices: np.ndarray,
        carbon: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate demand response optimization impact"""

        load_reduction = baseline_load - optimized_load

        peak_reduction_pct = (
            (np.max(baseline_load) - np.max(optimized_load))
            / np.max(baseline_load)
            * 100
        )

        baseline_cost = np.sum(baseline_load * prices)
        optimized_cost = np.sum(optimized_load * prices)
        cost_savings_pct = (
            (baseline_cost - optimized_cost) / baseline_cost * 100
        )

        baseline_carbon = np.sum(baseline_load * carbon)
        optimized_carbon = np.sum(optimized_load * carbon)
        carbon_savings_pct = (
            (baseline_carbon - optimized_carbon) / baseline_carbon * 100
        )

        self.results["demand_response"] = {
            "Average_Load_Reduction_kW": round(
                float(np.mean(load_reduction)), 2
            ),
            "Peak_Reduction_%": round(float(peak_reduction_pct), 2),
            "Cost_Savings_%": round(float(cost_savings_pct), 2),
            "Carbon_Savings_%": round(float(carbon_savings_pct), 2),
            "Total_Cost_Savings_$": round(
                float(baseline_cost - optimized_cost), 2
            ),
        }

        return self.results["demand_response"]

    # ------------------------------------------------------------------
    # Generate Text Report
    # ------------------------------------------------------------------
    def generate_report(
        self,
        output_file: str = "./outputs/reports/evaluation_report.txt",
    ) -> str:
        """Generate comprehensive evaluation report"""

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("COGNITIVE SMART GRID - EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        # Forecasting
        for key, metrics in self.results.items():
            if "forecasting" in key:
                report_lines.append("\n" + "=" * 70)
                report_lines.append("FORECASTING PERFORMANCE")
                report_lines.append("=" * 70)
                report_lines.append(f"\n{key}:")
                for metric, value in metrics.items():
                    report_lines.append(f"  {metric:20s}: {value}")

        # Peak Detection
        if "peak_detection" in self.results:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("PEAK DETECTION PERFORMANCE")
            report_lines.append("=" * 70)
            for metric, value in self.results["peak_detection"].items():
                report_lines.append(f"  {metric:20s}: {value}")

        # Demand Response
        if "demand_response" in self.results:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("DEMAND RESPONSE OPTIMIZATION")
            report_lines.append("=" * 70)
            for metric, value in self.results["demand_response"].items():
                report_lines.append(f"  {metric:30s}: {value}")

        # Summary
        if "demand_response" in self.results:
            dr = self.results["demand_response"]
            report_lines.append("\n" + "=" * 70)
            report_lines.append("SUMMARY")
            report_lines.append("=" * 70)
            report_lines.append(
                f"\n[OK] Peak reduction achieved: {dr['Peak_Reduction_%']:.1f}%"
            )
            report_lines.append(
                f"[OK] Cost savings achieved: {dr['Cost_Savings_%']:.1f}%"
            )
            report_lines.append(
                f"[OK] Carbon reduction achieved: {dr['Carbon_Savings_%']:.1f}%"
            )

        report_lines.append("\n" + "=" * 70)

        report_text = "\n".join(report_lines)

        # UTF-8 SAFE WRITING
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to: {output_file}")

        return report_text

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    def save_json(
        self,
        output_file: str = "./outputs/reports/evaluation_results.json",
    ) -> None:
        """Save results in JSON format"""

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        print(f"JSON results saved to: {output_file}")


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("RUNNING EXAMPLE EVALUATION")
    print("=" * 70 + "\n")

    np.random.seed(42)

    evaluator = SmartGridEvaluator()

    # Forecasting Example
    y_true = np.random.normal(650, 150, 1000)
    y_pred = y_true + np.random.normal(0, 20, 1000)

    print("Evaluating forecasting...")
    print(evaluator.evaluate_forecasting(y_true, y_pred, "LSTM"))

    # Peak Detection Example
    print("\nEvaluating peak detection...")
    threshold = np.percentile(y_true, 95)
    y_true_peak = (y_true > threshold).astype(int)
    y_pred_peak = (y_pred > threshold).astype(int)
    print(evaluator.evaluate_peak_detection(y_true_peak, y_pred_peak))

    # Demand Response Example
    print("\nEvaluating demand response...")
    optimized_load = y_true * 0.85
    prices = np.random.uniform(0.08, 0.20, 1000)
    carbon = np.random.uniform(200, 600, 1000)

    print(
        evaluator.evaluate_demand_response(
            y_true, optimized_load, prices, carbon
        )
    )

    print("\nGenerating report...")
    evaluator.generate_report()
    evaluator.save_json()
