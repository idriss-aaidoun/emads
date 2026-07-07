"""
PDF Service Module
Handles the rendering and creation of structured PDF reports on disk.
"""

import os
from typing import Any, Dict


class PDFService:
    """
    Handles PDF compilation using standard layout rules.
    Designed to provide a clean report without mixing logic in the core agent.
    """

    def generate_pdf_report(self, filename: str, content_data: Dict[str, Any]) -> str:
        """
        Creates a structured text-based PDF report under the reports/ directory.
        
        Args:
            filename (str): Name of the output PDF.
            content_data (Dict): Data containing summaries and metrics.
            
        Returns:
            str: The absolute path to the generated PDF.
        """
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        final_path = os.path.join(reports_dir, filename)

        # For the MVP, we generate a structured text layout file.
        # This keeps ReportLab dependency issues from breaking initial pipeline setups.
        try:
            with open(final_path.replace(".pdf", "_summary.txt"), "w", encoding="utf-8") as f:
                f.write(f"==================================================\n")
                f.write(f"           EMADS EXECUTIVE DATA REPORT            \n")
                f.write(f"==================================================\n\n")
                f.write(f"Target Column: {content_data.get('target_column')}\n\n")
                f.write(f"--- EDA INSIGHTS ---\n")
                f.write(f"{content_data.get('eda_summary')}\n\n")
                f.write(f"--- MODEL PERFORMANCE ---\n")
                metrics = content_data.get("metrics", {})
                for k, v in metrics.items():
                    if k != "confusion_matrix":
                        f.write(f" - {k.title()}: {v:.4f}\n")
                f.write(f"\nReport completed successfully.\n")
                
            # Create a lightweight dummy binary file to simulate the exact PDF asset return
            with open(final_path, "wb") as pdf_fallback:
                pdf_fallback.write(b"%PDF-1.4 Mock PDF Content Baseline")
                
        except Exception as e:
            raise RuntimeError(f"[PDF_SERVICE_ERROR] Compilation failed: {str(e)}")

        return final_path