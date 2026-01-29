
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd


class SBoxIO:
    """Handle S-box input/output in various formats."""
    
    # Paper-proposed matrix presets (K44, K128, K63)
    MATRIX_PRESETS = {
        'k44': np.array([
            [0, 1, 0, 1, 0, 1, 1, 1],  # 0x57
            [1, 0, 1, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 0]
        ], dtype=int),
        'k128': np.array([
            [1, 1, 1, 1, 1, 1, 1, 0],  # 0x1F
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1]
        ], dtype=int),
        'k63': np.array([
            [0, 0, 1, 1, 1, 1, 1, 1],  # 0x3F
            [1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0]
        ], dtype=int),
    }
    
    @staticmethod
    def get_matrix_preset(preset_name: str) -> Optional[np.ndarray]:
        """
        Get a matrix preset by name.
        
        Args:
            preset_name: Name of preset (k44, k128, k63)
        
        Returns:
            8x8 matrix or None if not found
        """
        return SBoxIO.MATRIX_PRESETS.get(preset_name.lower())
    
    @staticmethod
    def list_presets() -> List[str]:
        """Get list of available matrix presets."""
        return list(SBoxIO.MATRIX_PRESETS.keys())
    
    @staticmethod
    def save_sbox_json(sbox: np.ndarray, name: str, directory: str = 'sboxes') -> str:
        """
        Save S-box to JSON file.
        
        Args:
            sbox: S-box array
            name: Name for the S-box
            directory: Output directory
        
        Returns:
            Path to saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        filename = f"{name}.json" if not name.endswith('.json') else name
        filepath = os.path.join(directory, filename)
        
        data = {
            'name': name,
            'sbox': sbox.tolist(),
            'length': len(sbox),
            'type': 'custom' if len(sbox) != 256 else 'standard'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        return filepath
    
    @staticmethod
    def load_sbox_json(filepath: str) -> np.ndarray:
        """
        Load S-box from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            S-box array
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"S-box file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return np.array(data['sbox'], dtype=np.uint8)
        except Exception as e:
            raise ValueError(f"Error loading S-box: {e}")
    
    @staticmethod
    def save_sbox_excel(
        sbox: np.ndarray,
        name: str,
        metrics: Optional[Dict] = None,
        directory: str = 'sboxes'
    ) -> str:
        """
        Save S-box and metrics to Excel file.
        
        Args:
            sbox: S-box array
            name: Name for the S-box
            metrics: Optional dictionary of metrics
            directory: Output directory
        
        Returns:
            Path to saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        filename = f"{name}.xlsx" if not name.endswith('.xlsx') else name
        filepath = os.path.join(directory, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # S-box sheet
            sbox_df = pd.DataFrame({
                'Index': np.arange(256),
                'S-box Value': sbox
            })
            sbox_df.to_excel(writer, sheet_name='S-box', index=False)
            
            # Visualization of S-box as 16x16 grid
            sbox_grid = sbox.reshape(16, 16)
            grid_df = pd.DataFrame(sbox_grid, columns=[f"Col_{i}" for i in range(16)])
            grid_df.to_excel(writer, sheet_name='Grid', index=True)
            
            # Metrics sheet if provided
            if metrics:
                metrics_data = []
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_data.append({'Metric': key, 'Value': value})
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        return filepath
    
    @staticmethod
    def load_sbox_excel(filepath: str) -> np.ndarray:
        """
        Load S-box from Excel file.
        
        Args:
            filepath: Path to Excel file
        
        Returns:
            S-box array
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"S-box file not found: {filepath}")
        
        try:
            df = pd.read_excel(filepath, sheet_name='S-box')
            return np.array(df['S-box Value'], dtype=np.uint8)
        except Exception as e:
            raise ValueError(f"Error loading S-box: {e}")
    
    @staticmethod
    def batch_save_sboxes(
        sboxes: Dict[str, Tuple[np.ndarray, Dict]],
        directory: str = 'sboxes'
    ) -> Dict[str, str]:
        """
        Save multiple S-boxes with their metrics.
        
        Args:
            sboxes: Dictionary mapping names to (sbox, metrics) tuples
            directory: Output directory
        
        Returns:
            Dictionary mapping names to file paths
        """
        results = {}
        
        for name, (sbox, metrics) in sboxes.items():
            filepath = SBoxIO.save_sbox_excel(sbox, name, metrics, directory)
            results[name] = filepath
        
        return results
    
    @staticmethod
    def save_comparison_report(
        comparisons: Dict[str, Dict],
        output_path: str = 'sboxes/comparison_report.xlsx'
    ) -> str:
        """
        Save comparison report of multiple S-boxes.
        
        Args:
            comparisons: Dictionary of S-box comparison results
            output_path: Output file path
        
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        comparison_data = []
        for sbox_name, metrics in comparisons.items():
            row = {'S-box Name': sbox_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df.to_excel(output_path, index=False, sheet_name='Comparison')
        
        return output_path
    
    @staticmethod
    def create_matrix_from_input(matrix_input: str, constant_input: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create matrix and constant vector from string input.
        
        Args:
            matrix_input: 8x8 binary matrix as string
            constant_input: 8-element binary vector as string
        
        Returns:
            Tuple of (matrix, constant_vector)
        """
        try:
            # Parse matrix input
            matrix_lines = [line.strip() for line in matrix_input.strip().split('\n') if line.strip()]
            
            if len(matrix_lines) != 8:
                raise ValueError("Matrix must have 8 rows")
            
            matrix = []
            for line in matrix_lines:
                row = [int(x) for x in line.split() if x in ['0', '1']]
                if len(row) != 8:
                    raise ValueError("Each row must have 8 binary values")
                matrix.append(row)
            
            matrix = np.array(matrix, dtype=int)
            
            # Parse constant vector
            const_values = [int(x) for x in constant_input.split() if x in ['0', '1']]
            if len(const_values) != 8:
                raise ValueError("Constant vector must have 8 binary values")
            
            constant = np.array(const_values, dtype=int)
            
            return matrix, constant
        
        except ValueError as e:
            raise ValueError(f"Error parsing matrix/constant: {e}")
    
    @staticmethod
    def matrix_to_string(matrix: np.ndarray) -> str:
        """
        Convert matrix to string representation.
        
        Args:
            matrix: 8x8 matrix
        
        Returns:
            String representation
        """
        lines = []
        for row in matrix:
            lines.append(' '.join(str(int(x)) for x in row))
        return '\n'.join(lines)
    
    @staticmethod
    def constant_to_string(constant: np.ndarray) -> str:
        """
        Convert constant vector to string representation.
        
        Args:
            constant: 8-element vector
        
        Returns:
            String representation
        """
        return ' '.join(str(int(x)) for x in constant)
