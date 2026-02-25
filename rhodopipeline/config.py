# rhodopipeline/config.py
"""
Central configuration constants extracted from both source notebooks.

Edit the values under 'paths' to match your Google Drive layout before
running any notebook on Colab.
"""

CONFIG = {
    'paths': {
        # Environmental logger CSV (hourly reef-flat temperatures)
        'wally_csv': '/content/drive/MyDrive/PhD data/Environmental data/wally_hourly_reef_flat_temps.csv',
        # Base directory for microCT density CSV files
        'microct_base': '/content/drive/MyDrive/PhD data/microCT/measurement data/',
        # Directory where output CSVs and figures are saved
        'output_dir': '/content/drive/MyDrive/PhD data/Rhodolith papers/',
    },
    'sheets': {
        # Google Sheets workbook name for laser transect data
        'laser_workbook': 'Laser transects',
    },
    'dates': {
        # Rhodolith deployment window
        'deploy_start': '2024-03-16',
        'deploy_end':   '2024-07-23',
        # Start of the calibration period (post-deployment settlement)
        'calib_start':  '2024-04-16',
        # Alizarin stain date (start of the microCT growth window)
        'stain_date':   '2024-03-13',
    },
    'dtw': {
        # Sakoe–Chiba psi parameter (0 = no relaxation at endpoints)
        'psi': 0,
    },
    # Minimum number of valid laser-transect spots required per branch
    'min_spots': 10,
}

COLORS = {
    'Mg/Sr': 'magenta',
    'Mg/Ca': 'green',
    'Sr/Ca': 'tab:blue',
}
