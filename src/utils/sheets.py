import pandas as pd
import random

def get_google_sheet(sheet_id: str, sheet_gid: str) -> pd.DataFrame:
	"""
	Downloads a specific sheet from a Google Sheet into a pandas DataFrame.

	Args:
		sheet_id: The ID of the Google Sheet.
		sheet_gid: The GID of the specific sheet to download.

	Returns:
		A pandas DataFrame containing the data from the specified sheet.
	"""
	url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}'
	df = pd.read_csv(url)
	return df

# TODO: Change with the valid values in the sheets
# Fill missing values in 'Vocab' column with 'TBA1'/'TBA2'/'TBA3' randomly
def fill_vocab(value):
	if pd.isna(value):
		return random.choice(['TBA1', 'TBA2', 'TBA3'])
	return value