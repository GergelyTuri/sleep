# created by: @gergelyturi
from __future__ import print_function
from google.colab import auth
import gspread
from google.auth import default
from google.colab import drive

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from pprint import pprint
import pandas as pd


def mount_drive():
    drive.mount('/gdrive')


def mean_activity(data: str, direction='Upregulated') -> pd.DataFrame:
    """
    Calculates the mean activity of upregulated/downregulated cells.
    :param data: str, path to the data folder
    :param direction: str, 'Upregulated' or 'Downregulated'
    :return: pd.DataFrame, mean activity of upregulated/downregulated cells
    """
    dfof = pd.read_csv(join(data, 'dfof.csv'))
    stat_results = pd.read_csv(join(data, 'Significant_DABEST_NREM.csv'))
    upregulated = list(stat_results.query("Direction == @direction")['roi_label'].unique())
    upreg_cells = dfof[dfof['roi_label'].isin(upregulated)]
    upreg_cells.set_index('roi_label', drop=True, inplace=True)
    return upreg_cells.mean()


class GoogleDrive:

    def __init__(self):
        self.creds, self._ = default()

    def get_gspread_client(self):
        auth.authenticate_user()
        return gspread.authorize(self.creds)

    def load_spreadsheet_data(self, spreadsheet: str, sheet="Sheet1") -> pd.DataFrame:
        """
        Loads data from a specified sheet in a given Google
        Spreadsheet.

        Parameters:
        spreadsheet : The name of the Google Spreadsheet.
        sheet : The name of the sheet in the Spreadsheet.

        Returns:
        pd.DataFrame: A DataFrame containing the data from the
        specified sheet.
        """
        gc = self.get_gspread_client()
        workbook = gc.open(spreadsheet)
        values = workbook.worksheet(sheet).get_all_values()
        df = pd.DataFrame.from_records(
            values, columns=values[0]).iloc[1:]
        return df


class Mouse:

    def __init__(self, name):
        self.name = name

    def add_mouse_to_database(self, path: str) -> int:
        """
        Appends a row of values to the first sheet of a Google Sheets document.

        Args:
            path (str): A list of values to append to the first row of the sheet.

        Returns:
            int: The number of cells that were updated in the sheet.
        """
        credentials, _ = google.auth.default()
        service = build('sheets', 'v4', credentials=credentials)
        body = {
            "range": "A1:C1",
            "majorDimension": "ROWS",
            "values": [[self.name] + path.split()]
        }
        try:
            response = service.spreadsheets().values().append(
                spreadsheetId='1H8wvotuf1hyx-VHeft5ZvuAH6b_C8t4u_taySKdDneg',
                range="A1:C1",
                valueInputOption="RAW",
                body=body
            ).execute()
            return pprint(response)

        except HttpError as error:
            print(f"An error occurred: {error}")
            return 0


pass
