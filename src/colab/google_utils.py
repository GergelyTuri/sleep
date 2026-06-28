# created by: @gergelyturi
"""Functions designed to interact with Google Sheets and Google Drive.
still WIP and mostly untested."""
from __future__ import print_function

import logging
import os
from pathlib import Path
from pprint import pprint

import google.auth
import gspread
import pandas as pd
from google.auth import default

try:
    from google.colab import auth, drive
except ImportError:
    auth = None  # not running in Colab
    drive = None

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    from src.config import GDRIVE_ROOT as _GDRIVE_ROOT
except ImportError:
    from config import GDRIVE_ROOT as _GDRIVE_ROOT

logger = logging.getLogger(__name__)


class GoogleDrive:
    """functions for interacting with Google Drive"""

    def __init__(self):
        self.creds, self._ = default()

    def mount_drive(self):
        """Mounts the Google Drive to the Colab notebook."""
        drive.mount('/gdrive')

    def get_gspread_client(self):
        """Function for authenticating a Google Sheets client."""
        auth.authenticate_user()
        return gspread.authorize(self.creds)
    
    def shared_drive_data_path(self):
        """Returns the path of the Data folder of the Turi_lab Google Drive."""
        return Path(_GDRIVE_ROOT)

    def load_spreadsheet_data(self, spreadsheet: str, sheet: str="Sheet1") -> pd.DataFrame:
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
    """
    A class representing a mouse. Supposed to read and write a google spreadsheet. WIP.
    """

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
            logger.error("An error occurred: %s", error)
            return 0

class MouseDatabase(GoogleDrive):
    """A class representing a database of mice. The database is a Google Sheets document."""

    def __init__(self, sheet: str="Sheet1"):
        super().__init__()
        self.spreadsheet = "useful datasets"
        self.sheet = sheet

    def load_mouse_database(self):
        """Loads mouse data from the Google Sheets database.
        Initialize like this:
        >>> from base import core
        >>> core.mount_drive()
        >>> db = core.MouseDatabase()
        >>> mice = db.load_spreadsheet_data()
        >>> mice.head()
        """
        gc = self.get_gspread_client()
        workbook = gc.open(self.spreadsheet)
        values = workbook.worksheet(self.sheet).get_all_values()
        all_mice = pd.DataFrame.from_records(
            values, columns=values[0]).iloc[1:]
        return all_mice
    
    def add_mouse_to_database(self, values: list) -> int:
        """
        Appends a row of values to the first sheet of a Google Sheets document.

        Args:
            values (list): A list of values to append to the first row of the sheet.

        Returns:
            int: The number of cells that were updated in the sheet.
        """
        gc = self.get_gspread_client()
        sheet = gc.open(self.spreadsheet).sheet1
        row_to_append = [self.__class__.__name__] + values
        sheet.append_row(row_to_append)
        return len(row_to_append)
    