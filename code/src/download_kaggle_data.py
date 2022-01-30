"""
Data Literacy - WiSe 21
Project: Assessment of Spotify Chart Preferences across Different Countries
Authors: Florian Kriegel (5746680), Jonathan Waehrer (5776535)

Purpose of this script is to download the 'Spotify 1.2M+ Songs' data set from kaggle. We pre-downloaded this data set
and uploaded it to a dropbox account. Using the dropbox api, the .csv will then be downloaded. We chose this approach,
because the kaggle API requires local changes on a PC for the authentication process which is inconvenient if a tutor
wants to run our code.

Link to data set:
https://www.kaggle.com/rodolfofigueroa/spotify-12m-songs
"""
# --------------------------------------------------- IMPORTS -------------------------------------------------------- #
import dropbox


# ---------------------------------------------- FUNCTIONS/UTILITY --------------------------------------------------- #
def download_from_dropbox(file_path_on_dropbox, filepath_to_save):
    """
    Downloads a file from dropbox and stores it.
    Parameters
    ----------
    file_path_on_dropbox: String
    filepath_to_save: String
    -------
    """
    # Dropbox API authentication
    dbx = dropbox.Dropbox('DmWiM696D8kAAAAAAAAAASGkgC1p72H97JqhjJiQlmtwmB4kseT2TOnphNIfbrS0')

    with open(filepath_to_save, "wb") as f:
        metadata, res = dbx.files_download(path=file_path_on_dropbox)
        f.write(res.content)


# ---------------------------------------------------- MAIN ---------------------------------------------------------- #
def main(dropbox_file, output):
    print("     ...downloading kaggle data set '{}'...".format(dropbox_file))
    download_from_dropbox(file_path_on_dropbox=dropbox_file, filepath_to_save=output)
    print("     ...saved kaggle data set under '{}'...".format(output))


if __name__ == "__main__":
    main()
