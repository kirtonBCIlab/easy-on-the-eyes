"""
    Set of functions to work with the Boccia software pilot data
"""

# Import libraries
import os
import mne
import pyxdf
import numpy as np
import pandas as pd
import scipy.signal as signal


def import_data(xdf_file: str) -> tuple[mne.io.Raw, pd.DataFrame, pd.DataFrame]:
    """
    Imports xdf file and returns the EEG stream, Python response, and Unity stream

    Parameters
    ----------
        xdf_file: str

    Returns
    -------
        eeg_mne: mne.io.Raw
            EEG data in an MNE format
        python_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-python backend
        unity_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-unity frontend
    """

    [xdf_data, _] = pyxdf.load_xdf(xdf_file)

    # First, determine first sample that needs to be subtracted from all streams
    for i in range(len(xdf_data)):
        stream_name = xdf_data[i]["info"]["name"][0]
        if (stream_name == "DSI24") | (stream_name == "DSI7"):
            first_sample = float(xdf_data[i]["footer"]["info"]["first_timestamp"][0])

    # Second, separate all streams and save data
    for i in range(len(xdf_data)):
        stream_name = xdf_data[i]["info"]["name"][0]

        # Data stream
        if (stream_name == "DSI24") | (stream_name == "DSI7"):
            eeg_np = xdf_data[i]["time_series"].T
            srate = float(xdf_data[i]["info"]["nominal_srate"][0])
            n_chans = len(xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"])
            ch_names = [
                xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"][c]["label"][0]
                for c in range(n_chans)
            ]
            ch_types = "eeg"

            # Drop trigger channel
            eeg_np = eeg_np[:-1, :]
            ch_names = ch_names[:-1]

            # Create MNE object
            info = mne.create_info(ch_names, srate, ch_types)
            eeg_mne = mne.io.RawArray(eeg_np, info)

        # Unity stream
        if stream_name == "PythonResponse":
            python_series = xdf_data[i]["time_series"]
            python_time_stamps = xdf_data[i]["time_stamps"]

            dict_python = {
                "Time stamps": np.array(python_time_stamps) - first_sample,
                "Signal value": np.zeros(len(python_time_stamps)),
                "Event": [event[0] for event in python_series],
            }

            python_stream = pd.DataFrame(dict_python)

        # Unity events
        if stream_name == "UnityEvents":
            unity_series = xdf_data[i]["time_series"]
            unity_time_stamps = xdf_data[i]["time_stamps"]

            dict_unity = {
                "Time stamps": np.array(unity_time_stamps) - first_sample,
                "Signal value": np.zeros(len(unity_time_stamps)),
                "Event": [event[0] for event in unity_series],
            }

            unity_stream = pd.DataFrame(dict_unity)

    return (eeg_mne, python_stream, unity_stream)


def create_epochs(
    eeg: mne.io.Raw, markers: pd.DataFrame, time: list[float], events=list[str]
) -> mne.Epochs:
    """
    Creates MNE epoch data from the desired markers (can be Unity or Python stream)

    Parameters
    ----------
        eeg: mne.io.fiff.raw.Raw
            EEG raw array in an MNE format
        markers: pd.DataFrame
            Markers used to creathe epochs
        time: list[float]
            Times used for epoch data, must be len==2
        events: list[str]
            List of strings with the events to use for the epochs

    Returns
    -------
        eeg_epochs: mne.Epochs
    """
    markers_np = np.zeros(markers.shape)
    markers_np[:, :2] = markers.iloc[:, [0, 1]]
    event_categories = pd.Categorical(markers["Event"])
    markers_np[:, 2] = event_categories.codes

    events_ids = np.zeros(markers.shape[0])
    for event in events:
        # - Missing implementation
        # We have the categories in the data frame, where the 3rd column can be set as categories
        # the code should look for the matching categories and provide just the indices that match the list of
        # events. Then, create the epochs based on the markers from the selected events

        # categories_of_interest = event_categories.categories.str.contains(event)
        # events_ids[markers_np[categories_of_interest.tolist(), :]] = 1
        break

    # eeg_epochs = mne.Epochs(eeg, )

    eeg_epochs = 0
    return eeg_epochs


# EKL Additions
def import_data_byType_noPython(
    xdf_file: str,
) -> tuple[mne.io.Raw, pd.DataFrame, pd.DataFrame]:
    """
    Imports xdf file and returns the EEG stream, Python response, and Unity stream

    Parameters
    ----------
        xdf_file: str

    Returns
    -------
        eeg_mne: mne.io.Raw
            EEG data in an MNE format
        python_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-python backend
        unity_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-unity frontend
    """

    [xdf_data, _] = pyxdf.load_xdf(xdf_file)

    # First, we need to figure out the general type of data so we know what is what
    for stream in xdf_data:
        stream_type = stream["info"]["type"]

    # First, determine first sample that needs to be subtracted from all streams
    for i in range(len(xdf_data)):
        stream_type = xdf_data[i]["info"]["type"][0]
        if stream_type == "EEG":
            first_sample = float(xdf_data[i]["footer"]["info"]["first_timestamp"][0])

    # Second, separate all streams and save data
    for i in range(len(xdf_data)):
        stream_type = xdf_data[i]["info"]["type"][0]
        stream_name = xdf_data[i]["info"]["name"][0]

        # Data stream
        if stream_type == "EEG":
            headset = xdf_data[i]["info"]["name"][0]
            eeg_np = xdf_data[i]["time_series"].T
            srate = float(xdf_data[i]["info"]["nominal_srate"][0])
            n_chans = len(xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"])
            ch_names = [
                xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"][c]["label"][0]
                for c in range(n_chans)
            ]
            ch_types = "eeg"

            # Drop trigger channel
            eeg_np = eeg_np[:-1, :]
            ch_names = ch_names[:-1]

            # Create MNE object
            info = mne.create_info(ch_names, srate, ch_types)
            eeg_mne = mne.io.RawArray(eeg_np, info)

        # Unity events
        if stream_type == "LSL_Marker_Strings":
            unity_series = xdf_data[i]["time_series"]
            unity_time_stamps = xdf_data[i]["time_stamps"]

            dict_unity = {
                "Time Stamps": np.array(unity_time_stamps) - first_sample,
                "Signal Value": np.zeros(len(unity_time_stamps)),
                "Event": [event[0] for event in unity_series],
            }

            unity_stream = pd.DataFrame(dict_unity)

    # Changing return right now as a hardcode - but there should be a way to optionally include the outputs
    return (eeg_mne, unity_stream)


def create_epochs_es(
    eeg: mne.io.Raw, markers: pd.DataFrame, events=list[str]
) -> mne.Epochs:
    """
    Documentation will go here one day.
    This function is hardcoded right now to support Emily Schrag's Summer project.
    """

    eeg_epochs = 0
    return eeg_epochs


def build_events_list_es(eeg: mne.io.Raw, markers: pd.DataFrame):
    """
    Documentation to be here one day.
    This just is a hardcoded build of a basic event list for Emily Schrag's summer project
    """
    unity_markers = markers["Event"]
    unity_time_stamps = markers["Time Stamps"]
    # using the below functions to find calls
    idx_markers = find_indices_of_repeated_values(unity_markers)
    idx_markers.sort()

    # Stimulus off calls happen every other block. So we will move those into their own index array, and not include the first one as it is before the stim started. Note that the first value has to be included for it to fold properly
    stim_off_markers = idx_markers[1:-1:2]
    # Make sure that the stim_off markers are true and only the stimulus off ones.
    for m in stim_off_markers:
        assert unity_markers[m] == "Stimulus Off"

    # Put them into a 2D matrix of shape Nx4, ignoring the first off call (pre-marker), so these now correspond to the "off" times between each of the 24 different blocks of stimulus
    grouped_stimOff_array = group_into_new_dimension(stim_off_markers, 4)

    # Repeat with the stim markers
    stim_markers = idx_markers[2:-1:2]
    # Ok ok ok. No we group this as well using our grouped call above, so we have each dimension as one of the "blocks", with the index for where each start of that block.
    grouped_stim = group_into_new_dimension(stim_markers, 4)

    # This should be length 24 right now
    # for i in range(0,len(ts_grouped_markers)):

    #     #Handle edge case here, cause the final one will require something different and we need data from the next value up in the groups.
    #     if i < len(ts_grouped_markers):
    #         nextOffVar = np.array(ts_grouped_offStim[i+1][0])
    #         offVar = ts_grouped_offStim[i][1:4]
    #         offVar = np.append(offVar,nextOffVar)
    #         #Create your local variables based on the outer loop
    #         stimVar = ts_grouped_markers[i]

    #         #Create an empty output for stim/off values for a given block
    #         output = [[],[]]

    #         #Now have an internal second loop grabbing the data and putting it into a different array
    #         for jj in range(len(stimVar)):
    #             output[0].extend(eeg[stimVar[jj]:offVar[jj]])
    #             if jj < len(stimVar)-1:
    #                 output[1].extend(eeg[offVar[jj]:stimVar[jj+1]])

    #         # print(ts_grouped_markers[i])
    #     else:
    #         #Handle the last case
    #         stimVar = ts_grouped_markers[i]
    #         offVar = ts_grouped_offStim[i][1:3]
    #         #Create an empty output for stim/off values for a given block
    #         output = [[],[]]
    #         #Now have an internal second loop grabbing the data and putting it into a different array
    #         for jj in range(len(offVar)):

    #             if jj < len(offVar):
    #                 output[0].extend(eeg[stimVar[jj]:offVar[jj]])
    #                 output[1].extend(eeg[offVar[jj]:stimVar[jj+1]])
    #             else:
    #                 output[0].extend(eeg[stimVar[jj]:-1])
    #                 output[1].extend(eeg[offVar[jj]:stimVar[jj+1]])

    return idx_markers, grouped_stim, grouped_stimOff_array


##EKL Additions
def find_indices_of_repeated_values(arr):
    result = []  # List to store the indices of the first occurrence
    previous_str = None

    for idx, item in enumerate(arr):
        item_str = str(item)  # Convert to string representation

        if item_str != previous_str:
            # If the current string is different from the previous one,
            # add the index to the result
            result.append(idx)

        previous_str = item_str  # Update the previous string

    return result


def find_indices_with_substring(strings_list, substring):
    indices = [idx for idx, string in enumerate(strings_list) if substring in string]
    return indices


def group_into_new_dimension(input_array, group_size):
    return [
        input_array[i : i + group_size] for i in range(0, len(input_array), group_size)
    ]


def epochs_from_unity_markers(eeg_time, eeg_data, marker_time, marker_data):
    """
    This function returns a list of EEG epochs and a list of marker names, based on
    the marker data provided.

    Notes
    -----
        The marker data must have repeated markers
    """
    # Initialize empty list
    eeg_epochs = []

    (repeated_markers, repeated_labels) = find_repeats(marker_data)

    # Trim EEG data to marker data times
    for m in range(np.shape(repeated_markers)[0]):
        eeg_mask_time = (eeg_time >= marker_time[repeated_markers[m, 0]]) & (
            eeg_time <= marker_time[repeated_markers[m, 1]]
        )

        eeg_epochs.append(eeg_data[eeg_mask_time, :])

    return (eeg_epochs, repeated_labels)


def find_repeats(marker_data: list) -> tuple((np.ndarray, list)):
    """
    Finds the repeated values in the marker data

    Returns
    -------
        - `repeats`: Numpy array with n-rows for repeated values [start, stop]
        - `order`: List with the `marker_data` labels of the repeated values.
    """

    repeats = []
    start = None

    for i in range(len(marker_data) - 1):
        if marker_data[i] == marker_data[i + 1]:
            if start is None:
                start = i
        elif start is not None:
            repeats.append((start, i))
            start = None

    if start is not None:
        repeats.append((start, len(marker_data) - 1))

    repeats = np.array(repeats)
    labels = [marker_data[i][0] for i in repeats[:, 0]]

    return repeats, labels
