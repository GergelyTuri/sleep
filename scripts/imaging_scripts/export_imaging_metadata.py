from os.path import join, dirname, isdir, exists, splitext
from os import walk
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from datetime import timedelta

import json


def convert_seconds_to_hms(duration):
    """Converts seconds to hours:minutes:seconds format"""
    try:
        seconds = timedelta(seconds=float(duration))
    except ValueError:
        print("Seconds must be a value which can be converted to number")
    hours = seconds.seconds // 3600
    minutes = (seconds.seconds // 60) % 60
    seconds = seconds.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_xml(xml_file: str):
    items_to_extract = {}
    # parse the XML file
    tree = ET.parse(xml_file)

    # get the root element
    root = tree.getroot()
    items_to_extract["recording date and time"] = root.attrib["date"]
    items_to_extract["notes"] = root.attrib["notes"]

    for child in root:
        if child.tag == "SystemIDs":
            for system_id in child:
                items_to_extract[system_id.tag] = system_id.attrib[system_id.tag]

    keys_to_extract = [
        "activeMode",
        "objectiveLens",
        "opticalZoom",
    ]
    for key in keys_to_extract:
        element = root.find(f".//PVStateValue[@key='{key}']")
        items_to_extract[key] = element.attrib["value"]

    # Saving the FPS which was used during recording
    sequence = root.find(".//Sequence")
    if sequence is not None:
        pv_state_value = sequence.find(f".//PVStateValue[@key='framePeriod']")
    if pv_state_value is not None:
        items_to_extract["frame_period"] = pv_state_value.attrib["value"]
        items_to_extract["fps"] = 1 / float(pv_state_value.attrib["value"])

    # finding the laser wavelength
    element = root.find(f".//PVStateValue[@key='laserWavelength']")
    tag = element.find("IndexedValue")
    items_to_extract["laserWavelength"] = tag.attrib["value"]

    # finding the x,y,z pixel sizes
    element = root.find(f".//PVStateValue[@key='micronsPerPixel']")
    tag = element.find("IndexedValue")
    for indexed_value in element:
        if indexed_value.attrib["index"] == "XAxis":
            items_to_extract["micronsPerPixel_X"] = indexed_value.attrib["value"]
        elif indexed_value.attrib["index"] == "YAxis":
            items_to_extract["micronsPerPixel_Y"] = indexed_value.attrib["value"]
        elif indexed_value.attrib["index"] == "ZAxis":
            items_to_extract["micronsPerPixel_Z"] = indexed_value.attrib["value"]

    element = root.find(f".//PVStateValue[@key='pmtGain']")
    tag = element.find("IndexedValue")
    for indexed_value in element:
        if indexed_value.attrib["index"] == "0":
            items_to_extract[
                indexed_value.attrib["description"]
            ] = indexed_value.attrib["value"]
        elif indexed_value.attrib["index"] == "1":
            items_to_extract[
                indexed_value.attrib["description"]
            ] = indexed_value.attrib["value"]

    sequences = root.findall(".//Sequence")
    # extracting data depending on plane number
    # single plane
    if len(sequences) == 1:
        frames = root.findall(".//Frame")
        items_to_extract["sequence_type"] = "single plane"
        items_to_extract["number of images"] = frames[-1].attrib["index"]
        duration = frames[-1].attrib["relativeTime"]
        items_to_extract["recording length in seconds"] = duration
        items_to_extract[
            "recording length (hours:minutes:seconds)"
        ] = convert_seconds_to_hms(duration)
    # multi plane
    else:
        items_to_extract["number of sequences"] = len(sequences)
        items_to_extract["sequence_type"] = "multi plane"
        # Frame info
        frames = sequences[0].findall("Frame")
        items_to_extract["number of planes"] = frames[-1].attrib["index"]
        items_to_extract["number of images/plane"] = sequences[-1].attrib["cycle"]
        # Recording length
        last_frame = sequences[-1].findall("Frame")
        duration = last_frame[-1].attrib["relativeTime"]
        items_to_extract["recording length in seconds"] = duration
        items_to_extract[
            "recording length (hours:minutes:seconds)"
        ] = convert_seconds_to_hms(duration)

    with open(join(dirname(xml_file), "imaging_metadata.json"), "w") as f:
        json.dump(items_to_extract, f, indent=4)

def main(directory: str):
    if not isdir(directory):
        raise ValueError(f"{directory} is not a directory")

    for root, dirs, files in walk(directory):
        for file in files:
            if file.endswith(".xml"):
                json_filename = 'imaging_metadata.json'
                json_filepath = join(root, json_filename)

                # Check if the JSON file already exists
                if exists(json_filepath):
                    print(f"JSON file already exists for {file}, skipping...")
                    continue

                print(f"Parsing {file}")
                parse_xml(join(root, file)) 
                print(f"Finished parsing {file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default="",
        help="Directory to to look for xml files",
    )
    args = parser.parse_args()

main(args.directory)
