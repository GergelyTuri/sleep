#############################################################
# this script is used to pickls TDML files. This script was #
# taken from the old repo. Compatibility with lab3 needs to #
# be tested                                                 #
# GT 3/12 
# 1/21/2024 processed files can be saved as JSON            #
#############################################################

import argparse
import json
import logging
import os.path
import sys
import time
import pickle
from datetime import datetime

import numpy as np

exts = ['.vr', '.tdml']


def parseTime(time_string):
    time_formats = ['%Y-%m-%d %H:%M:%S', '%d-%b-%Y %H:%M:%S',
                    '%Y-%m-%d%H:%M:%S']
    for frmt in time_formats:
        try:
            return time.strptime(time_string, frmt)
        except ValueError:
            pass


def findFiles(directory, overwrite=False):
    """
    Find files in a directory.

    Args:
        directory (str): The directory to search for files.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    Returns:
        list: A list of file paths found in the directory.
    """
    paths = []
    json_paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        paths.extend(
            map(lambda f: os.path.join(dirpath, f),
                filter(lambda f: os.path.splitext(f)[1] in exts, filenames)))
        json_paths.extend(
            map(lambda f: os.path.join(dirpath, os.path.splitext(f)[0]),
                filter(lambda f: os.path.splitext(f)[1] == '.json', filenames)))

    if not overwrite:
        paths = filter(
            lambda f: os.path.splitext(f)[0] not in json_paths, paths)

    return paths


def loadSql(filename, trial_info, settings, experiment_info=None,
            trial_id=None):
    """
    Loads the data for a trial into the database.

    Args:
        filename (str): The filename of the trial.
        trial_info (dict): Information about the trial.
        settings (dict): Settings for the trial.
        experiment_info (dict, optional): Information about the experiment. Defaults to None.
        trial_id (int, optional): The ID of the trial. Defaults to None.
    """
    from lab3.experiment import database

    database.update_trial(
        filename, trial_info['mouse_name'], trial_info['start_time'],
        trial_info['stop_time'], trial_info['experiment_group'],
        trial_id=trial_id)

    if trial_id is None:        
        trial_id = database.fetch_trial_id(filename)
    else:
        database.delete_all_trial_attrs(trial_id)

    print('loading trial {}: {}'.format(trial_id, filename))

    for key in settings:
        if '_controller' in key:
            continue
        if settings[key] is None:
            continue
        if isinstance(settings[key], dict) or isinstance(settings[key], list):
            if key == 'contexts':
                for context in settings[key]:
                    if context.get('class') == 'vr':
                        settings[key].remove(context)
                        database.update_trial_attr(trial_id, 'vr_context',
                                                 json.dumps(context))
            settings[key] = json.dumps(settings[key])
        database.update_trial_attr(trial_id, key, settings[key])

    if 'laps' in trial_info.keys():
        database.update_trial_attr(trial_id, 'laps', trial_info['laps'])

    if experiment_info is not None:
        database.insert_into_experiment(
            trial_id, experiment_info['start_time'],
            experiment_info.get('stop_time', None))

    # for now, imaging layer will be in settings block as a single attribute,
    # but in the future we should make an imaging block with places to be
    # filled out in the UI and maybe we can break up the settings with:
    # experiment settings, permanent rig settings and imaging settings

def numpy_to_list(obj) -> list:
    """
    Recursively converts numpy arrays in the given object to lists.
    This step is necessary before saving the object as JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(v) for v in obj]
    else:
        return obj

 
def main(argv):
    """
    Converts behavior JSON files to `.pkl` and adds them to the database.
    Compatible with older version of BehaviorMate (0.0.6b)

    Args:
        argv (list): List of command-line arguments passed to the script.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    argParser = argparse.ArgumentParser(description="Converts behavior JSON files"\
                                            " to `.pkl` and adds them to the database. "\
                            "Compatible with older version of BehaviorMate (0.0.6b)")

    argParser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='overwirte existing .pkl file')
    argParser.add_argument(
        '-d', '--directory', type=str,
        help='process all .vr or .tdml files beneath this directory')
    argParser.add_argument(
        '-f', '--file', action='store', type=str,
        help='process specifically this file')
    argParser.add_argument(
        '-s', '--sql', action='store_true',
        help='load this experiment into the SQL database')
    argParser.add_argument(
        '-g', '--group', action='store', type=str,
        help='group this experiment belongs to')
    argParser.add_argument(
        '-t', '--trial_id', action='store', type=int,
        help='overwrite the data for a specified trial_id, must be used when \
              processing a single file. requires --sql.')
    argParser.add_argument(
        '-ft', '--file_type', action='store', type=str, default='json',
        help='output file type: "json" (default) or "pkl"')
    args = argParser.parse_args(argv)

    def _append(adict, key, val):
        try:
            adict[key].append(val)
        except KeyError:
            adict[key] = [val]

    if args.directory is not None:
        files = findFiles(args.directory, overwrite=args.overwrite)
    else:
        files = [args.file]

    sqltime_format = '%Y-%m-%d %H:%M:%S'
    if args.trial_id is not None and len(files) != 1:
        raise Exception('when specifying a trial id only a single file may be \
                        processed.')
    for fname in files:
        mdict = {}
        trial_info = {'experiment_group': None}
        inReward = False
        inlick = False
        valve_start = -1
        starttime_offset = 0
        print('processing ' + fname)

        with open(fname) as f:
            for line in f:
                data = json.loads(line)

                if 'settings' in data.keys():
                    break

        settings = data['settings']
        contexts = {c['id']: c for c in settings.get('contexts', [])}
        reward_contexts = filter(lambda n: 'reward' in n, contexts.keys())
        reward_contexts = list(reward_contexts)

        trial_info['reward_pins'] = set(filter(
            lambda p: p is not None, [contexts[c].get('valves', [None])[0] for
                                      c in reward_contexts]))

        if settings.get('reward') is not None:
            reward_id = settings['reward'].get('id', 'hidden_reward')
            if reward_id not in reward_contexts:
                reward_contexts.append(reward_id)
                contexts[reward_id] = settings["reward"]
                if "valves" in settings["reward"]:
                    trial_info["reward_pins"].add(
                        settings["reward"].get("valves")[0])
                else:
                    trial_info["reward_pins"].add(
                        settings["reward"].get("pin"))

        if 'behavior_controller' in settings:
            behavior_address = str(settings['behavior_controller']['ip']) + ':' + \
                str(settings['behavior_controller']['receive_port'])
        else:
            behavior_address = str(settings['controllers']['behavior_controller']['ip']) + ':' + \
                str(settings['controllers']['behavior_controller']['receive_port'])

        if 'position_controller' in settings:
            _pos_ctrl = settings['position_controller']
        else:
            _pos_ctrl = settings['controllers']['position_controller']
        position_address = str(_pos_ctrl['ip']) + ':' + str(_pos_ctrl['receive_port'])

        stims = {}
        if settings.get('contexts') is not None:
            for context in settings['contexts']:
                for stim in context.get('stims', []):
                    if stim['address'] == 'behavior_controller':
                        stims[stim['pin']] = stim['name']

        for stim in settings.get('stims', []):
            if stim['address'] == 'behavior_controller':
                stims[stim['pin']] = stim['name']

        trial_info['sync_pin'] = settings['sync_pin']

        behavior_address_valid = False
        with open(fname) as f:
            for line in f:
                data = json.loads(line)
                if behavior_address in data.keys():
                    behavior_address_valid = True
                    break
        if not behavior_address_valid:
            if 'behavior_controller' in settings:
                behavior_address = str(settings['behavior_controller']['ip']) + \
                    ':' + str(settings['behavior_controller']['send_port'])
            else:
                behavior_address = str(settings['controllers']['behavior_controller']['ip']) + ':' + \
                str(settings['controllers']['behavior_controller']['send_port'])
            logging.warning("received messages from send_port for file: %s", fname)

        experiment_info = None
        starts = {}
        last_ts = 0

        starttime_offset_set = False
        context_starts = {}
        with open(fname) as f:
            for line in f:
                data = json.loads(line)
                if ("trial_start" not in data.keys()) and (
                        "trial_end" not in data.keys()):
                    tStamp = data.get("time", -1)
                    if tStamp > last_ts:
                        last_ts = tStamp

                if 'mouse' in data.keys():
                    trial_info['mouse_name'] = data['mouse']

                if 'experiment_group' in data.keys():
                    trial_info['experiment_group'] = data['experiment_group']

                if 'start' in data.keys():
                    start_time = parseTime(data['start'])
                    trial_info['start_time'] = time.strftime(
                        sqltime_format, start_time)

                if 'experiment_start' in data.keys():
                    start_time = parseTime(data['experiment_start'])
                    experiment_info = {
                        'start_time': time.strftime(sqltime_format, start_time)
                    }

                if 'experiment_stop' in data.keys() and \
                        experiment_info is not None:
                    stop_time = parseTime(data['experiment_stop'])
                    experiment_info['stop_time'] = \
                        time.strftime(sqltime_format, stop_time)

                if 'tone' in data.keys():
                    if data['tone'] == 'start' and \
                            starts.get('tone', -1) == -1:
                        starts['tone'] = data['time']
                    elif data['tone'] == 'stop' and \
                            starts.get('tone', -1) != -1:
                        _append(mdict, 'tone', [starts['tone'], data['time']])
                        starts['tone'] = -1

                if 'stop' in data.keys():
                    stop_time = parseTime(data['stop'])
                    trial_info['stop_time'] = time.strftime(
                        sqltime_format, stop_time)

                if 'lap' in data.keys():
                    _append(mdict, 'lap', data['time'])

                if 'comments' in data.keys():
                    settings['comments'] = data['comments']

                if 'y' in data.keys():
                    _append(mdict, 'position_y', [data['time'],
                            float(data['y'])])
                    _append(mdict, 'treadmillDy', [data['time'],
                            int(data[position_address]['position']['dy'])])
                elif behavior_address in data.keys():
                    message = data[behavior_address]
                    if message.get('context', None) is not None:
                        context_message = message['context']
                        this_context = str(context_message.get('id', ''))
                        if this_context in reward_contexts:
                            if context_message.get('action', '') == 'start':
                                reward_start = data['time']
                                inReward = True
                            elif inReward and context_message.get('action', '') \
                                    == 'stop':
                                _append(mdict, 'reward',
                                        [reward_start, data['time']])
                                inReward = False

                        if this_context != '':
                            if context_message.get('action', '') == 'start':
                                context_starts[this_context] = data['time']
                            elif context_message.get('action', '') == 'stop' and \
                                    this_context in context_starts.keys():
                                _append(mdict, this_context,
                                        [context_starts[this_context],
                                         data['time']])
                                del context_starts[this_context]

                        for acontext in reward_contexts:
                            if context_message.get(acontext, '') == 'start':
                                reward_start = data['time']
                                inReward = True
                            elif inReward and context_message.get(
                                    acontext, '') == 'stop':
                                _append(mdict, 'reward', [reward_start,
                                                          data['time']])
                                inReward = False

                    if 'lick' in message.keys():
                        if message['lick']['action'] == 'start':
                            lick_start = data['time']
                            inlick = True
                        elif inlick and message['lick']['action'] == 'stop':
                            _append(mdict, 'licking',
                                    [lick_start, data['time']])
                            inlick = False

                    elif not set(map(
                            str, (trial_info['reward_pins']))).isdisjoint(
                                set(message.get('valve', {}).keys())):
                        pin = set(map(
                            str, (trial_info['reward_pins']))).intersection(
                                set(message.get('valve', {}).keys())).pop()
                        if message['valve'][pin] == "open":
                            valve_start = data['time']
                        elif valve_start != -1 and \
                                message['valve'][pin] == "close":
                            _append(mdict, 'water', [valve_start,
                                                     data['time']])
                            valve_start = -1

                    elif str(trial_info.get('sync_pin', -1)) in message.get(
                            'valve', {}).keys():
                        if message['valve'][str(
                                trial_info['sync_pin'])] == "open":
                            if not starttime_offset_set:
                                starttime_offset = data['time']
                                starttime_offset_set = True

                    if "tone" in message.keys():
                        if message["tone"]["action"] == "open" and starts.get(
                                "tone", -1) == -1:
                            starts["tone"] = data["time"]
                        elif message["tone"]["action"] == "close" and \
                                starts.get("tone", -1) != -1:
                            _append(mdict, "tone", [starts["tone"],
                                                    data["time"]])
                            starts["tone"] = -1

                    elif 'valve' in message.keys():
                        pin = message["valve"].get("pin", -10)
                        if pin in stims.keys():
                            if message['valve']["action"] == 'open' and \
                                    starts.get(stims[pin], -1) == -1:
                                starts[stims[pin]] = data['time']
                            elif message['valve']["action"] == 'close' and \
                                    starts.get(stims[pin], -1) != -1:
                                _append(mdict, stims[pin], [starts[stims[pin]],
                                                            data['time']])
                                starts[stims[pin]] = -1
                        elif pin in trial_info['reward_pins']:
                            if message['valve'].get('action', '') == "open":
                                valve_start = data['time']
                            elif valve_start != -1 and \
                                    message['valve'].get(
                                        'action', '') == "close":
                                _append(mdict, 'water', [valve_start,
                                                         data['time']])
                                valve_start = -1
                        elif pin == trial_info.get('sync_pin', -1):
                            if message['valve'].get(
                                    'action', 'close') == "open":
                                if not starttime_offset_set:
                                    starttime_offset = data['time']
                                    starttime_offset_set = True
                        else:
                            for key in message['valve'].keys():
                                try:
                                    key = int(key)
                                except:
                                    pass
                                else:
                                    if key in stims.keys():
                                        if message['valve'][str(key)] == \
                                                'open' and starts.get(
                                                    stims[key], -1) == -1:
                                            starts[stims[key]] = data['time']
                                        elif message['valve'][str(key)] == \
                                                'close' and starts.get(
                                                    stims[key], -1) != -1:
                                            _append(mdict, stims[key],
                                                    [starts[stims[key]],
                                                    data['time']])
                                            starts[stims[key]] = -1

        if 'treadmillDy' in mdict:
            assert len(mdict['treadmillDy']) == len(mdict['position_y']), (
                "treadmillDy and position_y have different lengths (%d vs %d) — "
                "dy collection is misaligned with position events"
                % (len(mdict['treadmillDy']), len(mdict['position_y']))
            )

        if starttime_offset != 0:
            for key in mdict.keys():
                if key == 'position_y':
                    position_array = np.array(mdict[key])
                    try:
                        timeArr = position_array[:, 0]
                        minTimeD = np.min(np.diff(timeArr))
                        if minTimeD > 0.05:
                            minTimeD = 0.05
                        posArr = position_array[:, 1]
                        position_array = np.vstack(
                            [position_array, np.vstack([timeArr[1:] - minTimeD, posArr[:-1]]).T])
                        sortInds = np.argsort(position_array[:, 0])
                        position_array = position_array[sortInds, :]

                        selBA = ((np.diff(position_array[:, 0]) > 0.1)
                                 | (np.abs(np.diff(position_array[:, 1])) > 0))
                        selBA = np.hstack([True, selBA])
                        position_array = position_array[selBA, :]
                    except:
                        pass

                    if np.any(position_array[:, 0] > starttime_offset):
                        position_array[:, 0] -= starttime_offset
                        position_array = position_array[
                            np.where(position_array[:, 0] >= 0)]
                    else:
                        position_array = np.array([
                            [0, position_array[-1, 1]],
                            [last_ts - starttime_offset,
                             position_array[-1, 1]]])
                    mdict[key] = position_array.tolist()
                elif key == 'treadmillDy':
                    dy_arr = np.array(mdict[key])
                    if np.any(dy_arr[:, 0] > starttime_offset):
                        dy_arr[:, 0] -= starttime_offset
                        dy_arr = dy_arr[dy_arr[:, 0] >= 0]
                    mdict[key] = dy_arr.tolist()
                else:
                    mdict[key] = (np.array(mdict[key]) -
                                  starttime_offset).tolist()

        mdict['trackLength'] = float(settings['track_length'])
        mdict['position_scale'] = float(settings['position_scale'])
        mdict['recordingDuration'] = settings.get(
            'trial_length', last_ts - starttime_offset)
        if settings.get("experimentType", "") == "salience":
            mdict["recordingDuration"] = last_ts - starttime_offset

        if "position_y" in mdict.keys():
            mdict['treadmillPosition'] = np.array(mdict['position_y'])
            treadpos = mdict['treadmillPosition']
            if np.any(treadpos[:, 1] > 0):
                treadpos[:, 1] /= mdict['trackLength']
                treadpos[:, 1] %= 1

            if (treadpos.size > 0) and (treadpos[0, 0] != 0):
                treadpos = np.vstack(([0, treadpos[0, 1]], treadpos))

            # Remove redundant times, keeping last position at that time
            if treadpos.size > 0:
                good_times = np.hstack([np.diff(treadpos[:, 0]) != 0, True])
                treadpos = treadpos[good_times]

            mdict['treadmillPosition'] = treadpos

        trial_info['laps'] = len(mdict.get('lap', []))
        if 'stop_time' not in trial_info.keys():
            time_format = '%Y-%m-%d %H:%M:%S'
            start_ti = time.mktime(
                time.strptime(trial_info['start_time'], time_format))
            trial_info['stop_time'] = datetime.fromtimestamp(
                start_ti + treadpos[-1, 0]).strftime(time_format)

        for key in mdict.keys():
            mdict[key] = np.array(mdict[key])

        # If licking or rewards were configured, ensure that the keys are
        # present even if the mouse never licked, got water, or experienced
        # a reward zone.
        if any(sensor.get('type') == 'lickport' for sensor in settings.get(
                'sensors', {})) and 'licking' not in mdict:
            mdict['licking'] = np.empty((0, 2))

        if 'reward' in settings:
            if 'water' not in mdict:
                mdict['water'] = np.empty((0, 2))
            if 'reward' not in mdict:
                mdict['reward'] = np.empty((0, 2))

        if args.file_type == 'json':
            mdict_converted = numpy_to_list(mdict)
            with open(os.path.splitext(fname)[0] + '.json', 'w') as f:
                json.dump(mdict_converted, f, indent=4)
        else:
            with open(os.path.splitext(fname)[0] + '.pkl', 'wb') as f:
                pickle.dump(mdict, f)

        if args.group is not None:
            trial_info['experiment_group'] = args.group

        if args.trial_id is not None and not args.sql:
            logging.warning("--trial_id has no effect without --sql, skipping database load")

        if args.sql:
            if trial_info['experiment_group'] is None:
                raise Exception('Experiment group is required to load sql')
            loadSql(fname, trial_info, settings,
                    experiment_info=experiment_info, trial_id=args.trial_id)


if __name__ == '__main__':
    main(sys.argv[1:])
