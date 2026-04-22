import os
import torch
import json
from config.classes import EVENT_DICTIONARY


def _parse_contact(val):
    """'With contact' -> 1, anything else -> 0"""
    return 1 if val == "With contact" else 0


def _parse_bodypart(val):
    """'Upper body' -> 0, 'Under body' -> 1, unknown -> 0"""
    if val == "Under body":
        return 1
    return 0


def _parse_try_to_play(val):
    """'Yes' -> 1, anything else -> 0"""
    return 1 if val == "Yes" else 0


def _parse_handball(val):
    """'handball' or 'handball attempt' -> 1, 'No handball' / '' -> 0"""
    if isinstance(val, str) and val.lower() in ("handball", "handball attempt"):
        return 1
    return 0


def label2vectormerge(folder_path, split, num_views):
    path_annotations = os.path.join(folder_path, split, "annotations.json")

    dictionary_action = EVENT_DICTIONARY['action_class']

    if os.path.exists(path_annotations):
        with open(path_annotations) as f:
            train_annotations_data = json.load(f)
    else:
        print("PATH DOES NOT EXISTS")
        exit()

    not_taking = []

    num_classes_action = 8
    num_classes_offence_severity = 4

    labels_action = []
    labels_offence_severity = []
    # Auxiliary binary labels (scalars stored as Python ints)
    labels_contact = []
    labels_bodypart = []
    labels_try_to_play = []
    labels_handball = []

    number_of_actions = []

    total_distribution = torch.zeros(num_classes_offence_severity, num_classes_action)
    distribution_action = torch.zeros(1, num_classes_action)
    distribution_offence_severity = torch.zeros(1, num_classes_offence_severity)

    for actions in train_annotations_data['Actions']:
        action_data = train_annotations_data['Actions'][actions]
        action_class = action_data.get('Action class', '')
        offence_class = action_data.get('Offence', '')
        severity_class = action_data.get('Severity', '')

        # --- auxiliary fields (graceful fallback to 0) ---
        contact_val = _parse_contact(action_data.get('Contact', ''))
        bodypart_val = _parse_bodypart(action_data.get('Bodypart', ''))
        try_to_play_val = _parse_try_to_play(action_data.get('Try to play', ''))
        handball_val = _parse_handball(action_data.get('Handball', ''))

        # --- same filtering as original ---
        if action_class == '' or action_class == 'Dont know':
            not_taking.append(actions)
            continue

        if (offence_class == '' or offence_class == 'Between') and action_class != 'Dive':
            not_taking.append(actions)
            continue

        if (severity_class == '' or severity_class == '2.0' or severity_class == '4.0') and \
                action_class != 'Dive' and \
                offence_class not in ('No offence', 'No Offence'):
            not_taking.append(actions)
            continue

        if offence_class == '' or offence_class == 'Between':
            offence_class = 'Offence'

        if severity_class == '' or severity_class == '2.0' or severity_class == '4.0':
            severity_class = '1.0'

        # --- determine severity index ---
        if offence_class in ('No Offence', 'No offence'):
            off_index = 0
        elif offence_class == 'Offence' and severity_class == '1.0':
            off_index = 1
        elif offence_class == 'Offence' and severity_class == '3.0':
            off_index = 2
        elif offence_class == 'Offence' and severity_class == '5.0':
            off_index = 3
        else:
            not_taking.append(actions)
            continue

        if num_views == 1:
            # One entry per clip
            for i in range(len(action_data['Clips'])):
                sev_vec = torch.zeros(1, num_classes_offence_severity)
                sev_vec[0][off_index] = 1
                labels_offence_severity.append(sev_vec)
                distribution_offence_severity[0][off_index] += 1

                act_vec = torch.zeros(1, num_classes_action)
                act_vec[0][dictionary_action[action_class]] = 1
                labels_action.append(act_vec)
                distribution_action[0][dictionary_action[action_class]] += 1
                total_distribution[off_index][dictionary_action[action_class]] += 1

                labels_contact.append(contact_val)
                labels_bodypart.append(bodypart_val)
                labels_try_to_play.append(try_to_play_val)
                labels_handball.append(handball_val)
        else:
            sev_vec = torch.zeros(1, num_classes_offence_severity)
            sev_vec[0][off_index] = 1
            labels_offence_severity.append(sev_vec)
            distribution_offence_severity[0][off_index] += 1

            act_vec = torch.zeros(1, num_classes_action)
            act_vec[0][dictionary_action[action_class]] = 1
            labels_action.append(act_vec)
            distribution_action[0][dictionary_action[action_class]] += 1
            total_distribution[off_index][dictionary_action[action_class]] += 1

            labels_contact.append(contact_val)
            labels_bodypart.append(bodypart_val)
            labels_try_to_play.append(try_to_play_val)
            labels_handball.append(handball_val)

            number_of_actions.append(actions)

    return (
        labels_offence_severity,
        labels_action,
        labels_contact,
        labels_bodypart,
        labels_try_to_play,
        labels_handball,
        distribution_offence_severity[0],
        distribution_action[0],
        not_taking,
        number_of_actions,
    )


def clips2vectormerge(folder_path, split, num_views, not_taking):
    path_clips = os.path.join(folder_path, split)

    if not os.path.exists(path_clips):
        return []

    folders = sum(len(d) for _, d, _ in os.walk(path_clips))
    clips = []

    for i in range(folders):
        if str(i) in not_taking:
            continue

        path_clip = os.path.join(path_clips, "action_" + str(i))

        if num_views == 1:
            for clip_name in ("clip_0.mp4", "clip_1.mp4", "clip_2.mp4", "clip_3.mp4"):
                cp = os.path.join(path_clip, clip_name)
                if clip_name in ("clip_0.mp4", "clip_1.mp4") or os.path.exists(cp):
                    clips.append([cp])
        else:
            clips_all_view = [os.path.join(path_clip, "clip_0.mp4"),
                              os.path.join(path_clip, "clip_1.mp4")]
            for clip_name in ("clip_2.mp4", "clip_3.mp4"):
                cp = os.path.join(path_clip, clip_name)
                if os.path.exists(cp):
                    clips_all_view.append(cp)
            clips.append(clips_all_view)

    return clips
