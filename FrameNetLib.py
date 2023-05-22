from string import whitespace


class NoManualAnnotation(Exception):
    pass


class FeeFeCollision(Exception):
    pass


FEE_TYPE_DICT = {
    'Core': 'core',
    'Peripheral': 'non-core',
    'Extra-Thematic': 'extra-thematic'
}


def extract_FEs_for_frame(frame):
    result = {
        'core': [],
        'non-core': [],
        'extra-thematic': []
    }
    for child in frame:
        if child.tag.endswith('FE') and child.attrib['coreType'] in FEE_TYPE_DICT:
            fe_type = FEE_TYPE_DICT[child.attrib['coreType']]
            result[fe_type].append(child.attrib['name'])
    return {'frame': frame.attrib['name'], 'frame_elements': result}


def extract_FEs(sentence):
    '''
    Extracts FEE and frame-element annotations from a sentence.
    '''

    # Extract text
    for child in sentence:
        if child.tag.endswith('text'):
            text = child.text
            break
    else:
        raise ValueError('Found no text in the sentence.')

    # Find the manual annotation
    for child in sentence:
        if child.tag.endswith('annotationSet') and child.attrib['status'] == 'MANUAL':
            annotation = child
            break
    else:
        raise NoManualAnnotation(
            'Found no manual annotation for the sentence.')

    # Extract the FEE
    for child in annotation:
        if child.tag.endswith('layer') and child.attrib['name'] == 'Target':
            try:
                label = child[0]
            except IndexError:
                continue
            fee_boundaries = (
                int(label.attrib['start']),
                int(label.attrib['end']))
            break
    else:
        raise ValueError('Found no FEE for the sentence.')

    # Extract the FEs
    frame_elements = []
    for child in annotation:
        if child.tag.endswith('layer') and child.attrib['name'] == 'FE':
            for label in child:
                try:
                    # Check if a FE has the same extent as the FEE.
                    # Cf. bark.v: it is both a FEE and a core FE
                    # "Sound". We filter out such cases.
                    if int(label.attrib['start']) == fee_boundaries[0]:
                        raise FeeFeCollision
                    frame_elements.append({
                        'type': label.attrib['name'],
                        'start': int(label.attrib['start']),
                        'end': int(label.attrib['end'])
                    })
                # Some frame elements may be left unexpressed
                except KeyError:
                    continue
    frame_elements.sort(key=lambda fe: fe['start'])
    return {
        'text': text,
        'FEE_boundaries': fee_boundaries,
        'FEs': frame_elements
    }


def get_tokens_with_ranges(text):
    tokens = []
    ranges = []
    inside_token = False
    buffer = []
    current_start = 0
    for i, char in enumerate(text):
        if char in whitespace:
            if inside_token:
                tokens.append(''.join(buffer))
                buffer.clear()
                ranges.append((current_start, i-1))
                inside_token = False
            continue
        if not inside_token:
            current_start = i
            inside_token = True
        buffer.append(char)
    if buffer:
        tokens.append(''.join(buffer))
        ranges.append((current_start, len(text)))
    return tokens, ranges


def prepend_class(label, fe_dict):
    if label.startswith('FRAME:'):
        return label
    elif label in fe_dict['core']:
        return f'CORE:{label}'
    elif label in fe_dict['non-core']:
        return f'NONCORE:{label}'
    elif label in fe_dict['extra-thematic']:
        return f'EXTRATHEMATIC:{label}'
    else:
        return None  # "Core-Unexpressed"


def prepare_sentence(sentence_dict, frame, fe_dict):
    '''
    Converts a dict with text, FEE boundaries, and FE types and boundaries
    into a pair of aligned lists with tokens on one hand and the name
    of the frame and class (core vs. non-core) and types of FEs on the
    other.
    '''
    text = sentence_dict['text']
    range_to_label = {}
    range_to_label[sentence_dict['FEE_boundaries']] = f'FRAME:{frame}'
    for fe in sentence_dict['FEs']:
        range_to_label[(fe['start'], fe['end'])] = fe['type']
    tokens, ranges = get_tokens_with_ranges(text)
    labels = []
    for range in ranges:
        # We hope to not see overlapping ranges
        # and only check for the start of the
        # interval.
        lo, _ = range
        for (r_lo, r_hi), label in range_to_label.items():
            if lo >= r_lo and lo < r_hi:
                labels.append(prepend_class(label, fe_dict))
                break
        else:
            labels.append(None)
    return tokens, labels
