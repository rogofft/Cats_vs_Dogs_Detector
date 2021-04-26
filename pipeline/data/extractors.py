import os


def get_data_extractor(config: str):
    if config == 'default':
        return extract_data_default
    else:
        raise BaseException('Data extractor not found!')


# User's data extractors


def extract_data_default(data_path_list) -> list:
    data = []
    for data_path in data_path_list:
        with open(os.path.splitext(data_path)[0] + '.txt') as f:
            # get path to jpg
            img_path = data_path
            metadata = list(map(int, f.readline().strip().split(' ')))
            # get class id
            class_id = metadata[0]
            # process coords: x1, y1, x2, y2 -> xc, yc, w, h
            x, y = (metadata[3] + metadata[1]) // 2, (metadata[4] + metadata[2]) // 2
            w, h = metadata[3] - metadata[1], metadata[4] - metadata[2]

            data.append((img_path, x, y, w, h, class_id))
    return data
