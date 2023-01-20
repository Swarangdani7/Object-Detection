def find_focal_length(known_distance, known_width, width_in_frame):
    focal_length = (width_in_frame * known_distance) / known_width
    return focal_length

def find_distance(focal_length, known_width, width_in_frame):
    d = (int)((known_width * focal_length) / width_in_frame)
    return d
    