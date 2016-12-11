import csv
import os

import cv2

import get_lightbox_color


def read_data(folder_path):
    csv_file_name = 'gaze_frame_data.csv'
    csv_file = open(os.path.join(folder_path, csv_file_name), 'r')
    reader = csv.DictReader(csv_file)
    frame_sets = []
    for row in reader:
        frame_set = {}
        frame_set['frame'] = row['frame_file']
        frame_set['gaze_data'] = [
            float(row['x_norm_pos']), float(row['y_norm_pos'])]
        frame_set['true_color'] = row['true_color']
        frame_set['true_direction'] = row['true_direction']
        frame_set['confidence'] = row['confidence']
        frame_sets.append(frame_set)
    return frame_sets


def write_data(folder_path, measured_color, classification, correct, position):
    csv_file_name = 'response.csv'
    csv_file = open(os.path.join(folder_path, csv_file_name), 'wb')
    fieldnames = ['color_classification', 'correct', 'position', 'hex', 'B', 'G', 'R']
    hex_color = []
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0, len(correct)):

        if classification is not 'no box found':
            hc = '#%02x%02x%02x' % (measured_color[i][2], measured_color[i][1], measured_color[i][0])
            hex_color.append(hc)
        else:
            hex_color.append('#000000')

        writer.writerow({'color_classification': classification[i], 'correct': correct[i], 'position': position[i],
                         'B': measured_color[i][0], 'G': measured_color[i][1], 'R': measured_color[i][2],
                         'hex': hex_color[i]})
    return True


if __name__ == '__main__':
    # pos = {0: 'blue', 1: 'green', 2: 'red', 3: 'cream'}
    folder_path = 'test_data'
    frame_sets = read_data(folder_path)
    measured_color = []
    classification = []
    correct = []
    position = []
    num_accurate = 0
    iterator = 0
    for frame_set in frame_sets:
        img = cv2.imread(os.path.join(folder_path, frame_set['frame']))
        gaze_center_x, gaze_center_y = frame_set['gaze_data']
        gaze_center_x = int(img.shape[1] * gaze_center_x)
        gaze_center_y = int(img.shape[0] * (1-gaze_center_y))
        cv2.circle(img, (gaze_center_x, gaze_center_y), 2, (255, 0, 255), -1)
        mc, c = get_lightbox_color.get_box_color(img, frame_set['gaze_data'])
        measured_color.append(mc)
        classification.append(c)
        if classification[iterator] == frame_set['true_color']:
            num_accurate += 1
            correct.append(1)
        else:
            correct.append(0)
        print("True color: {0} for image # {1}\n".format(frame_set['true_color'], iterator))
        position.append(iterator % 10)
        iterator += 1
    write_data(folder_path, measured_color, classification, correct, position)
    print num_accurate
