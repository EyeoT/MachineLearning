import os
import csv
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


if __name__ == '__main__':
    folder_path = 'test_data'
    frame_sets = read_data(folder_path)
    num_accurate = 0
    for frame_set in frame_sets:
        img = cv2.imread(os.path.join(folder_path, frame_set['frame']))
        gaze_center_x, gaze_center_y = frame_set['gaze_data']
        gaze_center_x = int(img.shape[1] * gaze_center_x)
        gaze_center_y = int(img.shape[0] * (1-gaze_center_y))
        cv2.circle(img, (gaze_center_x, gaze_center_y), 2, (255, 0, 255), -1)
        color = get_lightbox_color.get_box_color(img, frame_set['gaze_data'])
        if color == frame_set['true_color']:
            num_accurate += 1
    print num_accurate
