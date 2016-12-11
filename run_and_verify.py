import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import get_lightbox_color

# TODO: 3D scatter color plot (Gen's matplotlib + Harrison?)

folder_path = 'test_data'  # constant


def read_data():
    csv_file_name = 'gaze_frame_data_human.csv'
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


def write_data(measured_color, classification, true_color, correct, position):
    '''
    This function writes the results of the train_frames classification to a .csv file for offline analysis
    :param measured_color: The BGR-ordered triplet representing the color of the faceplate. [0, 0, 0] if None
    :param classification: The string representing the color guess ('cream', 'blue', 'green', 'red', or 'no box found')
    :param true_color: The string representing the intended color for the frame
    :param correct: Binary value 1 if the box was correctly looked at and classified by color.
    :param position: Which of the 10 locations stood at when taking data, used to catch patterns. Location 9, for
    instance, tends to fail more than others due to the intense angle
    :return: Returns true if csv write executed properly
    '''

    csv_file_name = 'response.csv'
    csv_file = open(os.path.join(folder_path, csv_file_name), 'wb')
    fieldnames = ['color_classification', 'correct', 'position', 'hex', 'B', 'G', 'R', 'true_color']
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
                         'hex': hex_color[i], 'true_color': true_color[i]})
    return True


def separate_train_and_test():
    '''
    This function reads in all the frames, randomly separating 10% of the frames for testing and the other 90% for
    training.
    :return: Returns the separated testing and training frames
    '''

    frame_sets = read_data()
    test_frames = np.random.choice(frame_sets, int(len(frame_sets) * 0.1), replace=False)  # pick 10% w/o replacement
    train_logic = (frame for frame in frame_sets if frame not in test_frames)
    train_frames = []

    for frame in train_logic:
        train_frames.append(frame)  # separate training data

    return test_frames, train_frames


def train(train_frames):
    '''
    :param train_frames: The 90% of total frames used to train the algorithm
    :return: Returns the BGR-ordered triplet representing the color of the faceplate, the string representing the color
    ('cream', 'blue', 'green', 'red', or 'no box found'), binary value 1 if the box was correctly fixated on and
    classified by color, and which of the 10 locations stood at when taking data, used to catch patterns.
    '''

    measured_color = []
    classification = []
    correct = []
    position = []
    true_color = []
    num_accurate = 0  # running count of successful classifications
    iterator = 0  # running count of total frames

    for frame in train_frames:
        img = cv2.imread(os.path.join(folder_path, frame['frame']))
        gaze_center_x = int(img.shape[1] * frame['gaze_data'][0])  # user's gaze x-coordinate
        gaze_center_y = int(img.shape[0] * (1 - frame['gaze_data'][1]))  # user's gaze y-coordinate
        cv2.circle(img, (gaze_center_x, gaze_center_y), 2, (255, 0, 255), -1)  # plots the gaze point for debugging
        mc, c = get_lightbox_color.get_box_color(img, frame['gaze_data'])
        measured_color.append(mc)
        classification.append(c)
        if classification[iterator] == frame['true_color']:
            num_accurate += 1
            correct.append(1)
        else:
            correct.append(0)
        print("True color: {0} for image {1}\n".format(frame['true_color'], frame['frame']))
        position.append(iterator % 10)  # positions repeated every 10 frames
        true_color.append(frame['true_color'])
        iterator += 1
    print("Total number accurately classified: {0} of {1} or {2}%\n".format(num_accurate, iterator, "%0.2f" % (100 *
            float(num_accurate) / iterator)))
    return measured_color, classification, true_color, correct, position


def plot_3D(measured_color):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in measured_color:
        ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    plt.show()


if __name__ == '__main__':
    test_frames, train_frames = separate_train_and_test()
    measured_color, classification, true_color, correct, position = train(train_frames)
    write_data(measured_color, classification, true_color, correct, position)
    plot_3D(measured_color)
