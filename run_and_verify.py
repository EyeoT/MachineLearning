import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

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
    num_train_frames = int(len(frame_sets) * 0.8)
    test_frames = []
    train_frames = []
    remaining_frames = []

    if (num_train_frames >= 5):

        p_cream = [0.166666667, 0.166666667, 0.166666667, 0.166666667, 0.166666667, 0.166666667, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0]
        n = float(1) / 31
        p_none = [0, 0, 0, 0, 0, 0, n, n, n, n, n, 0, 0, 0, 0, n, n, n, 0, n, n, 0, 0, 0, 0, n, n, n, n, n, n, 0, 0, 0,
                  0, n, n, n, n, 0, 0, 0, n, 0, 0, 0, 0, 0, 0, 0, n, n, n, n, n, n, n, n, n, n];

        p_green = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.066666667, 0.066666667, 0.066666667, 0.066666667, 0, 0, 0,
                   0.066666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.066666667, 0.066666667,
                   0.066666667,
                   0, 0.066666667, 0.066666667, 0.066666667, 0.066666667, 0.066666667, 0.066666667, 0.066666667, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0]

        p_blue = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        p_red = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25,
                 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        train_frames.append(np.random.choice(frame_sets, 1, replace=False, p=p_cream)[0])  # cream
        train_frames.append(np.random.choice(frame_sets, 1, replace=False, p=p_none)[0])  # no box found
        train_frames.append(np.random.choice(frame_sets, 1, replace=False, p=p_green)[0])  # green
        train_frames.append(np.random.choice(frame_sets, 1, replace=False, p=p_blue)[0])  # blue
        train_frames.append(np.random.choice(frame_sets, 1, replace=False, p=p_red)[0])  # red

        for frame in frame_sets:
            if frame not in train_frames:
                remaining_frames.append(frame)

        # pick 80% w/o replace
        new_train_frames = np.random.choice(remaining_frames, num_train_frames - 5, replace=False)

        for frame in new_train_frames:
            train_frames.append(frame)

        for frame in frame_sets:
            if frame not in train_frames:
                test_frames.append(frame)  # separate training data

    return test_frames, train_frames


def train_or_test(frames):
    '''
    :param frames: Either the 80% of total frames used to train the algorithm, or the 20% used to test
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

    for frame in frames:
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


def plot_ellipsoid_3d(mean, covariance, ax):
    """Plot 3-d Ellipsoid based on mean and covariance on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = mean + np.dot(covariance,
                                                      [x[i,j],y[i,j],z[i,j]])

    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='#2980b9', alpha=0.2)


def plot_3D(measured_color, classification, true_color, show_correct, gmm):
    '''
    :param measured_color: The BGR-ordered triplet representing the color of the faceplate. [0, 0, 0] if None
    :param classification: The string representing the color guess ('cream', 'blue', 'green', 'red', or 'no box found')
    :param true_color: The string representing the intended color for the frame
    :param show_correct: A string ('incorrect', 'correct', 'all') that toggles between showing all, the correctly, or
    incorrectly classified points.
    :return: The matplotlib plot object
    '''
    color_markers = {
        'cream': 'm',
        'blue': 'b',
        'red': 'r',
        'green': 'g',
        'no box found': 'k',
        'None': 'k'
        }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(0, 255)  # force axis to scale to [0, 255] RBG range
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)
    for bgr, case, truth in zip(measured_color, classification, true_color):
        x, y, z = bgr
        edge_color = color_markers[case]
        face_color = color_markers[truth]
        if (edge_color != face_color) and show_correct == 'incorrect':
            print "({0}, {1}, {2}) case: {3} truth:{4} face_color: {5} edge_color:{6}".format(x, y, z, case, truth,
                                                                                              face_color, edge_color)
            ax.scatter(x, y, z, facecolors=face_color, edgecolors=edge_color, marker='o')
        elif (edge_color == face_color) and show_correct == 'correct':
            ax.scatter(x, y, z, facecolors=face_color, edgecolors=edge_color, marker='o')
        elif show_correct == 'all':
            ax.scatter(x, y, z, facecolors=face_color, edgecolors=edge_color, marker='o')

    for i in range(gmm.n_components):
        plot_ellipsoid_3d(gmm.means_[i], gmm.covariances_[i], ax)

    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')

    plt.show()
    return fig


def color_classification(measured_color, true_color, test_frames):
    none_list = []
    green_list = []
    cream_list = []
    red_list = []
    blue_list = []

    for color in range(0, len(measured_color)):  # separate color BGR values
        if true_color[color] == 'None':
            none_list.append(measured_color[color])
        elif true_color[color] == 'green':
            green_list.append(measured_color[color])
        elif true_color[color] == 'cream':
            cream_list.append(measured_color[color])
        elif true_color[color] == 'red':
            red_list.append(measured_color[color])
        elif true_color[color] == 'blue':
            blue_list.append(measured_color[color])

    none_mean = np.mean(none_list, axis=0)
    green_mean = np.mean(green_list, axis=0)
    cream_mean = np.mean(cream_list, axis=0)
    red_mean = np.mean(red_list, axis=0)
    blue_mean = np.mean(blue_list, axis=0)

    # Fit a Gaussian mixture model with EM using five components for each color
    gmm = mixture.GaussianMixture(n_components=5, covariance_type='spherical', n_init=1).fit(measured_color)

    # Map GMM clusters to their proper labels
    gmm_order = ['unknown', 'unknown', 'unknown', 'unknown', 'unknown']  # initialize empty ordering
    check_in = False  # initialize to false
    for i in range(0, 5):
        gmm_mean = gmm.means_[0:5][i][0:3]
        none_mean_diff = np.sum(abs(gmm_mean - none_mean[0:3]))
        green_mean_diff = np.sum(abs(gmm_mean - green_mean[0:3]))
        cream_mean_diff = np.sum(abs(gmm_mean - cream_mean[0:3]))
        red_mean_diff = np.sum(abs(gmm_mean - red_mean[0:3]))
        blue_mean_diff = np.sum(abs(gmm_mean - blue_mean[0:3]))

        if none_mean_diff < 1:
            gmm_order[i] = 'None'
        elif green_mean_diff < 1:
            gmm_order[i] = 'green'
        elif cream_mean_diff < 1:
            gmm_order[i] = 'cream'
        elif red_mean_diff < 1:
            gmm_order[i] = 'red'
        elif blue_mean_diff < 1:
            gmm_order[i] = 'blue'
        else:
            print("gmm_mean diff too great: {0}!", gmm_mean)
            check_in = True

    if check_in and ('green' not in gmm_order) and ('unknown' in gmm_order):
        gmm_order[gmm_order.index('unknown')] = 'green'  # overwrite the value with green
    else:
        print("ERROR: Unable to map labels to GMM!")

    # TODO: Run get_lightbox_color on test_frames, put output into prediction function below:
    test_measured_colors, test_classification, test_true_color, test_correct, test_position = train_or_test(test_frames)
    prediction = gmm.predict(test_measured_colors)
    post_probs = gmm.predict_proba(test_measured_colors)
    num_correct = 0
    for i in range(0, len(prediction)):
        if gmm_order[int(prediction[i])] == test_true_color[i]:
            num_correct += 1
        print("Predicted color: {0} | True color: {1} for image {2}\n".format(gmm_order[int(prediction[i])],
                                                                              test_true_color[i],
                                                                              test_frames[i]['frame']))
    print(
    "FINAL RESULT: {0} of {1} frames correctly classified! ({2}%)".format(num_correct, len(prediction), "%0.2f" % (100 *
                                                                                                                   float(
                                                                                                                       num_correct) / len(
        prediction))))
    return gmm  # temporary

if __name__ == '__main__':
    test_frames, train_frames = separate_train_and_test()
    measured_color, classification, true_color, correct, position = train_or_test(train_frames)
    write_data(measured_color, classification, true_color, correct, position)
    gmm = color_classification(measured_color, true_color, test_frames)
    plot_3D(measured_color, classification, true_color, 'all', gmm)
