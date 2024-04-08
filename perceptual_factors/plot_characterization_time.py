# Read a csv file with pandas named pred_prob.csv
# Headers are ['confidence', 'class', 'depth']
# Plot the confidence depending on the depth

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    pred_prob = pd.read_csv('./nn_characterization/pred_prob_p.csv')

    # Get only the rows to the corresponding class 'bottle' or 'cup'
    selected_class = 'tv'
    # It is stored in 'name' column
    pred_prob = pred_prob[pred_prob['class'] == selected_class]

    # Get the confidence and depth columns
    confidence = pred_prob['confidence']
    depth = pred_prob['time']
    # Filter depth and confidence as only values below 4 meters and above 1 are right
    # confidence = confidence[depth < 4000]
    # confidence = confidence[depth > 1000]
    # depth = depth[depth < 4000]
    # depth = depth[depth > 1000]

    # Select the confidences that are 0
    confidence_0 = confidence[confidence == 0]
    depths_0 = depth[confidence == 0]

    # Select the confidences that are not 0
    confidence_not_0 = confidence[confidence != 0]
    depth_not_0 = depth[confidence != 0]

    # Plot the confidence depending on the depth
    plt.scatter(depth_not_0, confidence_not_0)
    # Plot the confidences that are 0 as red crosses
    plt.scatter(depths_0, confidence_0, marker='x', color='red')

    # Compute a polynomial fit
    z = np.polyfit(depth_not_0, confidence_not_0, 3)
    f = np.poly1d(z)
    # Plot the polynomial fit
    # Get max depth
    # max_depth = 4000
    # min_depth = 1000
    # x_new = np.linspace(min_depth, max_depth, 100)
    # y_new = f(x_new)
    # plt.plot(x_new, y_new, color='red')

    plt.xlabel('Time')
    plt.ylabel('Confidence')
    plt.show()

if __name__ == '__main__':
    main()
