# Read a csv file with pandas named pred_prob.csv
# Headers are ['confidence', 'class', 'depth']
# Plot the confidence depending on the depth

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    pred_prob = pd.read_csv('pred_prob.csv')

    # Get only the rows to the corresponding class 'bottle' or 'cup'
    selected_class = 'cup'
    # It is stored in 'name' column
    pred_prob = pred_prob[pred_prob['class'] == selected_class]

    # Get the confidence and depth columns
    confidence = pred_prob['confidence']
    depth = pred_prob['depth']
    # Filter depth and confidence as only values below 4 meters and above 1 are right
    confidence = confidence[depth < 4000]
    confidence = confidence[depth > 1000]
    depth = depth[depth < 4000]
    depth = depth[depth > 1000]
    # Plot the confidence depending on the depth
    plt.scatter(depth, confidence)

    # Compute a polynomial fit
    z = np.polyfit(depth, confidence, 3)
    f = np.poly1d(z)
    # Plot the polynomial fit
    # Get max depth
    max_depth = 4000
    min_depth = 1000
    x_new = np.linspace(min_depth, max_depth, 100)
    y_new = f(x_new)
    plt.plot(x_new, y_new, color='red')

    plt.xlabel('Depth')
    plt.ylabel('Confidence')
    plt.show()

if __name__ == '__main__':
    main()
