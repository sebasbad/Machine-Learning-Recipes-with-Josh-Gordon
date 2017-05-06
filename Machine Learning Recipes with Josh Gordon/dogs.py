import numpy as np
import matplotlib.pyplot as plt

# Create greyhound and labrador populations
greyhounds = 500
labs = 500

# Average heights arrays with random +- 4 values
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(greyhounds)

# Plot average heights histogram, greyhounds in red, labradors in blue
plt.hist([grey_height, lab_height], stacked = True, color = ['r', 'b'])
plt.show()

# Eyes color: useless feature that shouldn't include as a feature to test
# against, as this feature doesn't depend on the breed, so the distribution is
# about 50%

# Also should use independent features, as they give distinct type of
# information, e.g. we shouldn't use height in inches and height in centimeters
# because they are highly corelated; avoid redundant features

# Fearures should be easy to understand: to predict how long it would take to
# send a letter between 2 cities, the distance between them would be a better
# feature than the latitude and longitude of each city, as the former would need
# much more training cases to understand the relation between the latitude, the
# longutude, the distance and the time, to be able to get a similar prediction
# accuracy. Simpler relationships are easier to learn.

# Ideal features: informative, independent, simple.
