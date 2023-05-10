from package.features import *
import matplotlib.pyplot as plt
import numpy as np
import os

image = os.path.normpath(input("Enter path: "))
print(f"Letter slant: {get_letter_slant(image)}")
print(f"Line slant: {get_line_slant(image)[0]}")
print(f"Letter size: {get_letter_size(image)[0]}")
print(f"Word spacing: {gap_between_words(image)[0]}")

new_img = np.vstack([get_line_slant(image)[1], 
                     get_letter_size(image)[1],
                     gap_between_words(image)[1]])
plt.imshow(new_img)
plt.show()
