from package.features import *
import matplotlib.pyplot as plt
import os

image = os.path.normpath(input("Enter path: "))
print(f"Letter slant: {get_letter_slant(image)}")
print(f"Line slant: {get_line_slant(image)[0]}")

plt.imshow(get_line_slant(image)[1])
plt.show()
