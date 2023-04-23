from package.features import get_letter_slant, get_line_slant, auto_crop_image
import matplotlib.pyplot as plt

image = r"Personality_Prediction\dataset\data1\training_set\Agreeableness\IMG_20200215_164151.jpg"
print(f"Letter slant: {get_letter_slant(image)}")
print(f"Line slant: {get_line_slant(image)[0]}")



plt.imshow(get_line_slant(image)[1])
plt.show()