import splitfolders as sf

input_folder = "archive/"
sf.ratio(input_folder, output="../",
         ratio=(.7, .2, .1),
         group_prefix=None)
