I like mediapipe hands, but I want to make it completely run in pytorch. so that I can optimzie it further. 

first, make a simple python script that takes in an image, and runs mediapipe hands on it. make sure to use the latest medipapie hands model (whcih ends with .task). I will give you an image to run it with.

then, remove the need for the mediappie api. it should all run with just pytorch and some extracted model weights. you figure out where this is an how to do it. verify that your script works with the same image. it should give the exact same result.


Other ideas:
custom kernels / ops for faster pytorch mps or coreml speedup. 
other quantization methods, willing to trade off some accuracy for lower latency.
