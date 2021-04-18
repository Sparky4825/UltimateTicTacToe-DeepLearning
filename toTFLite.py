import tensorflow as tf
from UltimateTicTacToe import UltimateTicTacToeGame as UTicTacToe
from UltimateTicTacToe.keras.NNet import NNetWrapper

convert_new = False

if convert_new:
    converter = tf.lite.TFLiteConverter

    g = UTicTacToe.TicTacToeGame()
    nnet = NNetWrapper(g)
    nnet.load_checkpoint("./temp/", "best.ckpt")

    tflite_model = nnet.convert_to_tflite()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Print input type and shape
print("Input details")
for i in interpreter.get_input_details():
    print(f"{i['shape']} {i['dtype']}")
    print(i)

print("Output details")
for i in interpreter.get_output_details():
    print(f"{i['shape']} {i['dtype']}")
