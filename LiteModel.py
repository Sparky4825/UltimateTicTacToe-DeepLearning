import numpy as np


class LiteModel:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.pi_index = output_details[0]["index"]
        self.v_index = output_details[1]["index"]

        self.input_shape = input_details[0]["shape"]
        self.pi_shape = output_details[0]["shape"]
        self.v_shape = output_details[1]["shape"]

        self.input_dtype = input_details[0]["dtype"]
        self.pi_dtype = output_details[0]["dtype"]
        self.v_dtype = output_details[1]["dtype"]

    def predict(self, board):
        self.interpreter.set_tensor(
            self.input_index, np.array([board], dtype=self.input_dtype)
        )
        self.interpreter.invoke()

        pi = self.interpreter.get_tensor(self.pi_index)
        v = self.interpreter.get_tensor(self.v_index)

        return pi[0], v[0]
