from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load("C:\my_personal_code\TAPCV\model.onnx")
tf_rep = prepare(onnx_model)  # Convert ONNX to TensorFlow
tf_rep.export_graph("saved_model")