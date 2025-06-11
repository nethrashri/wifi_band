#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

int main(int argc, char* argv[]) {
    if (argc != 16) {
        std::cerr << "Usage: " << argv[0] << " <15 float inputs>" << std::endl;
        return 1;
    }

    // Read inputs from command line
    std::vector<float> input(15);
    for (int i = 1; i < argc; ++i) {
        input[i - 1] = std::stof(argv[i]);
    }

    // Load TFLite model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("service_predictor.tflite");
    if (!model) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }

    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter) {
        std::cerr << "Failed to construct interpreter." << std::endl;
        return 1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    // Copy inputs to tensor
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    for (size_t i = 0; i < input.size(); ++i) {
        input_tensor[i] = input[i];
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke TFLite interpreter." << std::endl;
        return 1;
    }

    // Print output
    float* output = interpreter->typed_output_tensor<float>(0);
    std::cout << "Prediction: " << output[0] << std::endl;

    return 0;
}

// g++ inference_check.cpp -o inference_check -ltensorflow-lite
// ./inference_check 1728748800 12 1 3 1 4 1 79.29 19.84 5.0 3 19.2 0.27 0.0 4.38
