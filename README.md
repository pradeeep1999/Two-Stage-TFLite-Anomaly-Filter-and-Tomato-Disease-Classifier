


# Two-Stage Intelligent Tomato Leaf Disease Classification Pipeline

An end-to-end, false-positive resistant Edge-AI pipeline designed to classify tomato leaf diseases. By implementing a custom **Two-Stage Validation and Inference Architecture**, this pipeline actively filters out non-leaf objects, user interfaces, app icons, and random background noise *before* running disease diagnostics, ensuring high-reliability deployments on mobile devices.

---

## 🌟 Key Features

* **False-Positive Mitigation:** Implements an advanced "Leaf Destruction Generator" in Stage 1 to learn fine organic structures, effectively rejecting digital UI, logos, and synthetic noise without requiring manual collection of real-world negative samples.
* **Optimized for Edge Devices:** Stage 2 outputs fully quantized **INT8 TFLite** models optimized for fast, resource-constrained mobile execution.
* **Alphabetical Calibration:** Fully mapped class configurations to resolve standard Keras folder-to-index shifts natively.

---

## 📐 Architecture Overview

The system processes input images through a strict sequential gate network:

```
                  [ Input Image ]
                         │
                         ▼
             ┌───────────────────────┐
             │   STAGE 1 FILTER      │  (stage1_leaf_filter.tflite)
             │   Is it a Real Leaf?  │
             └───────────┬───────────┘
                         │
                ┌────────┴────────┐
         Score < 0.60      Score >= 0.60
                │                 │
                ▼                 ▼
     [ Please provide ]   [ STAGE 2 CLASSIFIER ] (tomato_model_quantized.tflite)
     [ proper image   ]           │
                                  ├─► Tomato_healthy
                                  ├─► Tomato__Tomato_mosaic_virus
                                  └─► Tomato_Tomato_YellowLeaf_Curl_Virus

```

1. **Stage 1 (Binary Structural Filter):** A lightweight convolutional network trained using extreme channel shifts, color inversions, and digital masks to output a structural confidence score between `0.0` (Noise/Logo) and `1.0` (Organic Leaf).
2. **Stage 2 (Disease Classifier):** A multi-class CNN that extracts features to determine specific leaf pathologies, executing *only* if Stage 1 crosses the established validation threshold.

---

## 📂 Dataset Requirements

For training Stage 2, structure your extracted image directory as follows:

```text
/content/extracted_files/
  ├── Tomato_healthy/
  ├── Tomato_Tomato_YellowLeaf_Curl_Virus/
  └── Tomato__Tomato_mosaic_virus/

```

*Note: Keras automatically indexes these subdirectories alphabetically during compilation.*

---

## 🚀 Getting Started

### 1. Training Stage 1 (With Advanced Structural Negatives)

Run the training script to synthesize color-inverted and masked negatives directly out of your target training images. This establishes rigorous bounds for what constitutes an authentic leaf structure:

```python
# Train the model using the synthetic hard-negative loop
model_1.fit(m1_train_ds, steps_per_epoch=25, epochs=20)
# Export directly to TFLite for deployment
model_1.save('stage1_leaf_filter.keras')

```

### 2. Training Stage 2 & Exporting Quantized INT8 Models

Run your disease classification script to compile your baseline model. The script leverages a representative dataset generator to safely compress network weights into mobile-ready `INT8` layouts:

```python
quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

```

### 3. Pipeline Inference Implementation

Deploy the twin-engine TFLite execution gate using OpenCV and TensorFlow Lite interpreters on your edge system:

```python
# Execute Stage 1 validation gate
interpreter1.invoke()
leaf_score = interpreter1.get_tensor(i1_output_details[0]['index'])[0][0]

if leaf_score < 0.60:
    print("Result: Please provide proper image")
else:
    # Route input execution to the localized Stage 2 disease model
    interpreter2.invoke()

```

---

## 📊 Target Classes & Sorted Mapping

To bypass index mismatches common in standard deployment code, the Stage 2 model maps directly to the following alphabetical index positions:

| Target Class Name | Index | Description |
| --- | --- | --- |
| `Tomato_Tomato_YellowLeaf_Curl_Virus` | **0** | Characterized by severe curling and yellow leaf margins. |
| `Tomato__Tomato_mosaic_virus` | **1** | Identified by mottled, dark green/light yellow mosaic spots. |
| `Tomato_healthy` | **2** | Clean, normal organic leaf structure. |

---

## 🛠️ Performance & Tuning

* **Adjusting Thresholds:** If real leaves are accidentally rejected in high-glare field environments, lower your `LEAF_THRESHOLD` in the inference script to `0.50`. If complex green wallpapers continue to pass through, raise the threshold up to `0.75`.
* **Hardware Acceleration:** Ensure your deployment platform targets the Android `NNAPI` or iOS `Metal` delegate for fast, low-latency execution of the quantized `tomato_model_quantized.tflite` model.
