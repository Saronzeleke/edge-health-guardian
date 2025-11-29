# models/conversion/quantize_models.py
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os
from pathlib import Path
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelQuantizer")


class ModelQuantizer:
    def __init__(self, model_dir: str = "models/trained_models"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path("models/optimized_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_representative_dataset(self, input_shape: tuple, num_samples: int = 100):
        """
        Returns a function that yields representative inputs for quantization.
        input_shape should be shape WITHOUT batch dimension: e.g. (96,96,3) or (50,12)
        """
        def _representative():
            for _ in range(num_samples):
                if len(input_shape) == 3:
                    arr = np.random.randint(0, 255, size=(1, *input_shape)).astype(np.float32)
                else:
                    arr = np.random.randn(1, *input_shape).astype(np.float32)
                yield [arr]
        return _representative

    def quantize_face_model(self):
        model_path = self.model_dir / "face_analyzer.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found")

        model = tf.keras.models.load_model(str(model_path))
        # Use post-training quantization (or QAT if you have that)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        rep_fn = self.create_representative_dataset((96, 96, 3), num_samples=200)
        converter.representative_dataset = rep_fn
        # For stability, keep inference types as float unless you are sure your runtime expects uint8/int8
        # Do NOT force uint8 unless your input pipeline uses uint8
        # If you truly need full-int8:
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8

        try:
            tflite_model = converter.convert()
        except Exception as e:
            logger.error(f"Face model quantization failed: {e}")
            raise

        out = self.output_dir / "face_analyzer_quant.tflite"
        with open(out, "wb") as fh:
            fh.write(tflite_model)
        logger.info(f"Quantized face model saved: {out}")
        return out

    def quantize_movement_model(self):
        model_path = self.model_dir / "movement_analyzer.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found")

        model = tf.keras.models.load_model(str(model_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        rep_fn = self.create_representative_dataset((50, 12), num_samples=200)
        converter.representative_dataset = rep_fn

        try:
            tflite_model = converter.convert()
        except Exception as e:
            logger.error(f"Movement model quantization failed: {e}")
            raise

        out = self.output_dir / "movement_analyzer_quant.tflite"
        with open(out, "wb") as fh:
            fh.write(tflite_model)
        logger.info(f"Quantized movement model saved: {out}")
        return out

    def quantize_fusion_model(self):
        model_path = self.model_dir / "fusion_engine.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found")

        model = tf.keras.models.load_model(str(model_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # fusion likely expects shape (64,) -> pass (64,)
        rep_fn = self.create_representative_dataset((64,), num_samples=200)
        converter.representative_dataset = rep_fn

        try:
            tflite_model = converter.convert()
        except Exception as e:
            logger.error(f"Fusion model quantization failed: {e}")
            raise

        out = self.output_dir / "fusion_engine_quant.tflite"
        with open(out, "wb") as fh:
            fh.write(tflite_model)
        logger.info(f"Quantized fusion model saved: {out}")
        return out

    def benchmark_quantized_models(self):
        logger.info("Benchmarking quantized models in output folder")
        results = {}
        for f in self.output_dir.glob("*.tflite"):
            try:
                interpreter = tf.lite.Interpreter(model_path=str(f))
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                if not input_details:
                    logger.warning(f"No input details for {f.name}; skipping")
                    continue
                shape = tuple(input_details[0]["shape"])
                if any(int(x) <= 0 for x in shape):
                    # fallback to sensible shape
                    shape = input_details[0].get("shape_signature", shape)

                dtype = np.dtype(input_details[0]["dtype"].name) if hasattr(input_details[0]["dtype"], "name") else np.float32
                if np.issubdtype(dtype, np.integer):
                    input_data = np.random.randint(0, 255, size=shape).astype(dtype)
                else:
                    input_data = np.random.randn(*shape).astype(dtype)

                # warmup and timing
                times = []
                for i in range(30):
                    t0 = time.perf_counter()
                    interpreter.set_tensor(input_details[0]["index"], input_data)
                    interpreter.invoke()
                    t1 = time.perf_counter()
                    if i >= 5:  # skip first 5 as warmup
                        times.append((t1 - t0) * 1000.0)

                arr = np.array(times)
                results[f.name] = {
                    "avg_ms": float(arr.mean()),
                    "std_ms": float(arr.std()),
                    "model_size_mb": float(f.stat().st_size) / (1024 * 1024)
                }
                logger.info(f"Benchmarked {f.name}: avg {results[f.name]['avg_ms']:.2f} ms")
            except Exception as e:
                logger.warning(f"Benchmark failed for {f.name}: {e}")
        out_path = self.output_dir / "quant_benchmarks.json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        return results


def main():
    q = ModelQuantizer()
    try:
        q.quantize_face_model()
        q.quantize_movement_model()
        q.quantize_fusion_model()
        q.benchmark_quantized_models()
        logger.info("Quantization + benchmarking complete")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")


if __name__ == "__main__":
    main()
