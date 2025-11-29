import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path
import logging
import time
from typing import Callable, Generator, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TFLiteConverter")


def _safe_get_model_size(model_path: Path) -> int:
    try:
        return os.path.getsize(model_path)
    except Exception:
        return 0


class TFLiteConverter:
    def __init__(self, model_dir: str = "models/trained_models"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path("models/optimized_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TFLiteConverter initialized: model_dir={self.model_dir} output_dir={self.output_dir}")

    def convert_keras_to_tflite(
        self,
        model_name: str,
        optimization_level: str = "DEFAULT",
        representative_dataset_fn: Optional[Callable[[], Generator]] = None
    ) -> Tuple[Path, dict]:
        """
        Convert a Keras model to TFLite.

        Args:
            model_name: base file name (without extension) in model_dir
            optimization_level: "DEFAULT", "FP16", "INT8", "FULL_INT8"
            representative_dataset_fn: optional callable returning a generator for representative data
        Returns:
            (output_path, model_info)
        """
        model_path = self.model_dir / f"{model_name}.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Converting {model_name} with optimization {optimization_level}")

        model = tf.keras.models.load_model(str(model_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Base optimization
        if optimization_level in ("DEFAULT", "FP16", "INT8", "FULL_INT8"):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # FP16
        if optimization_level == "FP16":
            converter.target_spec.supported_types = [tf.float16]

        # INT8 calibration (partial) - requires representative dataset matching model input
        if optimization_level in ("INT8", "FULL_INT8"):
            if representative_dataset_fn is None:
                # create a small default representative dataset if not provided
                representative_dataset_fn = self._create_default_representative_dataset(model)
            converter.representative_dataset = representative_dataset_fn

        # FULL_INT8: require full integer quantization - be conservative and set types only if representative provided
        if optimization_level == "FULL_INT8":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Do not force inference types blindly; inspect input dtype
            input_dtype = model.inputs[0].dtype
            # If the model was trained with uint8 input pipeline, you may set uint8; else use int8
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        # Avoid unstable private flags; rely on stable APIs
        try:
            tflite_model = converter.convert()
        except Exception as e:
            logger.error(f"Conversion failed for {model_name}: {e}")
            raise

        output_path = self.output_dir / f"{model_name}_{optimization_level.lower()}.tflite"
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        model_info = self._generate_model_info(model, tflite_model, output_path)
        self._save_model_info(model_name, optimization_level, model_info)
        logger.info(f"Conversion succeeded: {output_path}")

        return output_path, model_info

    def _create_default_representative_dataset(self, model) -> Callable[[], Generator]:
        """Create a conservative representative dataset that respects model.input_shape"""
        input_shape = None
        try:
            input_shape = tuple(model.input_shape[1:])  # drop batch
        except Exception:
            # fallback to a typical image input size
            input_shape = (96, 96, 3)

        def _repr():
            for _ in range(100):
                if len(input_shape) == 3:
                    # image-like
                    arr = np.random.randint(0, 255, size=(1,) + input_shape).astype(np.float32)
                    # yield WITHOUT an extra batch dimension if the converter expects it - the converter expects list of arrays
                    yield [arr]
                else:
                    arr = np.random.randn(1, *input_shape).astype(np.float32)
                    yield [arr]

        return _repr

    def _generate_model_info(self, original_model, tflite_model, output_path: Path) -> dict:
        tflite_size = os.path.getsize(output_path)
        original_size = _safe_get_model_size(self.model_dir / f"{getattr(original_model, 'name', 'model')}.h5")
        try:
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
        except Exception as e:
            logger.warning(f"Failed to inspect TFLite model details: {e}")
            input_details = []
            output_details = []

        model_info = {
            "original_parameters": int(original_model.count_params()) if hasattr(original_model, "count_params") else 0,
            "original_size_mb": original_size / (1024 * 1024),
            "tflite_size_mb": tflite_size / (1024 * 1024),
            "compression_ratio": (original_size / tflite_size) if (original_size and tflite_size) else 0,
            "input_details": [
                {"name": d.get("name"), "shape": tuple(d.get("shape", [])), "dtype": str(d.get("dtype"))}
                for d in input_details
            ],
            "output_details": [
                {"name": d.get("name"), "shape": tuple(d.get("shape", [])), "dtype": str(d.get("dtype"))}
                for d in output_details
            ],
            "tensor_count": len(interpreter.get_tensor_details()) if input_details else 0
        }
        return model_info

    def _save_model_info(self, model_name: str, optimization_level: str, model_info: dict):
        info_dir = self.output_dir / "model_info"
        info_dir.mkdir(exist_ok=True)
        info_path = info_dir / f"{model_name}_{optimization_level.lower()}_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Model info saved: {info_path}")

    def batch_convert_models(self, model_configs: dict):
        results = {}
        for model_name, optimizations in model_configs.items():
            results[model_name] = {}
            for optimization in optimizations:
                try:
                    out_path, info = self.convert_keras_to_tflite(model_name, optimization)
                    results[model_name][optimization] = {"output_path": str(out_path), "model_info": info, "success": True}
                except Exception as e:
                    results[model_name][optimization] = {"success": False, "error": str(e)}
        self._save_batch_summary(results)
        return results

    def _save_batch_summary(self, results: dict):
        summary = {
            "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(results),
            "total_conversions": sum(len(v) for v in results.values()),
            "successful_conversions": sum(1 for v in results.values() for c in v.values() if c.get("success")),
            "models": results
        }
        summary_path = self.output_dir / "conversion_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Batch conversion summary saved: {summary_path}")

    def benchmark_tflite_model(self, model_path: Path, input_shape: Tuple[int, ...], iterations: int = 100):
        logger.info(f"Benchmarking {model_path.name} with input shape {input_shape}")
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if not input_details:
            raise RuntimeError("Model has no input details")

        # Build input_data using input_details shape/dtype (use provided input_shape as fallback)
        expected_shape = tuple(input_details[0]["shape"])
        dtype = np.dtype(input_details[0]["dtype"].name) if hasattr(input_details[0]["dtype"], "name") else np.float32
        # If shape has -1 or 0, fallback to provided input_shape
        if any(int(x) <= 0 for x in expected_shape):
            expected_shape = input_shape

        if np.issubdtype(dtype, np.integer):
            # uint8/int8 handling - use 0-255 uniform
            input_data = np.random.randint(0, 255, size=expected_shape).astype(dtype)
        else:
            input_data = np.random.randn(*expected_shape).astype(dtype)

        # Warm-up
        for _ in range(5):
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

        times_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        times_ms = np.array(times_ms)
        result = {
            "average_time_ms": float(np.mean(times_ms)),
            "std_time_ms": float(np.std(times_ms)),
            "min_time_ms": float(np.min(times_ms)),
            "max_time_ms": float(np.max(times_ms)),
            "throughput_fps": float(1000.0 / float(np.mean(times_ms))),
            "model_size_mb": os.path.getsize(model_path) / (1024 * 1024)
        }
        logger.info(f"Benchmark: {result}")
        return result

    def verify_xnnpack(self, model_path: Path) -> bool:
        """Try loading XNNPACK delegate to check compatibility — non-fatal if it fails."""
        try:
            delegate = tf.lite.load_delegate("XNNPACK")
            interpreter = tf.lite.Interpreter(model_path=str(model_path), experimental_delegates=[delegate])
            interpreter.allocate_tensors()
            logger.info("XNNPACK delegate loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"XNNPACK delegate not available or model incompatible: {e}")
            return False


def main():
    converter = TFLiteConverter()
    model_configs = {
        "face_analyzer": ["DEFAULT", "INT8", "FP16"],
        "movement_analyzer": ["DEFAULT", "INT8"],
        "fusion_engine": ["DEFAULT", "INT8"]
    }
    logger.info("Starting batch conversion")
    results = converter.batch_convert_models(model_configs)

    # Benchmark successful conversions (minimal example)
    benchmarks = {}
    for model_name, conversions in results.items():
        benchmarks[model_name] = {}
        for opt, meta in conversions.items():
            if meta.get("success"):
                try:
                    path = Path(meta["output_path"])
                    # choose fallback input shapes by model role
                    if "face" in model_name.lower():
                        input_shape = (1, 96, 96, 3)
                    elif "movement" in model_name.lower():
                        input_shape = (1, 50, 12)
                    else:
                        input_shape = (1, 64)
                    benchmarks[model_name][opt] = converter.benchmark_tflite_model(path, input_shape)
                except Exception as e:
                    logger.warning(f"Benchmark failed for {model_name} {opt}: {e}")

    with open(converter.output_dir / "benchmark_results.json", "w") as fh:
        json.dump(benchmarks, fh, indent=2)

    logger.info("Conversion + benchmarking finished")


if __name__ == "__main__":
    main()
