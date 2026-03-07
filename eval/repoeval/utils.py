import os
import glob
import json
import pickle

_BASE_DIR = os.path.dirname(__file__)


class CONSTANTS:
    ALL_METHODS = ["cast", "function", "declaration", "sliding"]
    REPOs = [
        "huggingface_diffusers",
        "nerfstudio-project_nerfstudio",
        "awslabs_fortuna",
        "huggingface_evaluate",
        "google_vizier",
        "alibaba_FederatedScope",
        "pytorch_rl",
        "opendilab_ACE",
    ]


class FilePathBuilder:
    python_repo_base_dir = os.path.join(_BASE_DIR, "repositories")

    @staticmethod
    def _ensure_dir(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    @staticmethod
    def repo_windows_path(repo, method, max_chunk_size):
        out = os.path.join(_BASE_DIR, "window", f"{repo}_{method}_{max_chunk_size}.jsonl")
        FilePathBuilder._ensure_dir(out)
        return out

    @staticmethod
    def query_windows_path(split, context_length, prompt_type, window_size):
        out = os.path.join(
            _BASE_DIR, "query",
            f"{split}_{context_length}_context_{prompt_type}_{window_size}.jsonl",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    @staticmethod
    def index_window_path(repo, method, max_chunk_size, model_name):
        safe = Tools.safe_model_name(model_name)
        out = os.path.join(_BASE_DIR, "index", safe, f"{repo}_{method}_{max_chunk_size}.index")
        FilePathBuilder._ensure_dir(out)
        return out

    @staticmethod
    def benchmark_path(split, context_length, prompt_type):
        return os.path.join(
            _BASE_DIR, "datasets",
            f"{split}_level_completion_{context_length}_context_{prompt_type}.test.jsonl",
        )

    @staticmethod
    def inference_corpus_path(method, max_chunk_size, model_name, split, topk):
        safe = Tools.safe_model_name(model_name)
        out = os.path.join(
            _BASE_DIR, "inference_corpus", safe,
            f"{split}_{method}_{max_chunk_size}_{topk}.jsonl",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    @staticmethod
    def code_completion_result_path(method, max_chunk_size, embed_model, llm, split, topk, max_crossfile_context=2048):
        safe_embed = Tools.safe_model_name(embed_model)
        safe_llm = Tools.safe_model_name(llm)
        out = os.path.join(
            _BASE_DIR, "completion", safe_embed, safe_llm,
            f"{split}_{method}_{max_chunk_size}_{max_crossfile_context}_{topk}.jsonl",
        )
        FilePathBuilder._ensure_dir(out)
        return out

    @staticmethod
    def output_summary_result_path(split):
        out = os.path.join(_BASE_DIR, "result", f"{split}_all_results.csv")
        FilePathBuilder._ensure_dir(out)
        return out


class Tools:
    @staticmethod
    def read_code(fname):
        with open(fname, "r", encoding="utf8") as f:
            return f.read()

    @staticmethod
    def load_pickle(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def dump_pickle(obj, fname):
        with open(fname, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def dump_jsonl(obj, fname):
        FilePathBuilder._ensure_dir(fname)
        with open(fname, "w", encoding="utf8") as f:
            for item in obj:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def load_jsonl(fname):
        with open(fname, "r", encoding="utf8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def iterate_repository(repo, language="python"):
        base_dir = FilePathBuilder.python_repo_base_dir
        ext = {"python": "*.py", "rust": "*.rs"}.get(language)
        if ext is None:
            raise ValueError(f"Unsupported language: {language}")

        pattern = os.path.join(base_dir, repo, "**", ext)
        files = [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]

        base_parts = os.path.normpath(base_dir).split(os.sep)
        skipped = []
        loaded = {}
        for fname in files:
            try:
                code = Tools.read_code(fname)
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_parts):])
                loaded[fpath_tuple] = code
            except Exception as e:
                skipped.append((fname, e))

        if skipped:
            print(f"Skipped {len(skipped)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped:
                print(f"  {fname}: {e}")
        return loaded

    @staticmethod
    def safe_model_name(model_name):
        return model_name.split("/")[-1]

    @staticmethod
    def scan_completion_directory():
        completion_dir = os.path.join(_BASE_DIR, "completion")
        combinations = []
        if not os.path.exists(completion_dir):
            return combinations

        for retriever in os.listdir(completion_dir):
            retriever_path = os.path.join(completion_dir, retriever)
            if not os.path.isdir(retriever_path) or retriever.startswith("."):
                continue
            for llm in os.listdir(retriever_path):
                llm_path = os.path.join(retriever_path, llm)
                if not os.path.isdir(llm_path) or llm.startswith("."):
                    continue
                combinations.append((retriever, llm))
        return combinations


def is_context_file(window, context_path, query_line):
    """Check if a window's content comes from code after the query hole.

    Returns True if ALL metadata entries are from the same file AND after the hole
    (meaning the window should be filtered out).
    """
    context_is_not_after_hole = []
    for metadata in window["metadata"]:
        if metadata["fpath_tuple"] != context_path:
            context_is_not_after_hole.append(True)
            continue
        if metadata["end_line_no"] <= query_line["metadata"]["context_start_lineno"]:
            context_is_not_after_hole.append(True)
            continue
        context_is_not_after_hole.append(False)
    return not any(context_is_not_after_hole)
