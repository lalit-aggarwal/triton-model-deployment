import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def execute(self, requests):
        responses = []

        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            raw = inp.as_numpy().reshape(-1)[0]

            prompt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

            tokens = self.tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **tokens,
                    max_length=256
                )

            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            out_tensor = pb_utils.Tensor(
                "OUTPUT",
                np.array([answer], dtype=object)
            )

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
